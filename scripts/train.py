import os
import sys

# Ensure the repo root is on sys.path so `dexformer` is importable when
# torchrun spawns worker processes from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim  # noqa: required before speechbrain to avoid k2 lazy-import crash

# Workaround: pre-import torch._dynamo so inspect.getmodule works during SpeechBrain init.
# Without this, SpeechBrain's lazy k2 importer triggers torch._dynamo which triggers
# inspect.getframeinfo on a lazy module that has no __file__, causing an ImportError.
try:
    import torch._dynamo  # noqa
except Exception:
    pass

import speechbrain as sb
from speechbrain.utils.logger import get_logger
from hyperpyyaml import load_hyperpyyaml

from dexformer.models.dexformer import DExFormer
from dexformer.losses.or_pit import compute_or_pit_loss

logger = get_logger(__name__)

class DExFormerSeparation(sb.Brain):
    def compute_forward(self, mix, targets, stage, noise=None):
        """
        Forward pass for DExFormer training.
        """
        mix, mix_lens = mix
        mix = mix.to(self.device)
        
        # In DExFormer training, we need num_speakers
        num_spks = self.hparams.num_spks
        
        # Targets are expected as a list of tensors per speaker or a single tensor [B, L, num_spks]
        # SpeechBrain typically provides targets as a list where targets[i] is a tuple of (tensor, len)
        target_tensors = []
        for i in range(num_spks):
            t = targets[i][0].to(self.device)
            target_tensors.append(t)
            
        # Optional: Apply augmentations (speed perturb, dropping, etc) if stage == sb.Stage.TRAIN
        
        # Run DExFormer extraction loop via __call__ so DDP can hook into forward()
        est_sources = self.modules.dexformer(
            mix, 
            num_speakers=num_spks, 
            training=(stage == sb.Stage.TRAIN)
        )
        
        return est_sources, target_tensors, mix

    def compute_objectives(self, predictions, targets, stage):
        """
        Compute OR-PIT loss.
        """
        est_sources, target_tensors, mix = predictions
        
        loss = compute_or_pit_loss(est_sources, target_tensors, mix)
        
        return loss

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        mixture = batch.mix_sig
        
        # Determine number of speakers
        num_spks = self.hparams.num_spks
        
        targets = [getattr(batch, f"s{i}_sig") for i in range(1, num_spks + 1)]
            
        noise = None # Not using WHAM noise in this script for simplicity

        with self.training_ctx:
            predictions, targets_unpacked, mix_unpacked = self.compute_forward(
                mixture, targets, sb.Stage.TRAIN, noise
            )
            loss = self.compute_objectives((predictions, targets_unpacked, mix_unpacked), targets, sb.Stage.TRAIN)

        if loss.nelement() > 0:
            self.scaler.scale(loss).backward()
            
            # Gradient Accumulation
            # SpeechBrain's check_gradients() handles accumulation properly if step%grad_accum == 0
            if self.step % self.hparams.gradient_accumulation_steps == 0:
                if getattr(self.hparams, "clip_grad_norm", -1) > 0:
                    self.scaler.unscale_(self.optimizer)
                    
                    # Track unclipped norm for logging
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(),
                        self.hparams.clip_grad_norm,
                    )
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
            logger.info("Empty loss! Skipping this batch")
            loss.data = torch.tensor(0.0).to(self.device)

        # Overfit audio saving every epoch (since we have 10 batches per epoch, we save every 10 steps)
        if self.step % 10 == 0:
            import torchaudio
            import os
            save_dir = os.path.join(self.hparams.output_folder, "overfit_audio")
            os.makedirs(save_dir, exist_ok=True)
            
            step_str = f"step_{self.step:04d}"
            
            # Save mixture and GT (only for the first element in batch)
            torchaudio.save(os.path.join(save_dir, f"{step_str}_mix.wav"), mixture[0].squeeze().unsqueeze(0).cpu(), 16000)
            for i, t in enumerate(targets):
                torchaudio.save(os.path.join(save_dir, f"{step_str}_gt_spk{i+1}.wav"), t[0][0].squeeze().unsqueeze(0).cpu(), 16000)
            
            # Save estimations
            # predictions is exactly est_sources (the list of speaker tensors)
            est_sources = predictions
            
            for i, s in enumerate(est_sources):
                torchaudio.save(os.path.join(save_dir, f"{step_str}_est_spk{i+1}.wav"), s[0].detach().squeeze().unsqueeze(0).cpu(), 16000)
                
            # Note: saving residuals is tricky because they are computed inside extract_all and not returned.
            # For this quick experiment, we will just save the final sources.
            
            # Log metrics
            mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            norm_val = total_norm.item() if 'total_norm' in locals() else 0.0
            logger.info(f"[Step {self.step}] Loss: {loss.item():.4f} | Grad Norm: {norm_val:.4f} | Peak Mem: {mem_mb:.1f} MB")

        return loss.detach().cpu()
        
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        mixture = batch.mix_sig
        targets = [getattr(batch, f"s{i}_sig") for i in range(1, self.hparams.num_spks + 1)]

        with torch.no_grad():
            predictions, targets_unpacked, mix_unpacked = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives((predictions, targets_unpacked, mix_unpacked), targets, stage)

        return loss.mean().detach()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        Logging at the end of epoch.
        """
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            
        if stage == sb.Stage.VALID:
            logger.info(f"Epoch {epoch}: Train Loss (OR-PIT SNR) = {self.train_loss:.4f} | Valid Loss = {stage_loss:.4f}")
            if hasattr(self.hparams, "checkpointer"):
                self.hparams.checkpointer.save_and_keep_only(
                    meta={"OR-PIT_loss": stage_loss}, min_keys=["OR-PIT_loss"]
                )

def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # Max samples to load (3 sec @ 16kHz). Prevents huge autograd graphs.
    max_len = int(hparams.get("max_audio_length", 3.0) * hparams.get("sample_rate", 16000))

    # 2. Provide audio pipelines
    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)
        return mix_sig[:max_len]

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)

    # Dynamically register one audio pipeline per speaker (s1_wav -> s1_sig, ..., sN_wav -> sN_sig).
    # A factory function is used to capture `wav_key` / `sig_key` by value, avoiding the classic
    # loop-closure pitfall where all lambdas would capture the same (final) loop variable.
    def make_audio_pipeline(wav_key, sig_key, max_samples):
        @sb.utils.data_pipeline.takes(wav_key)
        @sb.utils.data_pipeline.provides(sig_key)
        def _pipeline(wav_path):
            return sb.dataio.dataio.read_audio(wav_path)[:max_samples]
        return _pipeline

    for i in range(1, hparams["num_spks"] + 1):
        pipeline = make_audio_pipeline(f"s{i}_wav", f"s{i}_sig", max_len)
        sb.dataio.dataset.add_dynamic_item(datasets, pipeline)

    output_keys = ["id", "mix_sig"] + [f"s{i}_sig" for i in range(1, hparams["num_spks"] + 1)]
    sb.dataio.dataset.set_output_keys(datasets, output_keys)

    return train_data, valid_data, test_data


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Auto-detect torchrun (DDP) to trigger SpeechBrain's native distributed_launch
    if "LOCAL_RANK" in os.environ:
        run_opts["distributed_launch"] = True
        if "distributed_backend" not in run_opts:
            run_opts["distributed_backend"] = "nccl"

    # Alias DExFormer.forward to extract_all so PyTorch DDP can synchronize gradients correctly
    if not hasattr(DExFormer, "forward") or DExFormer.forward == torch.nn.Module.forward:
        DExFormer.forward = DExFormer.extract_all

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation
    train_data, valid_data, test_data = dataio_prep(hparams)

    # Brain class initialization
    separator = DExFormerSeparation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams.get("checkpointer"),
    )

    # Training
    separator.fit(
        separator.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Eval
    separator.evaluate(test_data, min_key="OR-PIT_loss")
