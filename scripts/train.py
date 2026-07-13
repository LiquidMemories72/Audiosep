import os
import sys
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
        
        # Run DExFormer extraction loop
        est_sources = self.modules.dexformer.extract_all(
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
        
        targets = [batch.s1_sig, batch.s2_sig]
        if num_spks == 3:
            targets.append(batch.s3_sig)
            
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
            torchaudio.save(os.path.join(save_dir, f"{step_str}_gt_spk1.wav"), targets[0][0][0].squeeze().unsqueeze(0).cpu(), 16000)
            torchaudio.save(os.path.join(save_dir, f"{step_str}_gt_spk2.wav"), targets[1][0][0].squeeze().unsqueeze(0).cpu(), 16000)
            if num_spks == 3:
                torchaudio.save(os.path.join(save_dir, f"{step_str}_gt_spk3.wav"), targets[2][0][0].squeeze().unsqueeze(0).cpu(), 16000)
            
            # Save estimations
            # predictions is exactly est_sources (the list of speaker tensors)
            est_sources = predictions
            
            torchaudio.save(os.path.join(save_dir, f"{step_str}_est_spk1.wav"), est_sources[0][0].detach().squeeze().unsqueeze(0).cpu(), 16000)
            torchaudio.save(os.path.join(save_dir, f"{step_str}_est_spk2.wav"), est_sources[1][0].detach().squeeze().unsqueeze(0).cpu(), 16000)
            if num_spks == 3:
                torchaudio.save(os.path.join(save_dir, f"{step_str}_est_spk3.wav"), est_sources[2][0].detach().squeeze().unsqueeze(0).cpu(), 16000)
                
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
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

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

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = sb.dataio.dataio.read_audio(s1_wav)
        return s1_sig[:max_len]

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        s2_sig = sb.dataio.dataio.read_audio(s2_wav)
        return s2_sig[:max_len]

    if hparams["num_spks"] == 3:
        @sb.utils.data_pipeline.takes("s3_wav")
        @sb.utils.data_pipeline.provides("s3_sig")
        def audio_pipeline_s3(s3_wav):
            s3_sig = sb.dataio.dataio.read_audio(s3_wav)
            return s3_sig[:max_len]

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    
    if hparams["num_spks"] == 3:
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s3)
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig"]
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
        )

    return train_data, valid_data, test_data


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

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
