import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import speechbrain as sb
from speechbrain.utils.logger import get_logger
from hyperpyyaml import load_hyperpyyaml
import torch.optim as optim
try:
    import torch._dynamo
except ImportError:
    pass
import speechbrain.utils.importutils
for k in list(sys.modules.keys()):
    if 'k2_fsa' in k or 'k2' == k:
        sys.modules.pop(k, None)
from dexformer.models.dexformer import DExFormer
from train import DExFormerSeparation, dataio_prep
logger = get_logger(__name__)
class DExFormerEvaluator(DExFormerSeparation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_sisnri = []

    def compute_forward(self, mix, targets, stage, noise=None):
        """
        Override compute_forward to force exactly num_spks extractions.
        """
        mix, mix_lens = mix
        mix = mix.to(self.device)
        num_spks = self.hparams.num_spks
        target_tensors = []
        for i in range(num_spks):
            t = targets[i][0].to(self.device)
            target_tensors.append(t)
            
        # Passing training=True forces it to extract exactly num_spks times
        # without using the Sequence Termination Criterion (STC) early stopping.
        # It does NOT affect nn.Module.eval() mode (dropout, batchnorm, etc. remain in eval mode).
        est_sources = self.modules.dexformer(
            mix, 
            num_speakers=num_spks, 
            training=True
        )
        return est_sources, target_tensors, mix

    def evaluate_batch(self, batch, stage):
        """
        Computations needed for validation/test batches during evaluation.
        We compute SI-SDR and SDR metrics here.
        """
        mixture = batch.mix_sig
        targets = [getattr(batch, f"s{i}_sig") for i in range(1, self.hparams.num_spks + 1)]
        with torch.no_grad():
            est_sources, target_tensors, mix = self.compute_forward(mixture, targets, stage)
            
            # Ensure est_sources has exactly num_spks elements to match targets
            est_sources_eval = est_sources[:self.hparams.num_spks]
            while len(est_sources_eval) < self.hparams.num_spks:
                est_sources_eval.append(torch.zeros_like(mix))
                
            loss = self.compute_objectives((est_sources_eval, target_tensors, mix), targets, stage)
            
            if stage == sb.Stage.TEST:
                from speechbrain.nnet.losses import get_si_snr_with_pitwrapper
                preds = torch.stack(est_sources_eval, dim=-1)
                trgts = torch.stack(target_tensors, dim=-1)
                
                # The function signature is get_si_snr_with_pitwrapper(source, estimate_source)
                pure_snr_loss = get_si_snr_with_pitwrapper(trgts, preds)
                
                mix_expanded = mix.unsqueeze(-1).expand_as(trgts)
                base_snr_loss = get_si_snr_with_pitwrapper(trgts, mix_expanded)
                
                sisnri = base_snr_loss - pure_snr_loss
                self.test_sisnri.extend(sisnri.tolist())
                
        return loss.mean().detach()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TEST:
            snr_db = -stage_loss
            sisnri_db = sum(self.test_sisnri) / len(self.test_sisnri) if self.test_sisnri else 0.0
            print(f"Test Evaluation Complete! OR-PIT Loss: {stage_loss:.4f} | Estimated Output SNR: {snr_db:.2f} dB | SI-SNRi: {sisnri_db:.2f} dB")
            self.test_sisnri = []
if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    if not hasattr(DExFormer, "forward") or DExFormer.forward == torch.nn.Module.forward:
        DExFormer.forward = DExFormer.extract_all
    train_data, valid_data, test_data = dataio_prep(hparams)
    if "checkpointer" in hparams:
        checkpointer = hparams["checkpointer"]
        if "optimizer" in checkpointer.recoverables:
            del checkpointer.recoverables["optimizer"]

    evaluator = DExFormerEvaluator(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams.get("checkpointer"),
    )
    
    if "checkpointer" in hparams:
        ckpt = hparams["checkpointer"].find_checkpoint(min_key="OR-PIT_loss")
        if ckpt is not None:
            print(f"✅ Loaded checkpoint from: {ckpt.path}")
        else:
            print("❌ WARNING: No checkpoint found! Evaluating an UNTRAINED, randomly initialized model.")
            
    print("Starting evaluation on the test set...")
    evaluator.evaluate(test_data, min_key="OR-PIT_loss")