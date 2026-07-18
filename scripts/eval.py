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

    def evaluate_batch(self, batch, stage):
        """
        Computations needed for validation/test batches during evaluation.
        We compute SI-SDR and SDR metrics here.
        """
        mixture = batch.mix_sig
        targets = [getattr(batch, f"s{i}_sig") for i in range(1, self.hparams.num_spks + 1)]
        with torch.no_grad():
            est_sources, target_tensors, mix = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives((est_sources, target_tensors, mix), targets, stage)
            
            if stage == sb.Stage.TEST:
                from speechbrain.nnet.losses import get_snr_loss, PitWrapper
                pit_snr = PitWrapper(get_snr_loss)
                preds = torch.stack(est_sources, dim=-1)
                trgts = torch.stack(target_tensors, dim=-1)
                pure_snr_loss = pit_snr(preds, trgts)
                
                mix_expanded = mix.unsqueeze(-1).expand_as(trgts)
                base_snr_loss = pit_snr(mix_expanded, trgts)
                
                sisnri = base_snr_loss - pure_snr_loss
                self.test_sisnri.extend(sisnri.tolist())
                
        return loss.mean().detach()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TEST:
            snr_db = -stage_loss
            sisnri_db = sum(self.test_sisnri) / len(self.test_sisnri) if self.test_sisnri else 0.0
            logger.info(f"Test Evaluation Complete! OR-PIT Loss: {stage_loss:.4f} | Estimated Output SNR: {snr_db:.2f} dB | SI-SNRi: {sisnri_db:.2f} dB")
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
    logger.info("Starting evaluation on the test set...")
    evaluator.evaluate(test_data, min_key="OR-PIT_loss")