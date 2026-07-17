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
    def evaluate_batch(self, batch, stage):
        """
        Computations needed for validation/test batches during evaluation.
        We compute SI-SDR and SDR metrics here.
        """
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)
        with torch.no_grad():
            est_sources, target_tensors, mix = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives((est_sources, target_tensors, mix), targets, stage)
        return loss.mean().detach()
if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    train_data, valid_data, test_data = dataio_prep(hparams)
    evaluator = DExFormerEvaluator(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams.get("checkpointer"),
    )
    logger.info("Starting evaluation on the test set...")
    evaluator.evaluate(test_data, min_key="OR-PIT_loss")