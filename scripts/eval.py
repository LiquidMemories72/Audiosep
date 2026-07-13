import os
import sys
import torch
import speechbrain as sb
from speechbrain.utils.logger import get_logger
from hyperpyyaml import load_hyperpyyaml

# Workaround for SpeechBrain k2 lazy import bug with torch._dynamo
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
            # est_sources is a list of [B, L]
            est_sources, target_tensors, mix = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives((est_sources, target_tensors, mix), targets, stage)
            
            # Here we could compute metrics like SI-SDR using speechbrain or torchmetrics
            # For simplicity, we just use the OR-PIT SNR loss which is our primary metric
            # Note: OR-PIT loss is negative SNR, so lower is better.
            
        return loss.mean().detach()

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Data preparation
    train_data, valid_data, test_data = dataio_prep(hparams)

    # Brain class initialization
    evaluator = DExFormerEvaluator(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams.get("checkpointer"),
    )

    logger.info("Starting evaluation on the test set...")
    
    # We use evaluate() which automatically loads the best model from the checkpointer
    evaluator.evaluate(test_data, min_key="OR-PIT_loss")
