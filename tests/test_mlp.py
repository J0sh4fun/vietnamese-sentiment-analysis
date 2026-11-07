import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from src.mlp import TfidfDataset, SentimentMLP, evaluate, make_loader, load_checkpoint
from src.run_mlp import fit_features, fit_mlp, load_splits


class MLPTests(unittest.TestCase):
    def test_validation_only_tokens_do_not_enter_vocabulary(self):
        vectorizer, train, val = fit_features(["tốt tốt", "tốt đẹp", "đẹp đẹp"], ["chỉ_validation"])
        self.assertNotIn("chỉ_validation", vectorizer.vocabulary_)
        self.assertEqual(val.nnz, 0)
        self.assertEqual(train.dtype, np.float32)

    def test_source_overlap_and_invalid_label_fail_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = pd.DataFrame({"source_id": ["a", "b"], "clean_text": ["tốt", "tệ"],
                                  "label": ["positive", "negative"]})
            val = pd.DataFrame({"source_id": ["c", "d"], "clean_text": ["đẹp", "hỏng"],
                                "label": ["positive", "negative"]})
            train.to_csv(root / "train_split.csv", index=False)
            val.to_csv(root / "validation_split.csv", index=False)
            self.assertEqual(len(load_splits(root, 2, 2)[0]), 2)
            val.loc[0, "source_id"] = "a"
            val.to_csv(root / "validation_split.csv", index=False)
            with self.assertRaisesRegex(ValueError, "source_id overlap"):
                load_splits(root, 2, 2)
            val.loc[0, "source_id"] = "c"
            val.loc[0, "label"] = "neutral"
            val.to_csv(root / "validation_split.csv", index=False)
            with self.assertRaisesRegex(ValueError, "both labels"):
                load_splits(root, 2, 2)

    def test_sparse_dataset_batch_and_eval_disable_dropout_and_gradients(self):
        dataset = TfidfDataset(csr_matrix(np.eye(4)), [0, 1, 0, 1])
        features, labels = next(iter(make_loader(dataset, 3)))
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(features.shape, (3, 4))
        model = SentimentMLP(4, dropout=0.3)
        seen = []
        hook = model.register_forward_hook(lambda module, inputs, output: seen.append(
            (module.training, torch.is_grad_enabled(), tuple(output.shape))))
        first = evaluate(model, make_loader(dataset, 3))
        second = evaluate(model, make_loader(dataset, 3))
        hook.remove()
        self.assertEqual(first, second)
        self.assertTrue(all(not training and not grad for training, grad, shape in seen))
        self.assertEqual(seen[0][2], (3, 2))

    def test_checkpoint_keeps_best_epoch_instead_of_final_epoch(self):
        torch.set_num_threads(1)
        dataset = TfidfDataset(csr_matrix(np.eye(4)), [0, 1, 0, 1])
        calls, states = [0], []

        def fake_evaluate(model, loader):
            calls[0] += 1
            # Calls 1/2 are train/validation at epoch 1; calls 3/4 at epoch 2.
            if calls[0] in (2, 4):
                states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            score = 0.5 if calls[0] == 4 else 1.0
            return {"loss": 0.1, "accuracy": score, "macro_f1": score, "predictions": [0, 1, 0, 1]}

        with tempfile.TemporaryDirectory() as temp, patch("src.run_mlp.evaluate", side_effect=fake_evaluate):
            with contextlib.redirect_stdout(io.StringIO()):
                summary, _, _ = fit_mlp({"name": "tiny", "learning_rate": 0.001, "dropout": 0},
                                        dataset, dataset, Path(temp), 42, 2, 4)
            model, checkpoint = load_checkpoint(Path(temp) / "tiny_best.pt")
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertEqual(checkpoint["artifact_version"], 2)
            self.assertEqual(next(iter(checkpoint["optimizer_state_dict"]["state"].values()))["step"].item(), 1)
            self.assertTrue(any(not torch.equal(states[0][key], states[1][key]) for key in states[0]))
            self.assertTrue(all(torch.equal(model.state_dict()[key], states[0][key]) for key in states[0]))


if __name__ == "__main__":
    unittest.main()
