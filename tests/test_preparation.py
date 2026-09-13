"""Synthetic-only preprocessing checks; no downloads or study observations."""

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_inputs import load_pinned_ushcn, write_dataset
from preprocessing import ENTITY_SPLIT_SEED, load_jena_weather, prepare_ushcn, prepare_weather


class PreparationTest(unittest.TestCase):
    def test_weather(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "synthetic.zip"
            text = "Date Time,a,b\n01.01.2000 00:00:00,1,-9999\n01.01.2000 00:00:00,3,8\n01.01.2000 00:20:00,6,8\n01.01.2000 03:10:00,100000,100000\n"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("synthetic.csv", text)
            record = load_jena_weather([archive])
            self.assertEqual(len(record[1]), 20)
            self.assertEqual(record[2][0].tolist(), [2.0, 8.0])
            self.assertFalse(record[3][1].any())
            splits, scaler = prepare_weather(record)
            self.assertEqual(set(splits), {"train", "val"})
            self.assertEqual(len(splits["train"][1]), 14)
            self.assertEqual(splits["val"][1].tolist(), [14.0, 15.0])
            np.testing.assert_array_equal(scaler.mean, [4.0, 8.0])
            np.testing.assert_array_equal(scaler.scale, [2.0, 1.0])
            np.testing.assert_array_equal(scaler.count, [2, 2])
            self.assertEqual(splits["train"][2][0].tolist(), [-1.0, 0.0])
            self.assertFalse(splits["train"][2][1].any())
            write_dataset(root, "weather", splits, scaler, ())
            self.assertEqual({p.name for p in (root / "weather").iterdir()}, {"train.pt", "val.pt", "scaler.npz", "preparation.json"})
            saved = torch.load(root / "weather/val.pt", weights_only=True)
            for tensor in saved[1:]:
                self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())
            receipt = json.loads((root / "weather/preparation.json").read_text())
            self.assertFalse(receipt["historical_byte_identity_verified"])
            with self.assertRaises(FileExistsError):
                write_dataset(root, "weather", splits, scaler, ())

    def test_ushcn(self):
        records = [(i, torch.tensor([0.0, 1.0]), torch.tensor([[float(i), 7.0], [float(i + 1), 7.0]]), torch.ones(2, 2, dtype=torch.bool)) for i in [11, 2, 4, 10, 3, 9, 5, 1, 8, 6]]
        ids = sorted(str(record[0]) for record in records)
        selected = [ids[i] for i in np.random.default_rng(ENTITY_SPLIT_SEED).permutation(len(ids))]
        splits, scaler = prepare_ushcn(records)
        self.assertEqual([str(r[0]) for r in splits["train"]], selected[:6])
        self.assertEqual([str(r[0]) for r in splits["val"]], selected[6:8])
        by_id = {str(r[0]): r for r in records}
        expected = np.concatenate([by_id[r][2].numpy() for r in selected[:6]]).astype(np.float64)
        np.testing.assert_allclose(scaler.mean, expected.mean(axis=0), rtol=0, atol=1e-14)
        np.testing.assert_allclose(scaler.scale, [expected[:, 0].std(), 1.0], rtol=0, atol=1e-14)
        changed = [(r[0], r[1], r[2] + (100000 if str(r[0]) not in selected[:6] else 0), r[3]) for r in records]
        _, unchanged_scaler = prepare_ushcn(changed)
        np.testing.assert_array_equal(scaler.mean, unchanged_scaler.mean)
        np.testing.assert_array_equal(scaler.scale, unchanged_scaler.scale)
        with self.assertRaises(ValueError):
            prepare_ushcn(records + [records[0]])

    def test_untrusted_pickle_not_loaded(self):
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp) / "untrusted.pt"
            bad.write_bytes(b"not the registered upstream input")
            with patch("prepare_inputs.torch.load") as loader:
                with self.assertRaises(ValueError):
                    load_pinned_ushcn(bad)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
