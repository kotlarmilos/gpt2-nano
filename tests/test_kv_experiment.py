from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import unittest

import torch

from src.kv_cache_study import allocate_policy_greedy, run_study
from src.kv_quant import (
    CachePolicy,
    QuantizedCacheGPT,
    QuantizedHeadTensor,
    _INT4_MAX,
    _INT8_MAX,
    dequantize_sym_int4,
    dequantize_sym_int8,
    quantize_sym_int4,
    quantize_sym_int8,
)
from src.model import GPT, GPTConfig


ROOT = Path(__file__).resolve().parents[1]


def _tiny_model(seed: int = 7) -> GPT:
    torch.manual_seed(seed)
    return GPT(
        GPTConfig(
            vocab_size=32,
            context_len=16,
            embedding_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
    ).eval()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PackedQuantizationTests(unittest.TestCase):
    def test_int8_roundtrip_and_byte_accounting(self) -> None:
        torch.manual_seed(11)
        values = torch.randn(2, 3, 8)
        quantized, scales = quantize_sym_int8(values)
        restored = dequantize_sym_int8(quantized, scales, torch.float32)

        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(scales.dtype, torch.float16)
        self.assertGreaterEqual(int(quantized.min()), -_INT8_MAX)
        self.assertLessEqual(int(quantized.max()), _INT8_MAX)
        step = float(scales.float().max()) / _INT8_MAX
        self.assertLessEqual(float((restored - values).abs().max()), step + 1e-4)

        stored = QuantizedHeadTensor.from_float(values, 8)
        self.assertEqual(stored.payload_nbytes(), values.numel())
        self.assertEqual(stored.scale_nbytes(), 2 * 3 * 2)

    def test_int4_packing_roundtrip_and_byte_accounting(self) -> None:
        for head_dim in (7, 8):
            with self.subTest(head_dim=head_dim):
                torch.manual_seed(20 + head_dim)
                values = torch.randn(2, 3, head_dim)
                packed, scales = quantize_sym_int4(values)
                restored = dequantize_sym_int4(
                    packed,
                    scales,
                    head_dim,
                    torch.float32,
                )

                packed_dim = (head_dim + 1) // 2
                self.assertEqual(packed.shape, (2, 3, packed_dim))
                self.assertEqual(packed.dtype, torch.uint8)
                self.assertEqual(scales.dtype, torch.float16)
                step = float(scales.float().max()) / _INT4_MAX
                self.assertLessEqual(
                    float((restored - values).abs().max()),
                    step + 1e-4,
                )

                stored = QuantizedHeadTensor.from_float(values, 4)
                self.assertEqual(stored.payload_nbytes(), 2 * 3 * packed_dim)
                self.assertEqual(stored.scale_nbytes(), 2 * 3 * 2)


class IncrementalKVTests(unittest.TestCase):
    def test_model_dtype_cache_matches_full_forward_and_appends(self) -> None:
        model = _tiny_model(seed=31)
        tokens = torch.randint(0, model.config.vocab_size, (1, 8))
        expected = model(tokens)
        adapter = QuantizedCacheGPT(
            model,
            CachePolicy.all_sixteen(
                model.config.num_layers,
                model.config.num_heads,
            ),
        )

        cache = None
        incremental = []
        for position in range(tokens.size(1)):
            logits, cache = adapter(
                tokens[:, position : position + 1],
                past_kv_cache=cache,
                use_cache=True,
            )
            incremental.append(logits)
            self.assertEqual(len(cache), model.config.num_layers)
            for layer in cache:
                self.assertTrue(
                    all(head.payload.shape[1] == position + 1 for head in layer.keys)
                )

        torch.testing.assert_close(
            torch.cat(incremental, dim=1),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

class AllocationAndEvidenceTests(unittest.TestCase):
    def test_allocator_respects_proxy_budget_and_transition_order(self) -> None:
        result = allocate_policy_greedy(
            [[0.04, 0.05]],
            [[0.06, 0.50]],
            kl_budget=0.07,
            num_layers=1,
            num_heads=2,
            head_dim=8,
            model_dtype_bytes=2,
        )

        self.assertEqual(result["policy"].bits_per_layer[0], (4, 16))
        self.assertAlmostEqual(result["predicted_kl"], 0.06)
        self.assertLessEqual(result["predicted_kl"], 0.07)
        self.assertEqual(
            [(step["from_bits"], step["to_bits"]) for step in result["steps"]],
            [(16, 8), (8, 4)],
        )
        self.assertTrue(
            all(step["cumulative_kl"] <= 0.07 for step in result["steps"])
        )

    def test_published_policy_and_negative_adaptation_evidence(self) -> None:
        payload = json.loads(
            (ROOT / "artifacts/kv-cache-publication/results.json").read_text()
        )
        policy = next(
            item
            for item in payload["policies"]
            if item["label"] == "greedy_kl_budget_0.01"
        )
        bits = [bit for layer in policy["bits_per_layer"] for bit in layer]
        self.assertEqual(
            {bit: bits.count(bit) for bit in (16, 8, 4)},
            {16: 7, 8: 19, 4: 70},
        )
        self.assertLessEqual(policy["predicted_kl"], 0.01)

        model_config = payload["model_config"]
        computed_bytes = CachePolicy(
            bits_per_layer=tuple(
                tuple(layer) for layer in policy["bits_per_layer"]
            )
        ).exact_cache_bytes(
            batch=1,
            seq_len=payload["splits"]["seq_len"],
            head_dim=model_config["embedding_dim"] // model_config["num_heads"],
            model_dtype_bytes=2,
        )
        self.assertEqual(computed_bytes, policy["cache_bytes"])
        self.assertEqual(computed_bytes["total_bytes"], 579_840)

        validation = next(
            item
            for item in payload["validation"]["results"]
            if item["policy_label"] == "greedy_kl_budget_0.01"
        )
        self.assertAlmostEqual(validation["mean_kl"], 0.0015279437648132443)
        self.assertEqual(validation["token_agreement_rate"], 0.978515625)

        seeds = payload["noise_fine_tuning"]["seeds"]
        self.assertEqual([item["seed"] for item in seeds], [1337, 2026, 9001])
        self.assertAlmostEqual(
            statistics.fmean(item["validation"]["mean_kl"] for item in seeds),
            0.0015153666026890278,
        )
        self.assertAlmostEqual(
            statistics.fmean(
                item["validation"]["token_agreement_rate"] for item in seeds
            ),
            0.9781901041666666,
        )


class SplitAndProvenanceTests(unittest.TestCase):
    def test_publication_hashes_and_splits_match_manifest(self) -> None:
        manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
        record = next(
            item
            for item in manifest["artifacts"]
            if item["path"] == "artifacts/kv-cache-publication/results.json"
        )
        results_path = ROOT / record["path"]
        config_path = ROOT / record["config"]
        payload = json.loads(results_path.read_text())

        self.assertEqual(_sha256(results_path), record["sha256"])
        self.assertEqual(_sha256(config_path), record["config_sha256"])
        self.assertEqual(
            payload["hashes"]["config_sha256"],
            record["config_sha256"],
        )
        self.assertEqual(
            payload["hashes"]["checkpoint_sha256"],
            record["checkpoint_sha256"],
        )
        self.assertEqual(
            payload["hashes"]["corpus_sha256"],
            record["corpus_sha256"],
        )
        self.assertFalse(payload["hashes"]["git_dirty"])
        self.assertEqual(payload["hashes"]["git_state_captured"], "study_start")

        splits = payload["splits"]
        self.assertTrue(
            splits["calib_start"]
            < splits["calib_end"]
            <= splits["noise_train_start"]
            < splits["noise_train_end"]
            <= splits["noise_selection_start"]
            < splits["noise_selection_end"]
            <= splits["val_start"]
            < splits["val_end"]
        )
        self.assertFalse(splits["overlap"])

    def test_study_rejects_overlapping_calibration_and_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlaps validation"):
            run_study(
                {
                    "device": "cpu",
                    "dtype": "float32",
                    "calib_start": 0,
                    "calib_end": 12,
                    "val_start": 8,
                    "val_end": 16,
                    "seq_len": 4,
                }
            )


class EndToEndSmokeTests(unittest.TestCase):
    def test_tiny_cpu_study_emits_core_experiment_fields(self) -> None:
        payload = run_study(
            {
                "device": "cpu",
                "dtype": "float32",
                "seed": 53,
                "calib_start": 0,
                "calib_end": 8,
                "val_start": 8,
                "val_end": 16,
                "seq_len": 4,
                "kl_budgets": [0.01],
                "time_sequences": 0,
                "model_config": {
                    "vocab_size": 23,
                    "context_len": 8,
                    "embedding_dim": 8,
                    "num_layers": 1,
                    "num_heads": 2,
                    "dropout": 0.0,
                },
            }
        )

        self.assertEqual(payload["status"], "measured")
        self.assertFalse(payload["splits"]["overlap"])
        self.assertIn("single-head", payload["calibration_score_definition"])
        results = {
            item["policy_label"]: item
            for item in payload["validation"]["results"]
        }
        self.assertEqual(
            set(results),
            {
                "uniform_int8",
                "uniform_int4",
                "all_fp32",
                "greedy_kl_budget_0.01",
            },
        )
        for item in results.values():
            self.assertIn("model_dtype_cache_bytes", item)
            self.assertIn("compression_factor_vs_model_dtype", item)


if __name__ == "__main__":
    unittest.main()
