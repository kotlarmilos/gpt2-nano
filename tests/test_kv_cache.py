"""Tests for KV-cache quantization (src/kv_quant.py and src/kv_cache_study.py).

All tests run on CPU and are designed to be fast (tiny models, short sequences).
"""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from src.kv_quant import (
    CachePolicy,
    NoisyBlock,
    NoisyKVGPT,
    QuantizedCacheGPT,
    QuantizedHeadTensor,
    QuantizedLayerKV,
    _INT4_MAX,
    _INT8_MAX,
    dequantize_sym_int4,
    dequantize_sym_int8,
    quantize_sym_int4,
    quantize_sym_int8,
)
from src.kv_cache_study import (
    allocate_policy_greedy,
    build_named_policies,
    compute_sensitivity_matrices,
    evaluate_policy,
    load_token_shard,
    run_study,
    teacher_forced_logits,
    quantized_incremental_logits,
    token_windows,
)
from src.model import GPT, GPTConfig
from src.objectives import categorical_kl

import numpy as np


def _tiny_model(seed: int = 7, num_heads: int = 4, num_layers: int = 2) -> GPT:
    torch.manual_seed(seed)
    cfg = GPTConfig(
        vocab_size=32,
        context_len=16,
        embedding_dim=16,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0,
    )
    return GPT(cfg).eval()


# ---------------------------------------------------------------------------
# INT4 pack / unpack
# ---------------------------------------------------------------------------


class Int4PackUnpackTests(unittest.TestCase):
    def _roundtrip(self, head_dim: int) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 5, head_dim)
        packed, scales = quantize_sym_int4(x)
        restored = dequantize_sym_int4(packed, scales, head_dim, torch.float32)
        self.assertEqual(restored.shape, (2, 5, head_dim))
        # Max error <= 0.5 * step = 0.5 * scale / _INT4_MAX (plus float16 scale precision)
        max_err = float((restored - x).abs().max().item())
        step = float(scales.float().max().item()) / _INT4_MAX
        self.assertLessEqual(max_err, step + 1e-4)

    def test_roundtrip_even_head_dim(self) -> None:
        self._roundtrip(head_dim=8)

    def test_roundtrip_odd_head_dim(self) -> None:
        self._roundtrip(head_dim=7)

    def test_roundtrip_head_dim_one(self) -> None:
        self._roundtrip(head_dim=1)

    def test_packed_shape_even(self) -> None:
        x = torch.randn(1, 3, 8)
        packed, _ = quantize_sym_int4(x)
        self.assertEqual(packed.shape, (1, 3, 4))
        self.assertEqual(packed.dtype, torch.uint8)

    def test_packed_shape_odd(self) -> None:
        x = torch.randn(1, 3, 7)
        packed, _ = quantize_sym_int4(x)
        self.assertEqual(packed.shape, (1, 3, 4))

    def test_zero_tensor_packs_to_zero(self) -> None:
        x = torch.zeros(1, 2, 6)
        packed, scales = quantize_sym_int4(x)
        restored = dequantize_sym_int4(packed, scales, 6, torch.float32)
        torch.testing.assert_close(restored, x)

    def test_boundary_values_int4(self) -> None:
        # Build tensor spanning -7 to 7 range
        vals = torch.tensor([[[float(v) for v in range(-7, 8)]]])  # (1,1,15)
        packed, scales = quantize_sym_int4(vals)
        restored = dequantize_sym_int4(packed, scales, 15, torch.float32)
        # All values should round-trip exactly
        torch.testing.assert_close(
            restored,
            vals,
            atol=scales.max().item() * 0.6,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# INT8 quantize / dequantize bounds
# ---------------------------------------------------------------------------


class Int8QuantizationTests(unittest.TestCase):
    def test_quantized_values_in_range(self) -> None:
        x = torch.randn(3, 10, 8) * 5.0
        q, scales = quantize_sym_int8(x)
        self.assertEqual(q.dtype, torch.int8)
        self.assertGreaterEqual(int(q.min().item()), -_INT8_MAX)
        self.assertLessEqual(int(q.max().item()), _INT8_MAX)

    def test_dequantize_approximate_inverse(self) -> None:
        x = torch.randn(2, 4, 8)
        q, sc = quantize_sym_int8(x)
        r = dequantize_sym_int8(q, sc, torch.float32)
        max_err = float((r - x).abs().max().item())
        # step = sc / _INT8_MAX (max_abs / 127); rounding error <= 0.5 * step
        step = float(sc.float().max().item()) / _INT8_MAX
        self.assertLessEqual(max_err, step + 1e-4)

    def test_scales_are_float16(self) -> None:
        x = torch.randn(1, 5, 4)
        _, sc = quantize_sym_int8(x)
        self.assertEqual(sc.dtype, torch.float16)

    def test_scales_shape(self) -> None:
        x = torch.randn(2, 5, 8)
        _, sc = quantize_sym_int8(x)
        self.assertEqual(sc.shape, (2, 5))


# ---------------------------------------------------------------------------
# INT4 quantize / dequantize bounds
# ---------------------------------------------------------------------------


class Int4QuantizationTests(unittest.TestCase):
    def test_quantized_nibble_values_in_range(self) -> None:
        x = torch.randn(2, 4, 8) * 3.0
        packed, scales = quantize_sym_int4(x)
        # Unpack nibbles and verify range
        lo = (packed.to(torch.int16) & 0x0F)
        hi = (packed.to(torch.int16) >> 4) & 0x0F
        lo_signed = torch.where(lo >= 8, lo - 16, lo)
        hi_signed = torch.where(hi >= 8, hi - 16, hi)
        self.assertGreaterEqual(int(lo_signed.min()), -8)
        self.assertLessEqual(int(lo_signed.max()), _INT4_MAX)
        self.assertGreaterEqual(int(hi_signed.min()), -8)
        self.assertLessEqual(int(hi_signed.max()), _INT4_MAX)

    def test_dequantize_approximate_inverse(self) -> None:
        x = torch.randn(1, 3, 6)
        packed, sc = quantize_sym_int4(x)
        r = dequantize_sym_int4(packed, sc, 6, torch.float32)
        max_err = float((r - x).abs().max().item())
        # step = sc / _INT4_MAX (max_abs / 7); rounding error <= 0.5 * step
        step = float(sc.float().max().item()) / _INT4_MAX
        self.assertLessEqual(max_err, step + 1e-4)


# ---------------------------------------------------------------------------
# Exact byte accounting
# ---------------------------------------------------------------------------


class ByteAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = 1
        self.seq_len = 4
        self.head_dim = 8
        self.num_layers = 2
        self.num_heads = 4

    def test_16bit_no_scale_bytes(self) -> None:
        x = torch.randn(self.batch, self.seq_len, self.head_dim)
        qt = QuantizedHeadTensor.from_float(x, 16)
        self.assertEqual(qt.scale_nbytes(), 0)
        expected_payload = self.batch * self.seq_len * self.head_dim * 4  # float32
        self.assertEqual(qt.payload_nbytes(), expected_payload)

    def test_8bit_scale_bytes_float16(self) -> None:
        x = torch.randn(self.batch, self.seq_len, self.head_dim)
        qt = QuantizedHeadTensor.from_float(x, 8)
        expected_scale = self.batch * self.seq_len * 2  # float16
        expected_payload = self.batch * self.seq_len * self.head_dim * 1  # int8
        self.assertEqual(qt.scale_nbytes(), expected_scale)
        self.assertEqual(qt.payload_nbytes(), expected_payload)

    def test_4bit_scale_and_packed_bytes(self) -> None:
        x = torch.randn(self.batch, self.seq_len, self.head_dim)
        qt = QuantizedHeadTensor.from_float(x, 4)
        packed_dim = (self.head_dim + 1) // 2
        expected_scale = self.batch * self.seq_len * 2
        expected_payload = self.batch * self.seq_len * packed_dim * 1
        self.assertEqual(qt.scale_nbytes(), expected_scale)
        self.assertEqual(qt.payload_nbytes(), expected_payload)

    def test_4bit_odd_head_dim_bytes(self) -> None:
        x = torch.randn(self.batch, self.seq_len, 7)
        qt = QuantizedHeadTensor.from_float(x, 4)
        packed_dim = 4  # ceil(7/2)
        self.assertEqual(qt.payload_nbytes(), self.batch * self.seq_len * packed_dim)

    def test_policy_exact_cache_bytes_16bit(self) -> None:
        policy = CachePolicy.all_sixteen(self.num_layers, self.num_heads)
        byt = policy.exact_cache_bytes(
            batch=1, seq_len=self.seq_len, head_dim=self.head_dim, model_dtype_bytes=4
        )
        expected = 2 * self.num_layers * self.num_heads * self.seq_len * self.head_dim * 4
        self.assertEqual(byt["payload_bytes"], expected)
        self.assertEqual(byt["scale_bytes"], 0)
        self.assertEqual(byt["total_bytes"], expected)

    def test_policy_exact_cache_bytes_8bit(self) -> None:
        policy = CachePolicy(
            bits_per_layer=tuple(
                tuple(8 for _ in range(self.num_heads))
                for _ in range(self.num_layers)
            )
        )
        byt = policy.exact_cache_bytes(
            batch=1, seq_len=self.seq_len, head_dim=self.head_dim, model_dtype_bytes=4
        )
        payload = 2 * self.num_layers * self.num_heads * self.seq_len * self.head_dim
        scales = 2 * self.num_layers * self.num_heads * self.seq_len * 2
        self.assertEqual(byt["payload_bytes"], payload)
        self.assertEqual(byt["scale_bytes"], scales)

    def test_policy_exact_cache_bytes_4bit(self) -> None:
        packed_dim = (self.head_dim + 1) // 2
        policy = CachePolicy(
            bits_per_layer=tuple(
                tuple(4 for _ in range(self.num_heads))
                for _ in range(self.num_layers)
            )
        )
        byt = policy.exact_cache_bytes(
            batch=1, seq_len=self.seq_len, head_dim=self.head_dim, model_dtype_bytes=4
        )
        payload = 2 * self.num_layers * self.num_heads * self.seq_len * packed_dim
        scales = 2 * self.num_layers * self.num_heads * self.seq_len * 2
        self.assertEqual(byt["payload_bytes"], payload)
        self.assertEqual(byt["scale_bytes"], scales)


# ---------------------------------------------------------------------------
# Heterogeneous per-head policies
# ---------------------------------------------------------------------------


class HeterogeneousPolicyTests(unittest.TestCase):
    def test_per_head_bits_stored_correctly(self) -> None:
        num_layers, num_heads = 2, 4
        policy = CachePolicy.all_sixteen(num_layers, num_heads).with_head(0, 1, 8)
        policy = policy.with_head(1, 3, 4)
        self.assertEqual(policy.bits_per_layer[0][0], 16)
        self.assertEqual(policy.bits_per_layer[0][1], 8)
        self.assertEqual(policy.bits_per_layer[1][3], 4)

    def test_quantized_layer_kv_roundtrip_heterogeneous(self) -> None:
        torch.manual_seed(3)
        batch, num_heads, seq_len, head_dim = 1, 4, 5, 4
        key = torch.randn(batch, num_heads, seq_len, head_dim)
        value = torch.randn(batch, num_heads, seq_len, head_dim)
        head_bits = (16, 8, 4, 16)
        lkv = QuantizedLayerKV.from_key_value(key, value, head_bits)

        k_deq, v_deq = lkv.dequantize(target_dtype=torch.float32)
        # 16-bit heads must be bit-exact
        torch.testing.assert_close(k_deq[:, 0, :, :], key[:, 0, :, :])
        torch.testing.assert_close(k_deq[:, 3, :, :], key[:, 3, :, :])
        # 8-bit and 4-bit heads are approximate
        self.assertEqual(k_deq.shape, (batch, num_heads, seq_len, head_dim))

    def test_rejects_mismatched_head_count(self) -> None:
        key = torch.randn(1, 4, 3, 4)
        value = torch.randn(1, 4, 3, 4)
        with self.assertRaises(ValueError):
            QuantizedLayerKV.from_key_value(key, value, (16, 8))  # 2 != 4

    def test_rejects_invalid_bits(self) -> None:
        with self.assertRaises(ValueError):
            CachePolicy(bits_per_layer=((7, 16),))


# ---------------------------------------------------------------------------
# 16-bit cache equivalence
# ---------------------------------------------------------------------------


class SixteenBitCacheEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _tiny_model()

    def test_all16_quantized_cache_gpt_matches_gpt_full_pass(self) -> None:
        torch.manual_seed(9)
        cfg = self.model.config
        policy = CachePolicy.all_sixteen(cfg.num_layers, cfg.num_heads)
        qmodel = QuantizedCacheGPT(self.model, policy)
        tokens = torch.randint(0, cfg.vocab_size, (1, 8))

        # Full-sequence baseline
        full_logits = self.model(tokens)

        # Incremental via QuantizedCacheGPT
        logits_list = []
        cache = None
        for pos in range(tokens.size(1)):
            tok = tokens[:, pos : pos + 1]
            out = qmodel(tok, past_kv_cache=cache, use_cache=True)
            logits, cache = out
            logits_list.append(logits)

        incremental_logits = torch.cat(logits_list, dim=1)
        torch.testing.assert_close(incremental_logits, full_logits, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Incremental cache correctness
# ---------------------------------------------------------------------------


class IncrementalCacheCorrectnessTests(unittest.TestCase):
    def test_incremental_all16_agrees_with_full_forward(self) -> None:
        model = _tiny_model(seed=13)
        cfg = model.config
        policy = CachePolicy.all_sixteen(cfg.num_layers, cfg.num_heads)
        qmodel = QuantizedCacheGPT(model, policy)
        seq = [1, 5, 3, 7, 2, 4, 6, 8]
        device = torch.device("cpu")

        baseline = teacher_forced_logits(model, [seq], device)[0]
        quant = quantized_incremental_logits(qmodel, [seq], device)[0]

        torch.testing.assert_close(quant, baseline, rtol=1e-5, atol=1e-5)

    def test_8bit_cache_differs_from_16bit(self) -> None:
        model = _tiny_model(seed=17)
        cfg = model.config
        device = torch.device("cpu")
        seq = list(range(8))

        policy_16 = CachePolicy.all_sixteen(cfg.num_layers, cfg.num_heads)
        policy_8 = CachePolicy(
            bits_per_layer=tuple(
                tuple(8 for _ in range(cfg.num_heads))
                for _ in range(cfg.num_layers)
            )
        )
        logits_16 = quantized_incremental_logits(
            QuantizedCacheGPT(model, policy_16), [seq], device
        )[0]
        logits_8 = quantized_incremental_logits(
            QuantizedCacheGPT(model, policy_8), [seq], device
        )[0]
        # INT8 introduces some error so logits differ
        self.assertFalse(torch.allclose(logits_8, logits_16, atol=1e-7))

    def test_incremental_cache_appends_only_new_sequence_positions(self) -> None:
        model = _tiny_model(seed=18)
        cfg = model.config
        policy = CachePolicy(
            bits_per_layer=tuple(
                tuple(4 for _ in range(cfg.num_heads))
                for _ in range(cfg.num_layers)
            )
        )
        adapter = QuantizedCacheGPT(model, policy)
        cache = None
        for expected_length in range(1, 5):
            token = torch.tensor([[expected_length]])
            _, cache = adapter(
                token,
                past_kv_cache=cache,
                use_cache=True,
            )
            self.assertEqual(
                cache[0].keys[0].payload.shape[1],
                expected_length,
            )


class TokenShardLoadingTests(unittest.TestCase):
    def test_header_is_validated_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_000000.bin"
            header = np.zeros(256, dtype="<i4")
            header[0] = 20250320
            header[1] = 1
            header[2] = 4
            tokens = np.array([1, 2, 9156, 4], dtype="<u2")
            path.write_bytes(header.tobytes() + tokens.tobytes())
            np.testing.assert_array_equal(
                load_token_shard(path),
                tokens.astype(np.int64),
            )

    def test_invalid_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.bin"
            path.write_bytes(np.zeros(256, dtype="<i4").tobytes())
            with self.assertRaises(ValueError):
                load_token_shard(path)


# ---------------------------------------------------------------------------
# Deterministic sensitivity
# ---------------------------------------------------------------------------


class DeterministicSensitivityTests(unittest.TestCase):
    def test_sensitivity_is_deterministic(self) -> None:
        model = _tiny_model(seed=21, num_layers=2, num_heads=4)
        seqs = [[1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 6, 8, 1, 3, 5, 7]]
        device = torch.device("cpu")

        s1 = compute_sensitivity_matrices(model, seqs, device)
        s2 = compute_sensitivity_matrices(model, seqs, device)

        self.assertEqual(s1["int8_matrix"], s2["int8_matrix"])
        self.assertEqual(s1["int4_matrix"], s2["int4_matrix"])

    def test_int4_sensitivity_geq_int8(self) -> None:
        model = _tiny_model(seed=22, num_layers=2, num_heads=4)
        seqs = [[1, 2, 3, 4, 5, 6, 7, 8]]
        device = torch.device("cpu")
        s = compute_sensitivity_matrices(model, seqs, device)
        for l in range(2):
            for h in range(4):
                self.assertGreaterEqual(
                    s["int4_matrix"][l][h],
                    s["int8_matrix"][l][h] - 1e-7,
                    msg=f"INT4 KL should be >= INT8 KL for layer {l} head {h}",
                )


# ---------------------------------------------------------------------------
# Policy precedence and budget adherence
# ---------------------------------------------------------------------------


class PolicyAllocationTests(unittest.TestCase):
    def _flat_sensitivity(
        self, num_layers: int, num_heads: int, int8_val: float, int4_val: float
    ) -> tuple[list[list[float]], list[list[float]]]:
        return (
            [[int8_val] * num_heads for _ in range(num_layers)],
            [[int4_val] * num_heads for _ in range(num_layers)],
        )

    def test_all16_policy_unchanged_at_zero_budget(self) -> None:
        int8_m, int4_m = self._flat_sensitivity(2, 4, 0.01, 0.05)
        result = allocate_policy_greedy(
            int8_m, int4_m,
            kl_budget=0.0,
            num_layers=2, num_heads=4, head_dim=4,
        )
        policy = result["policy"]
        for lb in policy.bits_per_layer:
            for b in lb:
                self.assertEqual(b, 16)

    def test_16_to_8_before_8_to_4(self) -> None:
        # With a budget that fits exactly one downgrade, choose 16->8
        int8_m, int4_m = self._flat_sensitivity(2, 4, 0.001, 0.01)
        result = allocate_policy_greedy(
            int8_m, int4_m,
            kl_budget=0.005,
            num_layers=2, num_heads=4, head_dim=4,
        )
        # All steps should be 16->8 (never 8->4 before 16->8)
        for step in result["steps"]:
            self.assertEqual(step["from_bits"], 16)
            self.assertEqual(step["to_bits"], 8)

    def test_budget_not_exceeded(self) -> None:
        int8_m, int4_m = self._flat_sensitivity(2, 4, 0.01, 0.05)
        budget = 0.035
        result = allocate_policy_greedy(
            int8_m, int4_m,
            kl_budget=budget,
            num_layers=2, num_heads=4, head_dim=4,
        )
        self.assertLessEqual(result["predicted_kl"], budget + 1e-9)

    def test_build_named_policies_contains_required_labels(self) -> None:
        int8_m, int4_m = self._flat_sensitivity(2, 4, 0.01, 0.05)
        policies = build_named_policies(
            int8_m, int4_m,
            num_layers=2, num_heads=4, head_dim=4,
            kl_budgets=[0.05, 0.1],
        )
        labels = [p["label"] for p in policies]
        self.assertIn("uniform_int8", labels)
        self.assertIn("uniform_int4", labels)
        self.assertIn("all_fp32", labels)
        greedy_labels = [l for l in labels if l.startswith("greedy_")]
        self.assertEqual(len(greedy_labels), 2)

    def test_allocator_skips_infeasible_lower_ratio_transition(self) -> None:
        result = allocate_policy_greedy(
            [[0.04, 0.05]],
            [[0.06, 0.50]],
            kl_budget=0.07,
            num_layers=1,
            num_heads=2,
            head_dim=8,
            model_dtype_bytes=4,
        )
        self.assertAlmostEqual(result["predicted_kl"], 0.06)
        self.assertEqual(result["policy"].bits_per_layer[0], (4, 16))

    def test_cache_policy_all_sixteen(self) -> None:
        p = CachePolicy.all_sixteen(3, 4)
        self.assertEqual(p.num_layers, 3)
        self.assertEqual(p.num_heads, 4)
        for lb in p.bits_per_layer:
            self.assertTrue(all(b == 16 for b in lb))

    def test_cache_policy_with_head_immutable(self) -> None:
        p = CachePolicy.all_sixteen(2, 4)
        p2 = p.with_head(0, 1, 8)
        # Original unchanged
        self.assertEqual(p.bits_per_layer[0][1], 16)
        self.assertEqual(p2.bits_per_layer[0][1], 8)


# ---------------------------------------------------------------------------
# Split separation
# ---------------------------------------------------------------------------


class SplitSeparationTests(unittest.TestCase):
    def test_calib_val_do_not_overlap(self) -> None:
        tokens = np.arange(512, dtype=np.int64)
        calib = token_windows(tokens, 0, 128, 32)
        val = token_windows(tokens, 128, 256, 32)
        calib_set = {tuple(s) for s in calib}
        val_set = {tuple(s) for s in val}
        self.assertEqual(len(calib_set & val_set), 0)

    def test_study_rejects_overlapping_splits(self) -> None:
        with self.assertRaises(ValueError):
            run_study(
                {
                    "device": "cpu",
                    "dtype": "float32",
                    "seed": 1,
                    "calib_start": 0,
                    "calib_end": 200,
                    "val_start": 100,  # overlaps calib
                    "val_end": 300,
                    "seq_len": 8,
                    "kl_budgets": [0.01],
                    "model_config": {
                        "vocab_size": 32,
                        "context_len": 16,
                        "embedding_dim": 16,
                        "num_layers": 1,
                        "num_heads": 2,
                        "dropout": 0.0,
                    },
                }
            )


# ---------------------------------------------------------------------------
# Metric formulas
# ---------------------------------------------------------------------------


class MetricFormulaTests(unittest.TestCase):
    def test_kl_formula_matches_manual(self) -> None:
        logits_q = torch.tensor([[0.0, 1.0, -1.0]])
        logits_b = torch.tensor([[1.0, 0.0, -1.0]])
        kl_auto = categorical_kl(logits_q, logits_b)
        p = torch.softmax(logits_q, dim=-1)
        log_p = torch.log_softmax(logits_q, dim=-1)
        log_q = torch.log_softmax(logits_b, dim=-1)
        kl_manual = (p * (log_p - log_q)).sum()
        torch.testing.assert_close(kl_auto, kl_manual)

    def test_perplexity_formula(self) -> None:
        import math
        ce = 2.5
        ppl = math.exp(ce)
        ppl_tensor = float(torch.tensor(ce).exp().item())
        self.assertAlmostEqual(ppl, ppl_tensor, places=5)


# ---------------------------------------------------------------------------
# Teacher frozen / no gradients
# ---------------------------------------------------------------------------


class TeacherFrozenTests(unittest.TestCase):
    def test_teacher_receives_no_gradients(self) -> None:
        torch.manual_seed(5)
        cfg = GPTConfig(
            vocab_size=32, context_len=16, embedding_dim=16,
            num_layers=2, num_heads=4, dropout=0.0
        )
        teacher = GPT(cfg).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        student = NoisyKVGPT(cfg, noise_bits=8).train()
        student.load_state_dict(teacher.state_dict())

        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        student_logits = student(ids)[:, :-1]
        targets = ids[:, 1:]
        with torch.no_grad():
            teacher_logits = teacher(ids)[:, :-1]

        loss = (
            F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)),
                targets.reshape(-1),
            )
            + 0.1 * categorical_kl(student_logits, teacher_logits)
        )
        loss.backward()

        for p in teacher.parameters():
            self.assertIsNone(p.grad)

    def test_student_receives_gradients(self) -> None:
        torch.manual_seed(6)
        cfg = GPTConfig(
            vocab_size=32, context_len=16, embedding_dim=16,
            num_layers=2, num_heads=4, dropout=0.0
        )
        student = NoisyKVGPT(cfg, noise_bits=4).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 6))
        logits = student(ids)[:, :-1]
        targets = ids[:, 1:]
        F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        ).backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in student.parameters()
        )
        self.assertTrue(has_grad)


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_noisy_kv_gpt_loads_gpt_state_dict(self) -> None:
        torch.manual_seed(8)
        cfg = GPTConfig(
            vocab_size=32, context_len=16, embedding_dim=16,
            num_layers=2, num_heads=4, dropout=0.0
        )
        gpt = GPT(cfg)
        noisy = NoisyKVGPT(cfg, noise_bits=8, noise_type="gaussian")
        # Must not raise
        noisy.load_state_dict(gpt.state_dict())

    def test_noisy_kv_gpt_supports_incremental_cache_evaluation(self) -> None:
        torch.manual_seed(81)
        cfg = GPTConfig(
            vocab_size=32,
            context_len=16,
            embedding_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        source = GPT(cfg).eval()
        noisy = NoisyKVGPT(cfg, noise_bits=4).eval()
        noisy.load_state_dict(source.state_dict())
        adapter = QuantizedCacheGPT(
            noisy,
            CachePolicy.all_sixteen(cfg.num_layers, cfg.num_heads),
        )
        tokens = torch.randint(0, cfg.vocab_size, (1, 8))
        full = noisy(tokens)
        self.assertIsInstance(full, torch.Tensor)
        incremental = quantized_incremental_logits(
            adapter,
            [tokens.squeeze(0).tolist()],
            torch.device("cpu"),
        )[0]
        torch.testing.assert_close(incremental, full.squeeze(0), atol=1e-5, rtol=1e-5)

    def test_state_dicts_have_same_keys(self) -> None:
        torch.manual_seed(9)
        cfg = GPTConfig(
            vocab_size=32, context_len=16, embedding_dim=16,
            num_layers=2, num_heads=4, dropout=0.0
        )
        gpt = GPT(cfg)
        noisy = NoisyKVGPT(cfg, noise_bits=4)
        gpt_keys = set(gpt.state_dict().keys())
        noisy_keys = set(noisy.state_dict().keys())
        self.assertEqual(gpt_keys, noisy_keys)

    def test_noisy_kv_gpt_rejects_invalid_noise_bits(self) -> None:
        cfg = GPTConfig(vocab_size=32, context_len=8, embedding_dim=8, num_heads=2)
        with self.assertRaises(ValueError):
            NoisyKVGPT(cfg, noise_bits=16)

    def test_noisy_block_rejects_invalid_noise_type(self) -> None:
        cfg = GPTConfig(vocab_size=32, context_len=8, embedding_dim=8, num_heads=2)
        with self.assertRaises(ValueError):
            NoisyBlock(cfg, noise_bits=8, noise_type="laplace")


# ---------------------------------------------------------------------------
# Artifact schema
# ---------------------------------------------------------------------------


class ArtifactSchemaTests(unittest.TestCase):
    def _run_smoke(self) -> dict:
        return run_study(
            {
                "device": "cpu",
                "dtype": "float32",
                "seed": 42,
                "calib_start": 0,
                "calib_end": 64,
                "val_start": 64,
                "val_end": 128,
                "seq_len": 8,
                "kl_budgets": [0.005, 0.02],
                "time_sequences": 0,
                "model_config": {
                    "vocab_size": 32,
                    "context_len": 16,
                    "embedding_dim": 16,
                    "num_layers": 2,
                    "num_heads": 4,
                    "dropout": 0.0,
                },
            }
        )

    def test_artifact_top_level_keys(self) -> None:
        payload = self._run_smoke()
        required = {
            "schema_version", "status", "created_at", "experiment_config",
            "environment", "model_config", "splits", "sensitivity",
            "policies", "validation", "caveats", "hashes",
        }
        for key in required:
            self.assertIn(key, payload, f"missing key: {key}")

    def test_sensitivity_matrix_shape(self) -> None:
        payload = self._run_smoke()
        s = payload["sensitivity"]
        self.assertEqual(len(s["int8_matrix"]), s["num_layers"])
        self.assertEqual(len(s["int4_matrix"]), s["num_layers"])
        for row in s["int8_matrix"]:
            self.assertEqual(len(row), s["num_heads"])

    def test_validation_results_contain_required_metrics(self) -> None:
        payload = self._run_smoke()
        required_metrics = {
            "held_out_ce", "perplexity", "mean_kl", "p50_kl", "p95_kl",
            "p99_kl", "max_kl", "token_agreement_rate", "max_logit_deviation",
            "cache_total_bytes", "compression_ratio_vs_fp32",
            "compression_factor_vs_full_precision", "avg_payload_bits",
        }
        for result in payload["validation"]["results"]:
            for key in required_metrics:
                self.assertIn(key, result, f"missing metric {key} in {result.get('policy_label')}")

    def test_split_non_overlap_recorded(self) -> None:
        payload = self._run_smoke()
        self.assertFalse(payload["splits"]["overlap"])

    def test_policies_list_contains_expected_labels(self) -> None:
        payload = self._run_smoke()
        labels = [p["label"] for p in payload["policies"]]
        self.assertIn("uniform_int8", labels)
        self.assertIn("uniform_int4", labels)
        self.assertIn("all_fp32", labels)

    def test_caveats_mention_persistent_vs_transient(self) -> None:
        payload = self._run_smoke()
        combined = " ".join(payload["caveats"])
        self.assertIn("transient", combined.lower())

    def test_schema_version(self) -> None:
        payload = self._run_smoke()
        self.assertEqual(payload["schema_version"], "kv-cache-study-v1")


if __name__ == "__main__":
    unittest.main()
