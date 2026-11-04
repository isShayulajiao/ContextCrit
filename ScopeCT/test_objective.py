"""Numerical checks for the causal shift, CT gradients, and missing pairs."""
import math
import unittest

import torch
import torch.nn.functional as F

from ScopeCT.objective import scope_loss


class ScopeLossTests(unittest.TestCase):
    def fixture(self):
        logits = torch.zeros(2, 5, 4, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2, -100], [-100, -100, -100, 3, -100]])
        positions = torch.tensor([2, -1])
        positive = torch.tensor([1, -1])
        negative = torch.tensor([0, -1])
        return logits, labels, positions, positive, negative

    def test_exact_uniform_loss_and_batch_denominator(self):
        args = self.fixture()
        loss, metrics = scope_loss(*args, ct_weight=0.2, margin=1.0)
        self.assertAlmostEqual(loss.item(), math.log(4) + 0.1 * F.softplus(torch.tensor(1.)).item(), places=6)
        self.assertEqual(metrics["ct_pairs"].item(), 1)
        self.assertEqual(metrics["supervised_tokens"].item(), 3)

    def test_next_token_shift_and_direction(self):
        args = self.fixture()
        ordinary, _ = scope_loss(args[0], args[1])
        enhanced, _ = scope_loss(*args)
        delta = torch.autograd.grad(enhanced - ordinary, args[0])[0]
        self.assertLess(delta[0, 1, 1].item(), 0)  # Gradient descent raises correct logit.
        self.assertGreater(delta[0, 1, 0].item(), 0)
        delta[0, 1, 1] = 0
        delta[0, 1, 0] = 0
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_no_pairs_is_exact_sft(self):
        logits, labels, *_ = self.fixture()
        ordinary, _ = scope_loss(logits, labels)
        absent = torch.tensor([-1, -1])
        enhanced, metrics = scope_loss(logits, labels, absent, absent, absent)
        self.assertTrue(torch.equal(ordinary, enhanced))
        self.assertEqual(metrics["ct_loss"].item(), 0)
        enhanced.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_per_example_nll_and_chunk_equivalence(self):
        torch.manual_seed(17)
        logits = torch.randn(2, 5, 4, requires_grad=True)
        labels = self.fixture()[1]
        expected = torch.stack([
            F.cross_entropy(logits[0, 1:3], torch.tensor([1, 2])),
            F.cross_entropy(logits[1, 2:3], torch.tensor([3])),
        ]).mean()
        small, _ = scope_loss(logits, labels, chunk_size=1)
        large, _ = scope_loss(logits, labels, chunk_size=128)
        torch.testing.assert_close(small, expected)
        torch.testing.assert_close(large, expected)
        small_gradient = torch.autograd.grad(small, logits, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(small_gradient, expected_gradient)

    def test_margin_reduces_as_correct_gap_increases(self):
        args = list(self.fixture())
        _, before = scope_loss(*args)
        changed = args[0].detach().clone()
        changed[0, 1, 1] = 5
        changed[0, 1, 0] = -5
        args[0] = changed.requires_grad_()
        _, after = scope_loss(*args)
        self.assertLess(after["ct_loss"].item(), before["ct_loss"].item())
        self.assertAlmostEqual(after["ct_gap_mean"].item(), 10.)

    def test_prompt_padding_and_final_logits_have_no_nll_gradient(self):
        logits, labels, *_ = self.fixture()
        loss, _ = scope_loss(logits, labels)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[:, 0], torch.zeros_like(logits.grad[:, 0])))
        self.assertTrue(torch.equal(logits.grad[:, 3:], torch.zeros_like(logits.grad[:, 3:])))

    def test_bad_metadata_rejected(self):
        for change in ["position_zero", "position_end", "wrong_positive", "same_tokens", "out_of_vocab", "partial_absent"]:
            args = list(self.fixture())
            if change == "position_zero":
                args[2][0] = 0
            elif change == "position_end":
                args[2][0] = 5
            elif change == "wrong_positive":
                args[3][0] = 2
            elif change == "same_tokens":
                args[4][0] = 1
            elif change == "out_of_vocab":
                args[4][0] = 4
            elif change == "partial_absent":
                args[3][1] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                scope_loss(*args)

    def test_invalid_labels_and_empty_supervision_rejected(self):
        for value in [4, -2, -100]:
            logits, labels, *_ = self.fixture()
            labels[0, 2:4] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                scope_loss(logits, labels)

    def test_invalid_hyperparameters_rejected(self):
        logits, labels, *_ = self.fixture()
        for kwargs in [{"ct_weight": -1}, {"ct_weight": float("nan")}, {"margin": -1}, {"margin": float("inf")}, {"chunk_size": 0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                scope_loss(logits, labels, **kwargs)

    def test_fp16_extreme_logits_are_finite(self):
        logits = torch.tensor([[[0., 0., 0.], [100., -100., 0.], [0., 0., 0.]]], dtype=torch.float16, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1]])
        loss, _ = scope_loss(logits, labels, torch.tensor([2]), torch.tensor([1]), torch.tensor([0]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
