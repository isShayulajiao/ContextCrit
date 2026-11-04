"""Loss/gradient parity and retained-activation checks for the bounded head."""
from copy import deepcopy
import unittest

import torch
from torch import nn

from ScopeCT.objective import scope_loss
from ScopeSFT.memory import chunked_scope_loss


class RecordingHead(nn.Linear):
    def __init__(self, hidden, vocab):
        super().__init__(hidden, vocab)
        self.rows = []

    def forward(self, inputs):
        self.rows.append(inputs.shape[0])
        return super().forward(inputs)


class ChunkedLossTests(unittest.TestCase):
    def fixture(self, batch=3):
        torch.manual_seed(37)
        hidden = torch.randn(batch, 9, 5, requires_grad=True)
        head = RecordingHead(5, 13)
        labels = torch.tensor([
            [-100, -100, 2, 4, 6, 8, -100, -100, -100],
            [-100, -100, -100, -100, -100, 1, 3, 5, -100],
            [-100, 7, -100, 2, 8, -100, 10, 3, 4],
        ])[:batch]
        positions = torch.tensor([2, -1, 8])[:batch]
        positive = torch.tensor([2, -1, 4])[:batch]
        negative = torch.tensor([1, -1, 12])[:batch]
        return hidden, head, labels, positions, positive, negative

    def test_all_targets_ct_metrics_and_gradients_match_dense_head(self):
        for chunk_size in (1, 2, 128):
            for ct_weight in (0.0, 0.2, 1.3):
                with self.subTest(chunk_size=chunk_size, ct_weight=ct_weight):
                    hidden, head, labels, positions, pos, neg = self.fixture()
                    dense_hidden = hidden.detach().clone().requires_grad_()
                    dense_head = deepcopy(head)
                    expected, expected_metrics = scope_loss(
                        dense_head(dense_hidden), labels, positions, pos, neg,
                        ct_weight=ct_weight, margin=0.7,
                    )
                    actual, metrics = chunked_scope_loss(
                        hidden, head, labels, positions, pos, neg,
                        ct_weight=ct_weight, margin=0.7, chunk_size=chunk_size,
                    )
                    torch.testing.assert_close(actual, expected)
                    for key in metrics:
                        torch.testing.assert_close(metrics[key], expected_metrics[key])
                    dense_gradients = torch.autograd.grad(expected, (dense_hidden, dense_head.weight, dense_head.bias))
                    gradients = torch.autograd.grad(actual, (hidden, head.weight, head.bias))
                    for gradient, expected_gradient in zip(gradients, dense_gradients):
                        torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=2e-7)
                    self.assertEqual(metrics["supervised_tokens"].item(), 13)
                    self.assertEqual(metrics["ct_pairs"].item(), 2)

    def test_missing_pairs_is_response_sft_and_unsupervised_states_have_zero_gradient(self):
        hidden, head, labels, *_ = self.fixture()
        actual, metrics = chunked_scope_loss(hidden, head, labels, chunk_size=2)
        expected, _ = scope_loss(head(hidden), labels)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(metrics["ct_loss"].item(), 0)
        gradient = torch.autograd.grad(actual, hidden)[0]
        ignored = torch.cat((labels[:, 1:] == -100, torch.ones(3, 1, dtype=torch.bool)), dim=1)
        self.assertTrue(torch.equal(gradient[ignored], torch.zeros_like(gradient[ignored])))

    def test_checkpoint_does_not_retain_full_vocabulary_activations(self):
        hidden, head, labels, positions, pos, neg = self.fixture()
        saved_shapes = []

        def pack(tensor):
            saved_shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss, _ = chunked_scope_loss(hidden, head, labels, positions, pos, neg, chunk_size=2)
        # No logits or probabilities are saved, even for individual blocks.
        self.assertFalse(any(len(shape) >= 2 and shape[-1] == head.out_features for shape in saved_shapes))
        forward_calls = len(head.rows)
        loss.backward()
        self.assertGreater(len(head.rows), forward_calls)  # Head/CE actually recomputed.
        self.assertLessEqual(max(head.rows), 2)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_gradient_accumulation_preserves_example_mean_with_unequal_response_lengths(self):
        hidden, head, labels, positions, pos, neg = self.fixture(batch=2)
        dense_hidden = hidden.detach().clone().requires_grad_()
        dense_head = deepcopy(head)
        expected, _ = scope_loss(dense_head(dense_hidden), labels, positions, pos, neg)
        expected.backward()
        for row in range(2):
            loss, _ = chunked_scope_loss(
                hidden[row:row + 1], head, labels[row:row + 1],
                positions[row:row + 1], pos[row:row + 1], neg[row:row + 1], chunk_size=2,
            )
            (loss / 2).backward()
        torch.testing.assert_close(head.weight.grad, dense_head.weight.grad)
        torch.testing.assert_close(head.bias.grad, dense_head.bias.grad)
        torch.testing.assert_close(hidden.grad, dense_hidden.grad)

    def test_frozen_head_still_backpropagates_to_hidden_states(self):
        hidden, head, labels, *_ = self.fixture()
        head.requires_grad_(False)
        actual, _ = chunked_scope_loss(hidden, head, labels, chunk_size=2)
        expected, _ = scope_loss(head(hidden), labels)
        actual_gradient = torch.autograd.grad(actual, hidden, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, hidden)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_36k_context_keeps_late_response_eos_and_ct_target(self):
        # Exercise absolute target positions above 32768 without a large model.
        # This checks the head/indexing path, not GPU/backbone memory capacity.
        torch.manual_seed(73)
        hidden = torch.randn(1, 36001, 5, requires_grad=True)
        head = RecordingHead(5, 13)
        labels = torch.full((1, 36001), -100, dtype=torch.long)
        labels[0, -3:] = torch.tensor([2, 5, 12])
        positions, pos, neg = torch.tensor([36000]), torch.tensor([12]), torch.tensor([4])
        expected, _ = scope_loss(head(hidden), labels, positions, pos, neg)
        head.rows.clear()
        actual, metrics = chunked_scope_loss(hidden, head, labels, positions, pos, neg, chunk_size=2)
        torch.testing.assert_close(actual, expected)
        actual_gradient = torch.autograd.grad(actual, hidden, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, hidden)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertEqual(metrics["supervised_tokens"].item(), 3)
        self.assertEqual(metrics["ct_pairs"].item(), 1)
        self.assertLessEqual(max(head.rows), 2)

    def test_tied_embedding_and_head_receive_both_gradient_paths(self):
        torch.manual_seed(11)
        embedding = nn.Embedding(13, 5)
        head = nn.Linear(5, 13, bias=False)
        head.weight = embedding.weight
        dense_embedding = deepcopy(embedding)
        dense_head = nn.Linear(5, 13, bias=False)
        dense_head.weight = dense_embedding.weight
        tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, -100, -100, 9, 10]])
        positions, positive, negative = torch.tensor([2, -1]), torch.tensor([3, -1]), torch.tensor([7, -1])
        expected, _ = scope_loss(dense_head(dense_embedding(tokens)), labels, positions, positive, negative)
        actual, _ = chunked_scope_loss(embedding(tokens), head, labels, positions, positive, negative, chunk_size=2)
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(embedding.weight.grad, dense_embedding.weight.grad)

    def test_cpu_fp16_autocast_preserves_head_and_hidden_gradients(self):
        hidden, head, labels, positions, positive, negative = self.fixture()
        dense_hidden = hidden.detach().clone().requires_grad_()
        dense_head = deepcopy(head)
        with torch.autocast("cpu", dtype=torch.float16):
            expected, _ = scope_loss(dense_head(dense_hidden), labels, positions, positive, negative)
            actual, _ = chunked_scope_loss(hidden, head, labels, positions, positive, negative, chunk_size=2)
        expected_gradients = torch.autograd.grad(expected, (dense_hidden, dense_head.weight, dense_head.bias))
        gradients = torch.autograd.grad(actual, (hidden, head.weight, head.bias))
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        for gradient, expected_gradient in zip(gradients, expected_gradients):
            torch.testing.assert_close(gradient, expected_gradient, rtol=3e-3, atol=2e-4)

    def test_fp16_extreme_head_values_are_finite(self):
        hidden = torch.tensor([[[1., 0.], [1., 0.], [0., 0.]]], dtype=torch.float16, requires_grad=True)
        head = nn.Linear(2, 3, bias=False, dtype=torch.float16)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[100., 0.], [-100., 0.], [0., 0.]]))
        loss, _ = chunked_scope_loss(hidden, head, torch.tensor([[-100, -100, 1]]),
                                     torch.tensor([2]), torch.tensor([1]), torch.tensor([0]), chunk_size=1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(torch.isfinite(head.weight.grad).all())

    def test_validation_matches_dense_objective(self):
        for change in ("position_zero", "position_end", "wrong_positive", "same_tokens",
                       "out_of_vocab", "partial_absent", "empty_supervision", "invalid_label"):
            hidden, head, labels, positions, pos, neg = self.fixture()
            if change == "position_zero":
                positions[0] = 0
            elif change == "position_end":
                positions[0] = labels.shape[1]
            elif change == "wrong_positive":
                pos[0] = 3
            elif change == "same_tokens":
                neg[0] = pos[0]
            elif change == "out_of_vocab":
                neg[0] = head.out_features
            elif change == "partial_absent":
                pos[1] = 3
            elif change == "empty_supervision":
                labels[0] = -100
            elif change == "invalid_label":
                labels[0, 3] = head.out_features
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    scope_loss(head(hidden), labels, positions, pos, neg)
                with self.assertRaises(ValueError):
                    chunked_scope_loss(hidden, head, labels, positions, pos, neg)

    def test_invalid_hyperparameters_and_partial_metadata_rejected(self):
        hidden, head, labels, positions, *_ = self.fixture()
        for kwargs in ({"ct_weight": -1}, {"ct_weight": float("nan")}, {"margin": -1},
                       {"margin": float("inf")}, {"chunk_size": 0}, {"chunk_size": True},
                       {"critical_positions": positions}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                chunked_scope_loss(hidden, head, labels, **kwargs)


if __name__ == "__main__":
    unittest.main()
