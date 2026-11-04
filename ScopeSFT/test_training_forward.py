"""Native Qwen/PEFT and two-rank DDP checks for the bounded loss forward."""
from contextlib import nullcontext
from datetime import timedelta
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from ScopeCT.objective import scope_loss
from ScopeSFT.attention import register_chunked_attention
from ScopeSFT.model_support import lora_target_modules
from ScopeSFT.training_forward import install_training_forward, remove_training_forward


def native_model(kind, *, chunked=False):
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3_5TextConfig, Qwen3_5ForCausalLM

    shared = dict(vocab_size=23, hidden_size=16, intermediate_size=24,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
                  tie_word_embeddings=True, pad_token_id=0, eos_token_id=22)
    if kind == "qwen3":
        config, constructor = Qwen3Config(**shared), Qwen3ForCausalLM
    else:
        config = Qwen3_5TextConfig(
            **shared, layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=1, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5,
                             "mrope_section": [1, 1, 0]},
        )
        constructor = Qwen3_5ForCausalLM
    config._attn_implementation = register_chunked_attention(3) if chunked else "eager"
    return constructor(config)


def adapted_model(kind, *, patched, chunked=False, checkpointed=False, dropout=0.0):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(83)
    model = native_model(kind, chunked=chunked)
    targets = lora_target_modules(model)
    if patched:
        install_training_forward(model, ct_weight=0.2, margin=0.7, chunk_size=2)
    model = get_peft_model(model, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=targets,
    ))
    # Both A and B must contribute to parity checks; the usual initial B=0
    # would make all A gradients identically zero.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.03)
    model.enable_input_require_grads()
    if checkpointed:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.train()


def training_batch():
    tokens = torch.tensor([
        [1, 2, 3, 4, 5, 6, 0, 0, 0],
        [1, 4, 3, 8, 9, 10, 11, 12, 13],
        [1, 3, 5, 7, 9, 11, 13, 15, 17],
        [1, 6, 8, 10, 12, 14, 16, 0, 0],
    ])
    labels = torch.tensor([
        [-100, -100, 3, 4, 5, 6, -100, -100, -100],
        [-100, -100, -100, -100, -100, 10, 11, 12, 13],
        [-100, -100, -100, 7, 9, 11, 13, 15, 17],
        [-100, -100, -100, -100, -100, 14, 16, -100, -100],
    ])
    return dict(input_ids=tokens, attention_mask=(tokens != 0).long(), labels=labels,
                critical_positions=torch.tensor([2, -1, 5, 6]),
                positive_ids=torch.tensor([3, -1, 11, 16]),
                negative_ids=torch.tensor([5, -1, 4, 7]))


def dense_loss(model, batch):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    return scope_loss(outputs.logits, batch["labels"], batch["critical_positions"],
                      batch["positive_ids"], batch["negative_ids"], ct_weight=0.2, margin=0.7)


def trainable_gradients(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise AssertionError(f"Trainable parameter has no gradient: {name}")
            if not bool(torch.isfinite(parameter.grad).all()):
                raise AssertionError(f"Nonfinite trainable gradient: {name}")
            result[name] = parameter.grad.detach().cpu().clone()
    return result


def ddp_worker(rank, rendezvous, output_directory):
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model = adapted_model("qwen35", patched=True, chunked=True, checkpointed=True)
        wrapped = DistributedDataParallel(model, find_unused_parameters=False)
        calls = []
        wrapped.register_forward_pre_hook(lambda *args: calls.append(1))
        batch = training_batch()
        for microbatch in range(2):
            row = rank + 2 * microbatch
            inputs = {name: value[row:row + 1] for name, value in batch.items()}
            # Accumulate one local microbatch without reduction, then perform
            # exactly one distributed average for the two accumulated batches.
            with wrapped.no_sync() if microbatch == 0 else nullcontext():
                outputs = wrapped(**inputs, scope_compute_loss=True, use_cache=False)
                (outputs.loss / 2).backward()
        torch.save({"gradients": trainable_gradients(model), "wrapped_calls": len(calls)},
                   Path(output_directory) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


class TrainingForwardTests(unittest.TestCase):
    def assert_gradients_close(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        self.assertTrue(actual)
        for name in expected:
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual[name], expected[name], rtol=7e-4, atol=2e-6)

    def test_peft_loss_metrics_and_all_lora_gradients_match_native_qwen(self):
        for kind in ("qwen3", "qwen35"):
            with self.subTest(kind=kind):
                reference = adapted_model(kind, patched=False, dropout=0.05)
                bounded = adapted_model(kind, patched=True, chunked=True,
                                        checkpointed=True, dropout=0.05)
                bounded.load_state_dict(reference.state_dict())
                batch = training_batch()
                # LoRA dropout must use the same masks in both backbone calls.
                torch.manual_seed(71)
                expected, expected_metrics = dense_loss(reference, batch)
                expected.backward()
                torch.manual_seed(71)
                actual = bounded(**batch, scope_compute_loss=True, use_cache=False)
                actual.loss.backward()
                self.assertNotIn("logits", actual)
                torch.testing.assert_close(actual.loss, expected)
                for name in expected_metrics:
                    torch.testing.assert_close(actual.scope_metrics[name], expected_metrics[name])
                gradients = trainable_gradients(bounded)
                self.assertTrue(all("lora_" in name for name in gradients))
                self.assertTrue(any(bool(value.abs().sum()) for name, value in gradients.items() if "lora_A" in name))
                self.assertTrue(any(bool(value.abs().sum()) for name, value in gradients.items() if "lora_B" in name))
                self.assert_gradients_close(gradients, trainable_gradients(reference))

    def test_trainer_compute_loss_enters_peft_forward_and_returns_metrics(self):
        from ScopeSFT.runtime import trainer_class

        model = adapted_model("qwen35", patched=True)
        calls = []
        model.register_forward_pre_hook(lambda *args: calls.append(1))
        trainer = object.__new__(trainer_class())
        trainer.ct_weight, trainer.margin, trainer.low_memory = 0.2, 0.7, True
        loss, outputs = trainer.compute_loss(model, training_batch(), return_outputs=True,
                                             num_items_in_batch=torch.tensor(16))
        self.assertIs(loss, outputs.loss)
        self.assertEqual(len(calls), 1)
        self.assertEqual(trainer.last_loss_components["ct_pairs"], 3)
        self.assertEqual(trainer.last_loss_components["supervised_tokens"], 16)
        loss.backward()
        self.assertTrue(trainable_gradients(model))

    def test_native_keyword_and_positional_cached_inference_and_remove(self):
        for kind in ("qwen3", "qwen35"):
            with self.subTest(kind=kind):
                torch.manual_seed(32)
                model = native_model(kind).eval()
                original = model.forward
                keys = set(model.state_dict())
                tokens = training_batch()["input_ids"][:1, :5]
                padding = torch.ones_like(tokens)
                positions = torch.arange(tokens.shape[1])[None]
                with torch.no_grad():
                    expected = model(tokens, padding, positions, use_cache=True)
                    expected_labeled = model(input_ids=tokens, labels=tokens, use_cache=False)
                    install_training_forward(model, chunk_size=2)
                    positional = model(tokens, padding, positions, use_cache=True)
                    keyword = model(input_ids=tokens, attention_mask=padding,
                                    position_ids=positions, use_cache=True)
                    labeled = model(input_ids=tokens, labels=tokens, use_cache=False)
                for actual in (positional, keyword):
                    torch.testing.assert_close(actual.logits, expected.logits)
                    self.assertIsNotNone(actual.past_key_values)
                    self.assertIs(type(actual.past_key_values), type(expected.past_key_values))
                torch.testing.assert_close(labeled.loss, expected_labeled.loss)
                self.assertEqual(set(model.state_dict()), keys)
                with self.assertRaises(ValueError):
                    install_training_forward(model)
                remove_training_forward(model)
                self.assertFalse(hasattr(model, "_scope_original_forward"))
                self.assertIs(model.forward.__func__, original.__func__)
                self.assertIs(model.forward.__self__, model)
                remove_training_forward(model)  # Removal is safe before export twice.
                with torch.no_grad():
                    restored = model(tokens, padding, positions, use_cache=False)
                torch.testing.assert_close(restored.logits, expected.logits)

    def test_training_rejects_missing_labels_cache_and_logit_slicing(self):
        model = native_model("qwen3")
        install_training_forward(model)
        batch = training_batch()
        missing_labels = {name: value for name, value in batch.items() if name != "labels"}
        for inputs in (missing_labels, batch | {"use_cache": True}, batch | {"logits_to_keep": 2}):
            with self.subTest(keys=list(inputs)), self.assertRaises(ValueError):
                model(**inputs, scope_compute_loss=True)
        with self.assertRaises(ValueError):
            model(batch["input_ids"], scope_compute_loss=True, labels=batch["labels"])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "GLOO is unavailable")
    def test_two_rank_ddp_no_sync_gradients_equal_combined_native_batch(self):
        # GLOO still needs loopback sockets with a file store. A narrowly scoped
        # environment restriction is reported as a skip, not a passing DDP run.
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        with tempfile.TemporaryDirectory(prefix="scope-forward-ddp-") as directory:
            probe = str(Path(directory) / "probe")
            try:
                dist.init_process_group("gloo", init_method="file://" + probe, rank=0,
                                        world_size=1, timeout=timedelta(seconds=10))
            except RuntimeError as error:
                if "Operation not permitted" in str(error) or "Permission denied" in str(error):
                    self.skipTest("Loopback socket access is blocked: " + str(error))
                raise
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()
            reference = adapted_model("qwen35", patched=False)
            loss, _ = dense_loss(reference, training_batch())
            loss.backward()
            expected = trainable_gradients(reference)
            mp.spawn(ddp_worker, args=(str(Path(directory) / "rendezvous"), directory), nprocs=2, join=True)
            for rank in range(2):
                actual = torch.load(Path(directory) / f"rank-{rank}.pt", weights_only=True)
                self.assertEqual(actual["wrapped_calls"], 2)
                self.assert_gradients_close(actual["gradients"], expected)


if __name__ == "__main__":
    unittest.main()
