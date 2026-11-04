"""Real fast-tokenizer checks without downloading model/tokenizer artifacts."""
from copy import deepcopy
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from ScopeSFT.tokenization import Collator, encode_record


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
    "{% if not enable_thinking %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}"
    "{% endif %}"
)


def make_tokenizer(texts):
    """A genuine Rust fast tokenizer with offsets, fitted only to local fixtures."""
    backend = Tokenizer(models.WordLevel(unk_token="<|unk|>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    specials = ["<|pad|>", "<|unk|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    backend.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=specials, min_frequency=1))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<|unk|>",
                                       pad_token="<|pad|>", eos_token="<|im_end|>",
                                       additional_special_tokens=["<|im_start|>", "<think>", "</think>"])
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def record(chosen='{"text":"good"}', rejected='{"text":"bad"}'):
    start = chosen.index("good")
    other = rejected.index("bad") if "bad" in rejected else rejected.index("good")
    return {"messages": [{"role": "system", "content": "Return JSON."},
                         {"role": "user", "content": "Judge visible evidence."}],
            "chosen": chosen, "rejected": rejected,
            "semantic_span": {"field": "text", "chosen": [start, start + 4],
                              "rejected": [other, other + (3 if "bad" in rejected else 4)]},
            "metadata": {"sample_id": "token-test", "action": "PROPOSE"}}


class TokenizationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = make_tokenizer([
            'system user assistant Return JSON. Judge visible evidence.',
            '{"text":"good"}', '{"text":"bad"}', '{"text":"good","extra":1}',
            '{"text":"good","reason":"ok"}', '{"reason":"ok","text":"bad"}',
        ])

    def test_semantic_pair_uses_exact_first_divergence_and_target_index(self):
        source = record()
        encoded, audit = encode_record(self.tokenizer, source, 1000)
        self.assertEqual(audit["pair_status"], "valid")
        chosen = self.tokenizer.encode(source["chosen"], add_special_tokens=False)
        rejected = self.tokenizer.encode(source["rejected"], add_special_tokens=False)
        first = next(i for i, (a, b) in enumerate(zip(chosen, rejected)) if a != b)
        self.assertEqual(audit["response_token_index"], first)
        prompt_count = next(i for i, target in enumerate(encoded["labels"]) if target != -100)
        position = encoded["critical_positions"]
        self.assertEqual(position, prompt_count + first)
        self.assertEqual(encoded["labels"][position], chosen[first])
        self.assertEqual(encoded["positive_ids"], chosen[first])
        self.assertEqual(encoded["negative_ids"], rejected[first])
        self.assertGreater(position, 0)  # It is predicted by causal logits[position - 1].
        self.assertEqual(chosen[:first], rejected[:first])

    def test_first_difference_in_punctuation_is_not_semantic_pair(self):
        source = record(rejected='{"text":"good","extra":1}')
        encoded, audit = encode_record(self.tokenizer, source, 1000)
        self.assertEqual(audit["pair_status"], "first_divergence_outside_semantic_value")
        self.assertEqual(encoded["critical_positions"], -1)

    def test_field_order_change_is_not_semantic_pair(self):
        source = record('{"text":"good","reason":"ok"}', '{"reason":"ok","text":"bad"}')
        encoded, audit = encode_record(self.tokenizer, source, 1000)
        self.assertEqual(audit["pair_status"], "first_divergence_outside_semantic_value")
        self.assertEqual(encoded["critical_positions"], -1)

    def test_token_straddling_semantic_field_cannot_hide_outside_edit(self):
        chosen = '{"text":"good","extra":1}'
        rejected = '{"text":"good","extra":2}'
        # Deliberately make each entire JSON a token. Both token offsets overlap
        # "good", but the sole changed character is in the unrelated extra field.
        vocabulary = {"<|unk|>": 0, "<|pad|>": 1, "<|im_end|>": 2,
                      "<|im_start|>": 3, "<think>": 4, "</think>": 5,
                      chosen: 6, rejected: 7}
        backend = Tokenizer(models.WordLevel(vocab=vocabulary, unk_token="<|unk|>"))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<|unk|>",
                                           pad_token="<|pad|>", eos_token="<|im_end|>")
        tokenizer.chat_template = CHAT_TEMPLATE
        source = record(chosen, rejected)
        encoded, audit = encode_record(tokenizer, source, 1000)
        self.assertEqual(audit["pair_status"], "first_divergence_outside_semantic_value")
        self.assertEqual(encoded["critical_positions"], -1)

    def test_identical_pair_retains_positive_supervision(self):
        source = record(rejected='{"text":"good"}')
        encoded, audit = encode_record(self.tokenizer, source, 1000)
        self.assertEqual(audit["pair_status"], "no_token_divergence")
        self.assertEqual(encoded["critical_positions"], -1)
        self.assertIn(self.tokenizer.eos_token_id, encoded["labels"])

    def test_missing_span_or_pair_retains_positive_supervision(self):
        source = record()
        source.pop("semantic_span")
        _, audit = encode_record(self.tokenizer, source, 1000)
        self.assertEqual(audit["pair_status"], "missing_semantic_span")
        source["rejected"] = None
        encoded, audit = encode_record(self.tokenizer, source, 1000)
        self.assertEqual(audit["pair_status"], "no_pair")
        self.assertEqual(encoded["critical_positions"], -1)

    def test_prompt_and_padding_are_masked_response_and_eos_are_supervised(self):
        source = record()
        first, _ = encode_record(self.tokenizer, source, 1000)
        longer = deepcopy(source)
        longer["messages"][1]["content"] += " visible evidence visible evidence"
        second, _ = encode_record(self.tokenizer, longer, 1000)
        batch = Collator(self.tokenizer.pad_token_id)([first, second])
        pad_start = len(first["input_ids"])
        self.assertGreater(batch["input_ids"].shape[1], pad_start)
        self.assertTrue((batch["labels"][0, pad_start:] == -100).all())
        self.assertTrue((batch["attention_mask"][0, pad_start:] == 0).all())
        self.assertTrue((batch["input_ids"][0, pad_start:] == self.tokenizer.pad_token_id).all())
        response_start = next(i for i, token in enumerate(first["labels"]) if token != -100)
        self.assertTrue((batch["labels"][0, :response_start] == -100).all())
        self.assertEqual(batch["labels"][0, pad_start - 1].item(), self.tokenizer.eos_token_id)
        for key in ("critical_positions", "positive_ids", "negative_ids"):
            self.assertEqual(tuple(batch[key].shape), (2,))

    def test_non_thinking_template_and_hard_length_limit(self):
        source = record()
        prompt = self.tokenizer.apply_chat_template(source["messages"], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))
        encoded, _ = encode_record(self.tokenizer, source, 1000)
        exact, _ = encode_record(self.tokenizer, source, len(encoded["input_ids"]))
        self.assertEqual(exact, encoded)
        with self.assertRaisesRegex(ValueError, "未截断数据"):
            encode_record(self.tokenizer, source, len(encoded["input_ids"]) - 1)

    def test_prompt_logit_slicing_preserves_full_forward_loss_and_gradients(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from ScopeCT.objective import scope_loss
        from ScopeSFT.runtime import trainer_class

        torch.manual_seed(9)
        config = Qwen3Config(vocab_size=len(self.tokenizer), hidden_size=16,
                             intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1,
                             head_dim=8, max_position_embeddings=512,
                             sliding_window=None, use_sliding_window=False,
                             attention_dropout=0.0)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        first = record()
        second = deepcopy(first)
        second["messages"][1]["content"] += " visible evidence visible evidence"
        batch = Collator(self.tokenizer.pad_token_id)([
            encode_record(self.tokenizer, first, 1000)[0],
            encode_record(self.tokenizer, second, 1000)[0],
        ])
        full_outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             use_cache=False)
        full_loss, _ = scope_loss(full_outputs.logits, batch["labels"],
                                  batch["critical_positions"], batch["positive_ids"], batch["negative_ids"])
        full_gradient = torch.autograd.grad(full_loss, model.lm_head.weight)[0]
        # Exercise exactly the production compute_loss method without constructing
        # a Trainer/Accelerate process or allocating an optimizer.
        trainer = object.__new__(trainer_class())
        trainer.ct_weight, trainer.margin = 0.2, 1.0
        trimmed_loss = trainer.compute_loss(model, batch)
        trimmed_gradient = torch.autograd.grad(trimmed_loss, model.lm_head.weight)[0]
        torch.testing.assert_close(trimmed_loss, full_loss)
        torch.testing.assert_close(trimmed_gradient, full_gradient)
        self.assertEqual(trainer.last_loss_components["ct_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
