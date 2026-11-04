"""Real fast-tokenizer checks without downloading model/tokenizer artifacts."""
from copy import deepcopy
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from ScopeSFT.tokenization import Collator, configure_tokenizer, encode_record, render_prompt


# Local text-only fixture follows the official Qwen3.5 user/system rendering and
# explicit non-thinking generation branch; it does not copy vision/tool handling.
# https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/chat_template.jinja
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + (message['content']|trim) + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
    "{% if enable_thinking is defined and enable_thinking is true %}{{ '<think>\\n' }}"
    "{% else %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}"
    "{% endif %}"
)

# Official Qwen3-4B-Instruct-2507 has no thinking branch: generation ends at
# this assistant header. Keep it covered alongside the Qwen3.5 hybrid template.
# https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/tokenizer_config.json
INSTRUCT_ONLY_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def make_tokenizer(texts):
    """A genuine Rust fast tokenizer with offsets, fitted only to local fixtures."""
    backend = Tokenizer(models.WordLevel(unk_token="<|unk|>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    specials = ["<|endoftext|>", "<|unk|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    backend.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=specials, min_frequency=1))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<|unk|>",
                                       pad_token="<|endoftext|>", eos_token="<|im_end|>",
                                       additional_special_tokens=["<|im_start|>", "<think>", "</think>"])
    tokenizer.chat_template = CHAT_TEMPLATE
    return configure_tokenizer(tokenizer)


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
            '{"说明":"证据","text":"good"}', '{"说明":"证据","text":"bad"}',
        ])

    def test_checkpoint_pad_and_chat_eos_remain_distinct(self):
        original_pad = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left"
        configure_tokenizer(self.tokenizer)
        self.assertEqual(self.tokenizer.pad_token, "<|endoftext|>")
        self.assertEqual(self.tokenizer.eos_token, "<|im_end|>")
        self.assertEqual(self.tokenizer.pad_token_id, original_pad)
        self.assertNotEqual(self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)
        self.assertEqual(self.tokenizer.padding_side, "right")
        self.tokenizer.pad_token = None
        configure_tokenizer(self.tokenizer)
        self.assertEqual(self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)

    def test_non_thinking_template_must_be_honored_and_content_must_be_text(self):
        source = record()
        expected = ("<|im_start|>system\nReturn JSON.<|im_end|>\n"
                    "<|im_start|>user\nJudge visible evidence.<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertEqual(render_prompt(self.tokenizer, source["messages"]), expected)
        source["messages"][1]["content"] = [{"type": "image", "image": "unused.png"}]
        with self.assertRaisesRegex(ValueError, "纯文本"):
            encode_record(self.tokenizer, source, 1000)
        self.tokenizer.chat_template = "{{ 'assistant' }}"
        with self.assertRaisesRegex(ValueError, "非思考模式"):
            encode_record(self.tokenizer, record(), 1000)
        self.tokenizer.chat_template = "{{ '<|im_start|>assistant\\n<think>\\n' }}"
        with self.assertRaisesRegex(ValueError, "非思考模式"):
            encode_record(self.tokenizer, record(), 1000)

    def test_instruct_only_qwen3_template_remains_non_thinking(self):
        self.tokenizer.chat_template = INSTRUCT_ONLY_CHAT_TEMPLATE
        source = record()
        prompt = render_prompt(self.tokenizer, source["messages"])
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", prompt)
        encoded, audit = encode_record(self.tokenizer, source, 1000)
        self.assertEqual(audit["pair_status"], "valid")
        prompt_count = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        self.assertTrue(all(target == -100 for target in encoded["labels"][:prompt_count]))
        self.assertEqual(encoded["labels"][prompt_count:-1],
                         self.tokenizer.encode(source["chosen"], add_special_tokens=False))

    def test_original_response_offsets_survive_unicode_and_whitespace(self):
        source = record(' \n{"说明":"证据","text":"good"}\n ',
                        ' \n{"说明":"证据","text":"bad"}\n ')
        encoded, audit = encode_record(self.tokenizer, source, 1000)
        self.assertEqual(audit["pair_status"], "valid")
        chosen = self.tokenizer(source["chosen"], add_special_tokens=False, return_offsets_mapping=True)
        start, end = chosen["offset_mapping"][audit["response_token_index"]]
        self.assertEqual(source["chosen"][start:end], "good")
        response_start = next(i for i, target in enumerate(encoded["labels"]) if target != -100)
        self.assertEqual(encoded["labels"][response_start:-1], chosen["input_ids"])
        self.assertEqual(encoded["labels"][-1], self.tokenizer.eos_token_id)

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

    def test_hybrid_model_response_loss_preserves_full_forward_loss_and_gradients(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from ScopeCT.objective import scope_loss
        from ScopeSFT.runtime import trainer_class

        torch.manual_seed(9)
        config = Qwen3_5TextConfig(vocab_size=len(self.tokenizer), hidden_size=16,
                                  intermediate_size=32, num_hidden_layers=2,
                                  num_attention_heads=2, num_key_value_heads=1,
                                  head_dim=8, max_position_embeddings=512,
                                  linear_num_key_heads=1, linear_num_value_heads=2,
                                  linear_key_head_dim=8, linear_value_head_dim=8,
                                  layer_types=["linear_attention", "full_attention"],
                                  rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                   "partial_rotary_factor": 0.5,
                                                   "mrope_section": [1, 1, 0]},
                                  attention_dropout=0.0)
        config._attn_implementation = "eager"
        model = Qwen3_5ForCausalLM(config).eval()
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
