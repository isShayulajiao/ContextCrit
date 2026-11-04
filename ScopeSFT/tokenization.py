"""Response-only supervision and exact first-divergence validation."""
from __future__ import annotations


NON_THINKING_SUFFIX = "<think>\n\n</think>\n\n"
ASSISTANT_PREFIX = "<|im_start|>assistant\n"


def configure_tokenizer(tokenizer):
    """Keep the checkpoint's EOS/pad vocabulary and require reliable offsets."""
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError("需要具有官方 chat_template 与 offset mapping 的 Qwen fast tokenizer")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 没有 EOS")
    # Qwen3.5 uses <|endoftext|> for padding and <|im_end|> for chat EOS.
    # Preserve those distinct official tokens when they are present.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def render_prompt(tokenizer, messages):
    """Use the checkpoint template's explicit text-only, non-thinking branch."""
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError("此训练入口只接受纯文本 message.content，不接受图片或视频")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    # Qwen3/Qwen3.5 hybrid checkpoints close an empty thinking block here.
    # Qwen3 Instruct-2507 is non-thinking-only and ends at the assistant header.
    # Accept those two complete headers, never an arbitrary/open thinking suffix.
    if not prompt.endswith((ASSISTANT_PREFIX, ASSISTANT_PREFIX + NON_THINKING_SUFFIX)):
        raise ValueError("chat_template 未生成 Qwen 非思考模式前缀；请使用官方 Qwen3 / Qwen3.5 tokenizer")
    return prompt


def encode_record(tokenizer, record, max_length):
    prompt = render_prompt(tokenizer, record["messages"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Encode the response separately so offsets remain relative to the original
    # chosen/rejected JSON, rather than to a rendered or trimmed chat transcript.
    chosen = tokenizer(record["chosen"], add_special_tokens=False, return_offsets_mapping=True)
    response_ids = chosen["input_ids"]
    if not response_ids or tokenizer.eos_token_id is None:
        raise ValueError("空答案或 tokenizer 没有 EOS")
    ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        meta = record["metadata"]
        raise ValueError(f"样本 {meta} 长度 {len(ids)} 超过 --max-length {max_length}。"
                         "请增加上限（需足够显存）或整理更短的合法训练轨迹；未截断数据。")
    result = {"input_ids": ids, "attention_mask": [1] * len(ids),
              "labels": [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id],
              "critical_positions": -1, "positive_ids": -1, "negative_ids": -1}
    audit = {**record["metadata"], "tokens": len(ids), "pair_status": "no_pair"}
    if record.get("rejected"):
        negative = tokenizer(record["rejected"], add_special_tokens=False, return_offsets_mapping=True)
        neg_ids = negative["input_ids"]
        t = next((i for i, (u, v) in enumerate(zip(response_ids, neg_ids)) if u != v), None)
        span = record.get("semantic_span")
        if t is None:
            audit["pair_status"] = "no_token_divergence"
        elif not span:
            audit["pair_status"] = "missing_semantic_span"
        else:
            def overlaps(offset, bounds, text):
                a, b = max(offset[0], bounds[0]), min(offset[1], bounds[1])
                return a < b and bool(text[a:b].strip())
            # A BPE token can straddle a field boundary. Merely overlapping a
            # semantic value does not prove that the changed characters belong
            # to it (the actual edit could be punctuation or another field).
            first_character = next((i for i, (u, v) in enumerate(zip(record["chosen"], record["rejected"]))
                                    if u != v), min(len(record["chosen"]), len(record["rejected"])))
            characters_inside = all(bounds[0] <= first_character < bounds[1]
                                    for bounds in (span["chosen"], span["rejected"]))
            if (not characters_inside
                    or not overlaps(chosen["offset_mapping"][t], span["chosen"], record["chosen"])
                    or not overlaps(negative["offset_mapping"][t], span["rejected"], record["rejected"])):
                audit["pair_status"] = "first_divergence_outside_semantic_value"
            else:
                result.update(critical_positions=len(prompt_ids) + t,
                              positive_ids=response_ids[t], negative_ids=neg_ids[t])
                audit.update(pair_status="valid", response_token_index=t,
                             positive_token=response_ids[t], negative_token=neg_ids[t],
                             semantic_span=span)
    return result, audit


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        import torch
        length = max(len(x["input_ids"]) for x in examples)
        out = {}
        for key, pad in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100)):
            out[key] = torch.tensor([x[key] + [pad] * (length - len(x[key])) for x in examples], dtype=torch.long)
        for key in ("critical_positions", "positive_ids", "negative_ids"):
            out[key] = torch.tensor([x[key] for x in examples], dtype=torch.long)
        return out
