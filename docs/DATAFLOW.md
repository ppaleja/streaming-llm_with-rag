# Streaming-LLM Dataflow (Plaintext Diagrams)

This document shows how tokens, KV cache, and attention patching interact during inference and evaluation.

## Streaming Chat Generation

```text
User Prompts
   |
   v
[tokenizer] --encode--> input_ids
   |
   v
[enable_streaming_llm] -> returns [StartRecentKVCache] and patches model attentions
   |
   v
Loop per prompt:
  [kv_cache.evict_for_space(past_kv, seq_len + max_gen_len)]
   |
   v
  [model.forward(input_ids, past_key_values, use_cache=True)]
         |
         v
       outputs (logits, past_key_values)
         |
         +--> greedy step: argmax last logit -> next token
         |
         +--> append to generated_ids, pretty-print partial text
         |
         +--> repeat model.forward with new token
         |
         v
  [kv_cache.__call__(past_key_values)]  (optional, e.g., in other loops)
   |
   v
  Bounded cache keeps: first N tokens + last M tokens
```

Notes:
- `StartRecentKVCache` ensures memory stays bounded by retaining a short prefix + recent tokens.
- Patched attention applies rotary embeddings with capped query positions and true key positions for cached tokens.

## Per-Token Evaluation (PPL)

```text
Dataset sample text
   |
   v
[tokenizer] --full encode--> input_ids (length L)
   |
   v
For idx in [0..L-2]:
  slice input_ids[idx:idx+1]  -> step token
   |
   v
  [model.forward(step token, past_key_values, use_cache=True)]
         |
         v
       outputs (logits, past_key_values)
         |
         v
       compute CE loss against input_ids[idx+1]
         |
         v
  if kv_cache: kv_cache(past_key_values)  # trim after each step
```

## Where Position Shift Happens

```text
[Patched Attention Forward]
   - Build Q, K, V (concatenate past if provided)
   - Compute rotary cos/sin for kv_seq_len
   - Apply rotary to:
       * Query: positions based on limited (min(cache_size, i)) index
       * Key:   positions are absolute within concatenated cache
   - Softmax(QK^T / sqrt(d)) -> Attention -> combine with V
   - Project to output; return with updated present/past
```

## Key APIs Referenced

- `enable_streaming_llm.enable_streaming_llm(model, start_size, recent_size)`
- `kv_cache.StartRecentKVCache.__call__`, `.evict_for_space`, `.evict_range`
- `pos_shift.modify_llama.enable_llama_pos_shift_attention`
- `pos_shift.modify_gpt_neox.enable_gpt_neox_pos_shift_attention`
- `pos_shift.modify_falcon.enable_falcon_pos_shift_attention`
- Example loops: `examples/run_streaming_llama.py`, `examples/eval_long_ppl.py`