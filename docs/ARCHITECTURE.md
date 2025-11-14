# Streaming-LLM Architecture (Plaintext Diagrams)

This document outlines the core modules, classes, and how they relate. Diagrams use plaintext boxes and arrows for easy viewing in Markdown.

## Modules Overview

```text
[streaming_llm]
  |-- enable_streaming_llm.py   -> enable_streaming_llm(model, start_size, recent_size)
  |-- kv_cache.py               -> StartRecentKVCache (class)
  |-- utils.py                  -> load(), parse_args(), download_url(), load_jsonl()
  |-- pos_shift/
        |-- modify_llama.py     -> enable_llama_pos_shift_attention(), patched forward
        |-- modify_gpt_neox.py  -> enable_gpt_neox_pos_shift_attention(), patched forward
        |-- modify_falcon.py    -> enable_falcon_pos_shift_attention(), patched forward
```

## Core Components (Class/Function Map)

```text
[enable_streaming_llm(model, start_size, recent_size)]
  - Detects model type via model.config.model_type
  - Enables position-shifted attention for supported backends
  - Constructs and returns a [StartRecentKVCache]

[StartRecentKVCache]
  - Fields: start_size, recent_size, cache_size, k_seq_dim, v_seq_dim
  - Methods:
      __call__(past_key_values)         -> keep first N and last M tokens in KV
      evict_for_space(pkv, num_coming)  -> pre-evict so upcoming tokens fit
      evict_range(pkv, start, end)      -> evict arbitrary subsequence

[utils]
  - load(model_name_or_path) -> (model, tokenizer)
  - parse_args(), download_url(), load_jsonl()
```

## Attention Patching (Position Shift)

The repo injects custom attention forward functions to adjust rotary embeddings so that queries use capped positions while keys retain true positions in cache.

```text
                        (by model type)
[enable_streaming_llm]  ------------------------------+
   |                                                  |
   | llama                                           | gpt_neox
   v                                                  v
 [enable_llama_pos_shift_attention]              [enable_gpt_neox_pos_shift_attention]
   |                                                  |
   | patches: LlamaAttention.forward                  | patches: GPTNeoXAttention.forward
   v                                                  v
[llama_pos_shift_attention_forward]               [gpt_neox_pos_shift_attention_forward]

   +-- falcon ----------------------------------------------+
   |  [enable_falcon_pos_shift_attention] -> patches Falcon self_attention.forward
   |  [falcon_pos_shift_attention_forward]
   +---------------------------------------------------------+
```

Key ideas:

- Query positions are limited (shifted) to avoid growing rotary positions; key positions follow true cache indices.
- Past key/values are concatenated, then rotary applied post-concat for keys.

## External Dependencies (conceptual)

```text
[transformers]  -> AutoModelForCausalLM, AutoTokenizer, model attention modules
[torch]         -> tensors, no-grad execution, matmul/softmax, dtype handling
```

## Usage Integration

```text
[utils.load] -> (model, tokenizer)
   |
   v
[enable_streaming_llm] -> StartRecentKVCache
   |
   v
Application loop uses model(..., past_key_values=..., use_cache=True)
and applies StartRecentKVCache methods to bound memory.
```
