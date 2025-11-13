# Copilot / AI Agent Instructions

Purpose: help an AI coding agent become productive quickly in this repository (StreamingLLM).

- **Big picture**: StreamingLLM enables efficient, infinite-length decoding by (1) keeping a small set of "start" tokens as attention sinks and (2) keeping a sliding "recent" window of K/V cache entries. The code hooks into HuggingFace Transformer model implementations and monkey-patches their attention `forward` methods to apply a "position-shift" strategy and to manage a `StartRecentKVCache`.

- **Key locations**:
  - `streaming_llm/enable_streaming_llm.py` — entry point for enabling streaming behavior; picks model-type-specific settings and returns a `StartRecentKVCache` instance.
  - `streaming_llm/kv_cache.py` — `StartRecentKVCache` implementation; contains the slicing helpers (`DIM_TO_SLICE`) and eviction logic (`evict_for_space`, `evict_range`).
  - `streaming_llm/pos_shift/` — per-backend monkey-patches for attention (LLama, GPT-NeoX, Falcon). Each file replaces the model's attention `forward` with a pos-shift aware implementation.
  - `streaming_llm/utils.py` — model loading, CLI arg parsing, and small helpers used by examples; note `trust_remote_code=True` and `device_map="auto"` usage.
  - `examples/run_streaming_llama.py` — runnable demo that shows how streaming is enabled and used in generation loops.
  - `README.md` — authoritative developer instructions for environment setup and example runs.

- **Developer workflows / commands** (executable from repo root):
  - Create environment & install (from README):

    ```bash
    conda create -yn streaming python=3.8
    conda activate streaming
    pip install torch torchvision torchaudio
    pip install transformers==4.33.0 accelerate datasets evaluate
    python setup.py develop
    ```

  - Run the example chatbot (uses GPU if available):
    `CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py --enable_streaming --start_size 4 --recent_size 2000`

- **Important implementation patterns** (do not change lightly):
  - Monkey-patching: `pos_shift` modules traverse model submodules and set `module.forward = types.MethodType(new_forward, module)`. New model support must follow this pattern.
  - `StartRecentKVCache` uses numeric `k_seq_dim` / `v_seq_dim` to select slicing helpers. When adding a model type, choose correct dims for `k_seq_dim`/`v_seq_dim` in `enable_streaming_llm.py`.
  - Model loading uses `AutoTokenizer` / `AutoModelForCausalLM` with `trust_remote_code=True` and `device_map='auto'` — some models use custom HF code; keep `trust_remote_code` unless you add local model adapters.
  - The code assumes `use_cache=True` on model calls and works by manipulating `past_key_values` returned by the model.

- **When adding support for a new model family**:
  1. Add a new `streaming_llm/pos_shift/modify_<model>.py` implementing a pos-shifted attention `forward` matching that model's attention class.
  2. Update `enable_streaming_llm.py` to detect `model.config.model_type` and set `k_seq_dim`/`v_seq_dim`, and import & run the `enable_<model>_pos_shift_attention(model)` function.
  3. Add unit/functional tests (preferred) or run `examples/run_streaming_llama.py` manually for the model.

- **Integration / external dependencies**:
  - Heavily depends on HuggingFace `transformers` (some models rely on remote/trust_remote_code implementations).
  - GPU/FP16 is assumed in examples (`torch_dtype=torch.float16`). Some TP (tensor-parallel) setups may be incompatible — `utils.load` warns about tensor parallel bugs for Falcon.

- **Gotchas & quick notes**:
  - The project does not change the model context window; StreamingLLM retains only selected KVs.
  - `StartRecentKVCache` prints its sizes at construction — useful quick debug.
  - There are no formal tests in the repo; run `examples/` entry points for manual verification.

If any section is unclear or you want more detail (e.g., a quick patch template for adding a new model, or specific places to add tests), tell me which part and I will expand or adapt this file.
