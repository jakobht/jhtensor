# Plan: Train a Small GPT on TinyStories in Rust

## Current State

Tensor library with `Backend` trait, `CPUBackend` + `MetalBackend`. Ops available: matmul (with ReLU), add, transpose, sum_axis, broadcast. DataTypes: f32, i32, i16. No autograd, no model, no training loop.

## Architecture Decisions

**No autograd** - hardcoded forward/backward per layer. Simpler to debug, more explicit, fits the existing code style.

**Metal GPU throughout** - M2 Max has 40+ GPU cores. The `Backend` trait already abstracts this cleanly.

## Target Model

Small GPT-2 style transformer:
- **vocab_size**: ~50k (BPE tokenizer)
- **n_embd**: 512
- **n_layer**: 6
- **n_head**: 8
- **n_ctx**: 128 (short context = faster training, still good for TinyStories)
- **Params**: ~17M
- Should train to convergence in 3-6 hours on M2 Max

No biases to reduce parameter count and simplify backward pass. Tied input/output embeddings to halve the vocabulary projection parameters.

## Implementation Phases

### Phase 1: Missing Tensor Ops (~30% of effort)

Ops needed for transformer forward/backward that don't exist yet:

- `zeros_like` / `fill_scalar` - allocate and fill tensors with a value
- `mul_elementwise` - element-wise multiplication (LayerNorm, attention masking, softmax gradients)
- `sub` - element-wise subtraction (LayerNorm, residual gradients)
- `scalar_mul` - multiply tensor by a scalar
- `reduce_mean` / `reduce_var` - or reuse `sum_axis` with scalar operations
- `exp` - element-wise exponential (softmax)
- `softmax` along axis (or build from exp + sum + div)
- `index_select` / embedding lookup - gather rows by index tensor (token to embedding)
- `slice` / advanced indexing for picking logits at target positions
- Random initialization (`randn`, Xavier/He init)

Many of these are 3-5 line CPU loops and compact Metal kernels. The Backend trait extends naturally. Note: current ops are mostly 2D-only - some will need to support N-D or at least 1D + 2D for bias/broadcast additions.

### Phase 2: Tokenizer + Data Loading (~15%)

- Add `tokenizers` crate dependency (HuggingFace `tokenizers`)
- Load GPT-2 BPE tokenizer via its files (vocab.json + merges.txt) - download at runtime or ship with repo (theyre small, ~1MB total)
- `Dataset` struct: load TinyStories text files, tokenize into flat token array
- Simple sequential batching: chunk tokens into batches of `[batch_size, n_ctx]`, targets are shifted by 1

TinyStories dataset: a few hundred MB of .jsonl files. Simple download script or using huggingface_hub. Add `serde` + `serde_json` for parsing .jsonl. Consider using `reqwest` or just a shell script to download.

### Phase 3: Model Layers (~25%)

Each layer implements forward + backward manually:

1. **Embedding** - `index_select` into vocab*n_embd weight matrix + position embedding table. Backward: scatter-add gradients back to weights.
2. **LayerNorm** - subtract mean, divide by sqrt(var+eps), mul by gamma, add beta. (Or use gamma only since we drop biases elsewhere.) Backward: standard LayerNorm gradient.
3. **MultiHeadAttention** - QKV projections (matmul), reshape for heads, scaled dot-product attention (matmul + softmax + matmul), output projection (matmul). Backward: chain through each sub-step.
4. **FeedForward** - two matmuls with GELU/ReLU in between. Backward: standard MLP backward.
5. **TransformerBlock** - LayerNorm -> SelfAttention -> residual add -> LayerNorm -> FFN -> residual add. (Pre-LaNorm architecture for training stability.)

Each forward call allocates temporary buffers on the backend for intermediate activations needed during backward. Each backward call computes gradients into pre-allocated gradient buffers.

### Phase 4: Model Assembly + Optimizer (~15%)

- `GPT` struct holding all layers, config, and allocated forward/backward buffers
- Weight initialization (Xavier for linear layers)
- Cross-entropy loss: softmax over logits, gather at target index, negative log mean. Can do this as softmax + slice + sum_axis / count.
- Adam optimizer. Step = `w -= lr * m / (sqrt(v) + eps)`. Simple element-wise tensor operations per parameter group.

### Phase 5: Training Loop (~10%)

Training loop with:
- Learning rate warmup + cosine decay
- Gradient clipping (L2 norm)
- Progress logging every N steps
- `cargo run` as the entry point

### Phase 6: Extras (~5%)

- Tiny generation/sampling function to see model outputs
- Clean structure and tests for new ops

## File Structure After Implementation

```
src/
  lib.rs
  main.rs                    # training entry point
  shaders/                   # existing Metal kernels + new ones
  tensor/                    # existing tensor library (extended)
  model/
    embedding.rs
    layer_norm.rs
    attention.rs
    feed_forward.rs
    transformer_block.rs
    gpt.rs                   # full model assembly
    loss.rs
    optimizer.rs
  data/
    dataset.rs               # TinyStories loading + batching
    tokenizer.rs             # BPE tokenizer wrapper
    download.rs             # fetch TinyStories + tokenizer files
```

## Dependencies to Add

- `tokenizers` - BPE tokenizer
- `reqwest` or `ureq` - download TinyStories + tokenizer
- `serde` + `serde_json` - parse .jsonl dataset files
- (`rand` already in dev-deps, may move to deps for weight initialization on GPU)

## Risks / Considerations

- **Metal kernel complexity**: softmax and attention are more complex kernels than matmul. May need multi-step approach (exp -> reduce_sum -> div) instead of a single fused kernel.
- **Memory**: 17M param model + activations fits easily on M2 Max 64GB unified memory.
- **Current ops mostly 2D-only**: embedding lookup and some reductions need flexible indexing. This is the biggest gap in the current tensor API.
