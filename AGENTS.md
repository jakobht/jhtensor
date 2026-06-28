# AGENTS.md - Project Context for jhtensor

## What is this project?
A Rust tensor library with CPU + Apple Metal GPU backends, being extended into a small GPT training framework to train on TinyStories dataset.

## Goal
Train a small GPT (12-50M params) on TinyStories that can run and train in hours on a Mac M2 MAX.

## Architecture

### Current state
- Tensor library with backend abstraction (`Backend` trait)
- Two backends: `CPUBackend` and `MetalBackend`
- Supported ops: matmul (with ReLU), add, transpose, sum/reduce axis, broadcast
- Data types: `f32`, `i32`, `i16`
- No autograd, no model code, no training loop yet

### File structure
```
src/
  lib.rs              - re-exports tensor module
  tensor/
    mod.rs            - Backend trait definition
    tensor.rs         - Tensor struct + ops (matmul, add, transpose, sum, broadcast)
    cpu_backend.rs    - CPU implementations
    metal_backend.rs  - Metal GPU implementations (via objc2-metal)
    dtype.rs          - DType enum + TensorDType trait
    shape.rs          - Shape struct
    opts.rs           - Activation enum (None, ReLU)
    tensor_error.rs   - Error types
```

## Design conventions
- Manual backward pass per op (not autograd/comp graph) - simpler and more explicit
- Inplace operations are the primary API pattern
- Backend is a generic parameter on Tensor (`Tensor<B: Backend>`)
- Storage is backend-specific (`B::Storage`)
- Tests use `CPUBackend` as the default test backend, and tests the `MetalBackend` (and future additions) against it
- No comments in code unless absolutely necessary
- Follow existing Rust idioms and crate patterns

## Model target
- Tiny GPT (~12M params, GPT-2 small architecture)
- Should train overnight on Mac M2

## Key dependencies to add
- `tokenizers` - BPE tokenizer for TinyStories
- Random number generation already available via `rand` (dev-dep)
