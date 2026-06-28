Now I have a complete picture of every file in the codebase. Let me compile the comprehensive duplication report.
Code Duplication Report for jhtensor
# FINDING #1: CPU Backend DType Dispatch Boilerplate (HIGHEST IMPACT)
**Files affected:** /Users/jht/Uber/experiments/jhtensor/src/tensor/cpu_backend.rs

**Lines involved:** Every operation method, lines 22-53, 70-89, 105-124, 140-159, 173-195, 210-244, 267-298

Every backend operation in CPUBackend repeats this identical structure:

1. A locally-scoped macro that takes $t:ty
2. Unsafe pointer casts to create typed slices
3. A match dtype { Float32 => ..., Int32 => ..., Int16 => ... } dispatch

For example, compare `add_arrays_inplace` (lines 70-89) and `mul_arrays_inplace` (lines 105-124): both use this identical shell:
```
macro_rules! compute_ADDOROTHER { ($t:ty) => {{ ... }}; }
match dtype {
    DType::Float32 => compute_*(f32),
    DType::Int32   => compute_*(i32),
    DType::Int16   => compute_*(i16),
}
Ok(())
```

Within each macro body, the pointer-to-slice conversion is also duplicated. This exact pattern appears in 7 separate functions: `mat_mul_inplace`, `add_arrays_inplace`, `mul_arrays_inplace`, `sub_arrays_inplace`, `transpose_inplace`, `sum_axis_inplace`, `broadcast_inplace`.

**Suggestion**: Create a helper that performs the `dtype` dispatch and slice conversion. A utility like:
```
trait DTypeDispatch {
    fn dispatch<'a, R>(&self, bytes_a: &'a [u8], bytes_b: Option<&'a [u8]>, dest: &'a mut [u8], count: usize);
}
```

Or more practically, a macro at the module level that generates each operation by accepting the inner loop body as a parameter, reducing each function from ~40 lines to ~15.

The common pattern of `from_raw_parts` with `.cast::<$t>()` could be extracted into a helper struct (e.g., `TypedSlice<'a, T>`) so the unsafe casting is only in one place.

# FINDING #2: Metal Backend Element-wise Method Boilerplate (HIGHEST IMPACT)

**Files affected:** /Users/jht/Uber/experiments/jhtensor/src/tensor/metal_backend.rs

**Lines involved:** 119-177 (add), 179-237 (mul), 239-297 (sub)

The three element-wise operations `add_arrays_inplace`, `mul_arrays_inplace`, and `sub_arrays_inplace` are structurally identical. They differ only in:
- The pipeline name string constant (`"add_arrays_*"`, `"mul_arrays_*"`, `"sub_arrays_*"`)
- The method signature
Comparing the bodies of all three (lines 126-176 vs 186-236 vs 246-296), the entire flow is the same:
get_metal_context -> commandBuffer -> computeCommandEncoder -> get_pipeline(dtype match)
-> setComputePipelineState -> setBuffer x3 (a, b, dest) -> thread sizing -> dispatch -> endEncoding -> commit -> waitUntilCompleted

That's ~60 lines of exact duplication per method.

**Suggestion:** Create a private helper function:
```
fn dispatch_elementwise(
    pipeline_name: &str,
    a: &Self::Storage, b: &Self::Storage, dest: &mut Self::Storage,
    array_length: usize, dtype: DType,
) -> Result<(), TensorError> { ... }
```
Each of the three methods would then be ~5 lines: call get_metal_context, compute array_length = shape.product(), call the helper. This eliminates ~120 lines of duplication.
# FINDING #3: Metal Backend Pipeline Name Dispatch (HIGH IMPACT)
**Files affected:** /Users/jht/Uber/experiments/jhtensor/src/tensor/metal_backend.rs

**Lines involved:** 69-74, 139-144, 198-204, 259-264, 315-321, 373-379, 451-457
Every single Metal backend operation contains a nearly identical match dtype block to map DType to a pipeline name string:
```
ctx.get_pipeline(match dtype {
    DType::Float32 => "prefix_f32",
    DType::Int32   => "prefix_i32",
    DType::Int16   => "prefix_i16",
})
```
This pattern repeats 7 times across 7 operations. The only thing that varies is the prefix string.

**Suggestion:** Add a method to DType or a free function:
```
    fn pipeline_name(&self, prefix: &str) -> String {
        let suffix = match self { Float32 => "f32", Int32 => "i32", Int16 => "i16" };
        format!("{}_{}", prefix, suffix)
    }
```
This reduces each 4-line match block to a single function call.

# FINDING #4: Metal Backend Command Buffer Lifecycle (HIGH IMPACT)

**Files affected:** /Users/jht/Uber/experiments/jhtensor/src/tensor/metal_backend.rs
**Lines involved:** 60-63, 130-135, 191-196, 250-255, 307-313, 390-396, 467-473 (creation)
and lines 111-115, 171-175, 231-235, 291-296, 354-358, 423-427, 496-500 (completion)

Every Metal backend operation repeats:
1. Command buffer creation from command queue (~4 lines)
2. Compute encoder creation from command buffer (~3 lines)
3. The termination sequence endEncoding -> commit -> waitUntilCompleted (~3 lines)

That is ~10 lines x 7 operations = ~70 lines of boilerplate. Two slightly different error-handling styles are used: `.expect`(...) (`mat_mul`, `add`, `mul`, `sub`, `transpose`) and `.ok_or_else(|| TensorError::BackendFailure(...))? (sum_axis, broadcast).` The latter is actually superior since it integrates with the Result return type properly, so the expect variants should also be converted.

**Suggestion:** These could be handled by a helper like:
```
fn with_compute_encoder<F>(ctx: &MetalContext, body: F) -> Result<(), TensorError>
where F: FnOnce(&dyn MTLComputeCommandEncoder) { ... }
```
Or a closure-based approach that acquires the encoder, runs user code inside it, then handles endEncoding/commit/wait.

# FINDING #5: tensor.rs Element-wise Validation Boilerplate (MEDIUM-HIGH IMPACT)
**Files affected:** /Users/jht/Uber/experiments/jhtensor/src/tensor/tensor.rs
**Lines involved:** 54-81 (add_inplace), 137-164 (mul_inplace), 177-204 (sub_inplace)
The three `*_inplace` methods for add, mul, and sub are structurally identical. They all perform these exact four checks in the same order:
1. self.shape != other.shape -> ShapeMismatch
2. dest.shape != self.shape -> ShapeMismatch
3. self.dtype != other.dtype -> TypeMismatch
4. dest.dtype != self.dtype -> TypeMismatch
5. Call backend dispatch method

That is ~26 lines x 3 = ~78 lines of duplicated validation logic.

Similarly, the non-inplace wrappers add (lines 126-135), mul (lines 166-175), and sub (lines 206-215) are all identically structured: allocate empty buffer, construct result Tensor, call inplace, return.

**Suggestion:** Extract a private helper method:
```
fn validate_elementwise(&self, other: &Self, dest: &mut Self) -> Result<(), TensorError> {
    // the 4 checks above
    Ok(())
}
```
And for the non-inplace ops, extract:
```
fn elementwise<F>(&self, other: &Self, backend_op: F) -> Result<Self, TensorError>
where F: FnOnce(&Self, &Self, &mut Self) -> Result<(), TensorError> { ... }
```
Or more idiomatically in Rust, use a macro to generate all three elementwise ops with just the backend method name and operation name as parameters.
# FINDING #6: tensor.rs Inline Test Boilerplate (MEDIUM-HIGH IMPACT)
**Files affected:** /Users/jht/Uber/experiments/jhtensor/src/tensor/tensor.rs
**Lines involved:** 486-522 (add tests), 524-560 (add_inplace tests), 562-598 (mul tests), 600-636 (mul_inplace tests), 638-674 (sub tests), 676-712 (sub_inplace tests)
There are 6 test modules here for add/mul/sub (each with inplace and non-inplace variants). Each module contains the same two tests:

- test_tensor_shape_mismatch (or similar error name): creates mismatched shape tensors, verifies error
- test_tensor_type_mismatch: creates mismatched dtype tensors, verifies error
For inplace variants: test_dest_shape_mismatch and test_dest_type_mismatch.
The test structure is identical -- only the method being called differs. This section alone (lines 486-712) is 227 lines of heavily duplicated tests.
Suggestion: A single macro could generate all six modules:
macro_rules! test_elementwise_errors {
    ($method:ident, $variant:ident, $shape_err:ident, $type_err:ident) => { ... }
}
// or use rstest/proptest parameterized testing
FINDING #7: Integration Test _types.rs Macro Structure (MEDIUM IMPACT)
Files affected:
- /Users/jht/Uber/experiments/jhtensor/tests/add/add_types.rs
- /Users/jht/Uber/experiments/jhtensor/tests/sub/sub_types.rs
- /Users/jht/Uber/experiments/jhtensor/tests/mul/mul_types.rs
- /Users/jht/Uber/experiments/jhtensor/tests/transpose/transpose_types.rs
- /Users/jht/Uber/experiments/jhtensor/tests/sum_axis/types.rs
- /Users/jht/Uber/experiments/jhtensor/tests/broadcast/types.rs
- /Users/jht/Uber/experiments/jhtensor/tests/matmul/types.rs

All seven files follow the exact same pattern:
use:
```
jhtensor::tensor::{CPUBackend, MetalBackend, ...};

macro_rules! test_*_for { ($backend:ident, $t:ident) => { mod $t { use super::*; tests... } }; }
mod metal { test_*_for!(MetalBackend, f32); test_*_for!(MetalBackend, i32); test_*_for!(MetalBackend, i16); }
mod cpu   { test_*_for!(CPUBackend, f32); test_*_for!(CPUBackend, i32); test_*_for!(CPUBackend, i16); }
```
The only variation is the inner test names and assertions. The macro scaffolding and dispatch calls are identical across all 7 files.

**Suggestion:** Create a shared _types.rs pattern or shared_macros.rs module in tests/ that provides:
```
macro_rules! test_for_all_backends { ($op_macro:ident, $test_fn: expr) => { ... } }
```
However, the per-op tests are structurally different (matmul tests have dot product + full matmul, add has small + large, etc.), so this is a lower-value refactor. A metaprogramming approach (e.g., using rstest or derive-based test generation) might be more appropriate here than hand-written macros.

# FINDING #8: Integration Test fuzzy.rs Structure (MEDIUM IMPACT)
**Files affected:**
- /Users/jht/Uber/experiments/jhtensor/tests/matmul/fuzzy.rs
- /Users/jht/Uber/experiments/jhtensor/tests/broadcast/fuzzy.rs
- /Users/jht/Uber/experiments/jhtensor/tests/sum_axis/fuzzy.rs
- /Users/jht/Uber/experiments/jhtensor/tests/transpose/fuzzy.rs

All four fuzzy.rs files share:
1. #[macro_export] macro_rules! test_fuzzy declaration (with small variations in parameters)
2. RNG initialization with same seed (ChaCha8Rng::seed_from_u64(42))
3. Data generation via rng.random_range(-10..10)
4. CPU-as-ground-truth comparison pattern
5. Three size tiers: small (8..16), medium (16..64), large (64..128)
6.
The structural skeleton is identical. Each file is 50-73 lines of largely the same setup logic, with only the specific tensor operation (and its parameters) differing.
Suggestion: Consolidate into a single fuzzy.rs or fuzzy/mod.rs that parameterizes the test by operation type:
```
pub trait FuzzableOp {
    fn cpu_fwd(&self);
    fn gpu_fwd(&self);
    fn rng_ranges() -> Vec<(...) >;
}
```

Or use a single macro that accepts the forward-operation as a closure.

# FINDING #9: Integration Test metal_fuzzy.rs Files (LOW IMPACT)
**Files affected:**
- /Users/jht/Uber/experiments/jhtensor/tests/matmul/metal_fuzzy.rs (18 lines - slightly more due to linear/relu modules)
- /Users/jht/Uber/experiments/jhtensor/tests/broadcast/metal_fuzzy.rs (6 lines)
- /Users/jht/Uber/experiments/jhtensor/tests/sum_axis/metal_fuzzy.rs (6 lines)
- /Users/jht/Uber/experiments/jhtensor/tests/transpose/metal_fuzzy.rs (6 lines)
-
The broadcast, sum_axis, and transpose metal_fuzzy.rs files are byte-for-byte identical except for the macro imported:
```
use crate::test_fuzzy;
use jhtensor::tensor::MetalBackend;
test_fuzzy!(MetalBackend, f32);
test_fuzzy!(MetalBackend, i32);
test_fuzzy!(MetalBackend, i16);
```
Note: matmul's `metal_fuzzy.rs` additionally wraps these in linear and relu modules with an extra $activation parameter. These could also be consolidated once the fuzzy macros are unified (Finding #8).

# FINDING #10: DType Implementations in dtype.rs (LOW IMPACT)
**Files affected:** /Users/jht/Uber/experiments/jhtensor/src/tensor/dtype.rs
**Lines involved:** 23-39
Three nearly identical trait implementations:
```
impl TensorDType for f32 { fn dtype() -> DType { DType::Float32 } }
impl TensorDType for i32 { fn dtype() -> DType { DType::Int32 } }
impl TensorDType for i16 { fn dtype() -> DType { DType::Int16 } }
```
**Suggestion:** A macro at ~3 lines could generate all three. Though this is minor and clear as-is, so low priority.

# FINDING #11: Metal Backend Buffer Dispatch Pattern for Ops with Params (MEDIUM IMPACT)
**Files affected:** /Users/jht/Uber/experiments/jhtensor/src/tensor/metal_backend.rs
**Lines involved:** transpose (299-361), sum_axis (363-429), broadcast (431-502)

After handling the three element-wise ops, these three operations share a secondary pattern: get_metal_context -> create params struct on stack -> setBytes_length_atIndex for ptr/size -> dispatch with threadgroup sizing. While not as identical as the element-wise boilerplate (each has unique thread sizing logic), the command submission / params-passing boilerplate is still duplicated ~3x across these operations.

**Suggestion:** A helper to manage params passing:
```
fn set_params<F: Sized>(encoder: &MTLComputeCommandEncoder, params: &mut F, index: u64) { ... }
```
# Summary Table

1.	CPU DType dispatch macros
2.	Metal element-wise boilerplate
3.	Metal DType -> pipeline name
4.	Metal command buffer lifecycle
5.	tensor.rs elementwise validation
6.	tensor.rs inline test boilerplate
7.	Types test macro structure
8.	Fuzzy test framework structure
9.	metal_fuzzy dispatch files
10.	TensorDType impls
11.	Metal params-passing pattern

The biggest wins are **Finding #2** (Metal element-wise) and **Finding #4** (Metal command buffer lifecycle), which can be combined: extracting the common Metal compute dispatch pattern would reduce metal_backend.rs by approximately 250-300 lines. The CPU backend dedup (**Finding #1**) is also high value at ~280 lines across 7 operations.
