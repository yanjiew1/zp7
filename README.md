# ZP7 (Zach's Peppy Parallel-Prefix-Popcountin' PEXT/PDEP Polyfill)

This is a fast branchless replacement for the PEXT/PDEP instructions.
If you don't know what those are, this probably won't make much sense.

These instructions are very fast on Intel chips that support them
(3 cycles latency), but have a much slower implementation on AMD.
This code will be much slower than the native instructions on Intel
chips, but faster on AMD chips, and generally faster than a naive
loop (for all but the most trivial cases).

A detailed description of the algorithm used is in `zp7.c`.

# Usage
This is distributed as a single C file, `zp7.c`.
These two functions are drop-in replacements for `_pext_u64` and `_pdep_u64`:
```c
uint64_t zp7_pext_64(uint64_t a, uint64_t mask);
uint64_t zp7_pdep_64(uint64_t a, uint64_t mask);
```

There are also variants for precomputed masks, in case the same mask is used
across multiple calls (whether for PEXT or PDEP--the masks are the same for both).
In this case, a `zp7_masks_64_t` struct is created from the input mask using the
`zp7_ppp_64` function, and passed to the `zp7_*_pre_64` variants:
```c
zp7_masks_64_t zp7_ppp_64(uint64_t mask);
uint64_t zp7_pext_pre_64(uint64_t a, const zp7_masks_64_t *masks);
uint64_t zp7_pdep_pre_64(uint64_t a, const zp7_masks_64_t *masks);
```

Three #defines can change the instructions used, depending on the target CPU, as
listed below. If none of these symbols are defined, the code should portable to
any architecture.
* `HAS_CLMUL`: whether the processor has the
[CLMUL instruction set](https://en.wikipedia.org/wiki/CLMUL_instruction_set), which
is on most x86 CPUs since ~2010.  Using CLMUL gives a fairly significant
speedup and code size reduction.

This code is hardcoded to operate on 64 bits. It could easily be adapted
for 32 bits by changing `N_BITS` to 5, replacing `uint64_t` with `uint32_t`,
and modifying the popcount/bzhi intrinsics/polyfills. This will be slightly
faster and will save some memory for pre-calculated masks, but is left out
for simplicity. Smaller inputs could likewise be supported by similar modifications.
