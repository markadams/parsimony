# Grid fine-operator performance on Perlmutter: running notes

Started 2026-10-06. Frontier access has no date yet, so the fine-operator
kernel work starts on Perlmutter A100s. The kernel source is the same as on
Frontier; the machine is not.

## Build (2026-10-06)

- Source `/pscratch/sd/m/madams/Grid-dev`, Grid `develop` `3232b438` (Peter's
  last push, 2026-10-05 15:47 UTC) plus `grid/patches/` (branch
  `madams/cuda13-build`, also in `~/Codes/Grid`).
- Build `/pscratch/sd/m/madams/Grid-dev/build`, by `build_grid.sh`: CUDA 13.2,
  gcc-native/14, sm_80, `--enable-tracing=nvtx`. libGrid.a 188 MB, ~16 min at
  `-j8` on a login node. Logs in `/pscratch/sd/m/madams/gridperf/build*.log`.
- `benchmarks/Benchmark_dwf` and `Benchmark_dwf_fp32` built.
- Top-level `make` stops in `examples/`. That is upstream, not ours (see below);
  the library, tests and benchmarks are complete before it.

## Upstream problems found (candidates for a PR or a note to Peter)

1. `Grid/perfmon/PerfCount.h`: the `GRID_CUDA` `__rdtsc` shim fails under
   CUDA 13 ("cannot overload functions distinguished by return type alone").
   Patch 0001.
2. `--enable-tracing=nvtx` cannot build on CUDA 13. Patch 0002 fixes three
   things:
    - CUDA 13 ships only the header-only NVTX3 (no `nvToolsExt.h`, no `libnvToolsExt`).
    - `Tracing.h` includes the header inside `namespace Grid`. Once CUDA's own
      headers include NVTX3 at global scope, its types are undefined
      (`nvToolsExtPayload.h: identifier "nvtxDomainHandle_t" is undefined`).
    - `traceStart` and `traceStop` had empty bodies; `traceStart` returned nothing.
3. `examples/Example_pvdagm_3level_FourierSmoother.cc:62` includes
   `Grid/qcd/utils/FreeMobius5D_claude.h`, which is not in the repo. This breaks
   `make` in `examples/`.
4. `tests/Test_dwf_dslash_repro.cc:76`: `for(int blane;blane<vobj::Nsimd();blane++)`.
   The loop counter is never initialized (nvcc warning #549-D).

## Runs

`dwf_1node.slurm`: 1 node, 4 A100, `--grid 16.8.24.48 --mpi 1.1.2.2 -Ls 24`,
which gives 16x8x12x24 per GPU, Peter's production local volume. Modes:
`base`, `nsys`, `ncu`. Each run is kept whole in
`/pscratch/sd/m/madams/gridperf/runs/dwf_<mode>_<jobid>/`.

| job | mode | result |
|---|---|---|
| 59452275 | base, generic kernel, fp64 | Dw 1.174 ms/call/GPU, 995 GF/s/GPU; Deo 979 GF/s; halo exchange 0.386 ms/call (on-node, overlapped); norm diff 2e-31 |
| 59452383 | ncu, rank 0, 16 kernels, `--set full` | see "Counters" below |
| 59452384 | nsys | failed: `-t mpi` interposer calls MPI before Grid's `MPI_Init` (Grid starts the GPU first). Rerun without MPI tracing |
| 59456440 | ncu, `Benchmark_dwf_fp32` | see "fp32 interior" below |
| 59457808 | cells: base, ifirst, rr64, rr80, rr64+ifirst, unroll+ifirst, 2 reps | see FINDINGS-2026-10-06.md |
| 59457809 / 59457811 | ncu, build-rr64 / build-rr80 | spills 7.4 M / 2.8 M; DRAM 2310 / 877 MB |
| 59457812 | nsys, GRID_DHOP_INTERIOR_FIRST=1 | interior starts at +10 us, runs 862 us (was 756); comms hidden |
| 59452756 | nsys `-t cuda,nvtx` | see "Timeline" below; Dw 1.21 ms/call under nsys |
| 59452387 | base `--dslash-unroll` | Dw 1.118 ms/call (−5%), 1044 GF/s; Deo +9%; norm diff 2e-31 |
| 59452411 | base, `Benchmark_dwf_fp32` | Dw 0.505 ms/call, 2314 GF/s = **2.33x fp64**; norm diff 6e-14 |

Byte models for one Dw on 884,736 5D sites per GPU (A100-40GB peak 1555 GB/s):
- Grid's own `data_mem` (no neighbour reuse: 9 spinors + links/Ls per site, 1776 B): 1.57 GB, so 1.34 TB/s = 86% of peak.
- Perfect reuse (spinor in + out + links/Ls, 432 B): 0.38 GB, so 0.33 TB/s = 21% of peak.
The measured DRAM bytes (ncu `dram__bytes`) decide which regime the kernel is in.

Default Wilson kernel is `OptGeneric` (`WilsonKernelsInstantiation.cc:36`); every
Perlmutter output Peter committed also says GENERIC. `--dslash-unroll` selects the Nc=3
hand-unrolled kernel.

## Counters: one fp64 Dw, generic kernel (job 59452383, `ncu_summary.py`)

One `Dw` = 4 Gather, 1 DhopInterior, 4 Merge, 1 DhopExterior, launched in that
order. Times are ncu replay times. Their sum, 1136 us, matches the benchmark's
1174 us per call, so on one node the Dslash is the kernels run one after
another; the 386 us of halo exchange hides under the interior kernel.

| kernel | us | share | DRAM MB | % of DRAM peak | regs | occupancy theor / achieved | top stall |
|---|---|---|---|---|---|---|---|
| DhopInterior | 819 | 72% | 695 (534 R, 161 W) | 55% | 104 | 25% / 24% | long_scoreboard |
| DhopExterior | 149 | 13% | 195 | 84% | 128 | 25% / 23% | long_scoreboard |
| Gather x4 | 107 | 9% | 93 | 50–60% | 64 | 50% / 15–26% | long_scoreboard, lg_throttle |
| Merge x4 | 61 | 5% | 47 | 38–57% | 56 | 50% / 14–23% | lg_throttle, long_scoreboard |

**The interior kernel is latency-bound, not bandwidth-bound.**
- 104 registers per thread limit it to 16 warps per SM (`launch__occupancy_limit_registers`
  16, with 1-warp blocks), so 25% theoretical occupancy.
- 0.13 eligible warps per cycle, 10% issue-active, dominated by `long_scoreboard`
  (waiting on global loads). That leaves too few loads in flight to reach peak.
- Loads are coalesced (9.7 sectors per request, stores 16), so access pattern is not the problem.
- DRAM traffic is 695 MB: 1.8x the 382 MB minimum and well under the 1.57 GB
  no-reuse model. The neighbour reuse comes from L2 (2.5 GB of L2 traffic, 64% hit),
  not L1 (14% hit).

**Block size cannot fix this.** `--accelerator-threads 8` gives Nsimd 4 x 8 = 32-thread
blocks. The interior kernel's limit is registers, which block size does not change. Peter's
flat `--accelerator-threads` sweep on MI250X is consistent with that: it rules out
block-count occupancy, not register-limited latency. (A100 only: MI250X needs its own counters.)

**What it would be worth.** At the exterior kernel's 84% of peak with the same bytes, the
interior drops from 819 to about 536 us, which is -24% on the whole Dw. At the 382 MB
minimum as well, about 290 us. These are estimates from the counters, not measurements.

fp32 runs 2.33x faster than fp64, more than the 2x a bandwidth-bound kernel could gain.
That is consistent with latency, since fp32 halves the registers per value. To confirm,
profile the fp32 interior kernel.

### Next experiments (cheap, 1 node each)

1. ncu on the fp32 interior kernel: registers, occupancy, % of DRAM peak.
2. Cap registers on the Wilson kernels (`__launch_bounds__` or `-maxrregcount=64`):
   check for spills, then occupancy and time.
3. ncu on `--dslash-unroll`: the 5% gain, is it registers or instructions?

## Timeline: one fp64 Dw at boost clocks (job 59452756, rank 0, calls 100-102)

ncu runs kernels at base clock with cold caches, so its times are long. Take time from
this timeline and bytes from the counters. All kernels run on one stream (14); the
peer-to-peer halo copies run on stream 13.

| us from start of call | GPU stream 14 | stream 13 / host |
|---|---|---|
| 20–143 | 4 Gather kernels (123 us) | host waits for stream 14 (38–145) |
| 143–228 | **idle, about 85 us** | host: CommunicateBegin starts at ~196 |
| 228–984 | **DhopInterior, 756 us (62%)** | 4 peer copies, 21.2 MB, 207–588 (380 us, one after another, 37–82 GB/s) |
| 985–1058 | 4 Merge (73 us) | host waits for copies until ~592 |
| 1058–1211 | DhopExterior (152 us) | |

At boost clock, 695 MB in 756 us is 0.92 TB/s, 59% of DRAM peak: the same picture as the counters.

**Second finding: the interior kernel waits for the gather.** `DhopInternalOverlappedComms`
(`WilsonFermion5DImplementation.h:313`) runs Gather (with a barrier), then CommunicateBegin,
then Interior, then CommunicateComplete, Merge and Exterior. The interior needs no halo data,
but it starts only after the gathers (123 us) and the host gap (85 us): about 17% of the call
on the critical path. Running the gathers on a second stream, concurrently with the interior,
would take most of that off. That is an estimate: the gathers would compete with the interior
for bandwidth. An `#if 0` block in the same function shows an alternative ordering was considered.

On one node the halo copies (380 us) stay hidden under the 756 us interior. Across nodes they
go through MPI, which is where Peter measures 63% of the time with MPI in flight. The copies to
the four neighbours also run one after another on one stream.

## Summary of opportunities (A100, one node, estimates)

| | where | estimated saving on Dw |
|---|---|---|
| 1 | interior kernel: register-limited occupancy, latency-bound at 59% of DRAM peak | up to ~25% |
| 2 | gather + host gap before interior (overlap them) | up to ~17% |
| 3 | `--dslash-unroll` (measured) | 5% |
| 4 | fp32 fine operator (measured; changes precision) | 57% (2.33x) |

## fp32 interior kernel (job 59456440): the register explanation holds

| interior kernel | fp64 | fp32 |
|---|---|---|
| registers per thread | 104 | 69 |
| block (Nsimd x 8 threads) | 32 | 64 |
| warps per SM, register limit | 16 | 28 |
| occupancy theoretical / achieved | 25% / 24% | 44% / 40% |
| eligible warps per cycle | 0.13 | 0.57 |
| issue-active | 10% | 32% |
| DRAM bytes | 695 MB | 214 MB (0.31x) |
| L2 hit rate | 64% | 77% |

fp32 has 1.75x the resident warps and 4x the eligible warps, which is why it gains more than 2x.
It also moves only 0.31x the DRAM bytes of fp64, not the 0.5x that halving each value would
give, and its L2 hit rate rises 64% to 77%. Halving the working set lets neighbour spinors
stay in the 40 MB L2. So the fp64 kernel also loses L2 reuse: its 695 MB is 1.8x the 382 MB
minimum. Hypothesis, not tested: cache blocking the fp64 site loop for L2 reuse.

**Write-up: `grid/FINDINGS-2026-10-06.md`.** Jobs up to 59457812 ran on m1516_g; from now on m1489_g.
