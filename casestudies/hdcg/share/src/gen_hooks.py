#!/usr/bin/env python3
"""Generate share/grid-precision-hooks.html from the committed sources.

Run after editing any file it quotes:  python3 share/src/gen_hooks.py

Every 'real' code segment is read from the repo by line range, so the page
cannot drift from the code it shows.  Proposed and pseudo-code segments are
written here and labelled as such on the page.
"""
import json, pathlib, re

HERE = pathlib.Path(__file__).resolve().parent          # casestudies/hdcg/share/src
H = HERE.parents[1]                                       # casestudies/hdcg

def lines(path, a, b):
    L = (H / path).read_text().splitlines()
    return "\n".join(L[a-1:b])

def count(path):
    return len((H / path).read_text().splitlines())

rec = json.loads((H / "results/sw_v16_c0.json").read_text())
rec_text = json.dumps(rec, indent=2)

# ------------------------------------------------------------------ proposed
DECK = """{
  "study":    "fine-solver-links-and-vectors",
  "grid":     { "lattice": [16, 16, 16, 32], "mpi": [1, 1, 1, 4], "Ls": 16 },
  "operator": { "action": "Mobius", "mass": 0.01, "M5": 1.8, "b": 1.0, "c": 0.0 },
  "solver":   { "outer": "CG", "tol": 1e-8, "maxit": 10000,
                "preconditioner": "fp32 V-cycle behind MrhsMixedPrecPreconditioner" },
  "gauge":    [ "<config A>", "<config B>", "<config C>" ],
  "cases":    [ 0, 1 ],
  "variables": {
    "links":   [ "v24", "v16", "v12", "v10", "v8", "bf16" ],
    "vectors": [ "v24", "v16", "v12", "v10", "v8", "bf16" ]
  },
  "sweep":    "product",
  "output":   "results/{gauge}/case{case}.jsonl"
}"""

LOOP = """// PROPOSED -- not written.  Set up once, then sweep inside one process.
setup();                                  // gauge field, fp64 outer operator,
                                          // fp32 preconditioner: ~all of the cost
LatticeGaugeFieldF UmuF_exact = UmuF;     // rounding is in place, so keep a copy

for (const Point &pt : deck.points) {     // 6 x 6 = 36 (links, vectors) per config
  UmuF = UmuF_exact;                      // restore
  double d_links = Parsimony::RoundField(UmuF, pt.links);
  DiracF.ImportGauge(UmuF);               // re-derive the doubled links from it
  VecRound.spec = pt.vectors;             // read by the rounding operator below

  Result r = outer_solve(src, sol);       // fp64 CG, fp32 V-cycle preconditioner
  r.true_resid = exact_residual(src, sol);// judged with the fp64 operator
  r.delta = { {"links", d_links}, {"vectors", VecRound.worst} };
  append_jsonl(out, pt, r);               // one record per point
}"""

VECOP = """// PROPOSED -- rounds the vectors the preconditioner writes, not just its links.
template<class Field>
class RoundedHermOp : public LinearOperatorBase<Field> {
  LinearOperatorBase<Field> &_op;
public:
  Parsimony::RoundSpec spec;              // set per sweep point
  double worst = 0.0;                     // positive check, as for the links

  RoundedHermOp(LinearOperatorBase<Field> &op) : _op(op) {}
  void HermOp(const Field &in, Field &out) override {
    _op.HermOp(in, out);                  // the product is computed in fp32 ...
    Parsimony::RoundFieldFast(out, spec); // ... then stored narrow
  }
  // OpDiag, OpDir, Op, AdjOp forward the same way
};"""

PSEUDO = """# Pseudo-code of grid_eval.run() -- the search that drives Grid.

reference = run_driver({every variable: v53})       # calibrate the budget
for slack in [0, 0.1, 0.25]:                         # Pareto curve: bits vs iterations
    criterion = IterationBudget(slack)               # re-score cached runs, no new solves

    # 1. Prefilter: each variable alone, others at fp64.  Bisect the ladder.
    for var in variables:
        floor[var] = lowest k such that run_driver({var: k}) passes

    # 2. Delta debugging from those floors: mixed configurations.
    config = PrecisionDDMin(ladder).search(floors, run_driver)
    record(slack, config, run_driver(config))


def run_driver(config):
    if config in cache:                              # probes repeat; solves cost
        return cache[config]
    for case in cases:
        srun Test_coarse_bits  --round-<var> <fmt> ...  --case <case>  --json out.json
        rec = read(out.json)          # no file   -> error, not "diverged"
        for var rounded in config:
            assert rec.delta[var] > 0 # zero      -> RoundingNoOp, stop
    cache[config] = scores(recs)      # converged, iterations, true_resid
    return criterion.accept(cache[config])"""

PRODUCT = """# Pseudo-code of the product sweep the next study needs (not written).
# Two variables are cheap enough to map completely, and the map shows the
# interaction that one-at-a-time floors can hide.

deck = {links: ladder, vectors: ladder, gauge: [A, B, C], cases: [0, 1]}
write(deck, "deck.json")
srun Test_fine_bits --deck deck.json            # ONE launch per configuration
for rec in read_jsonl("results/*.jsonl"):
    check rec.delta > 0 for every rounded variable
    grid[rec.links][rec.vectors] = (rec.iterations, rec.true_resid)"""

ITEMS = [
  # ---------------------------------------------------------------- Python
  dict(id="py-search", group="Parsimony (Python)", title="The search that drives Grid",
       status="pseudo", file="casestudies/hdcg/grid_eval.py", start=None, lang="py", code=PSEUDO,
       what="Parsimony calibrates on an all-fp64 run, establishes one floor per variable, then searches mixed configurations with delta debugging. Every probe goes through run_driver, which launches the Grid binary once per case.",
       why="Pseudo-code of the real `run()` at grid_eval.py:271. This loop is tested end to end against a stand-in driver with planted knees, which it recovers exactly. It has not yet driven Grid on the machine: the published ladder came from the shell loop under The seam."),
  dict(id="py-command", group="Parsimony (Python)", title="Building one probe's command line",
       status="tested", file="casestudies/hdcg/grid_eval.py", start=140, lang="py", code=lines("grid_eval.py",140,146),
       what="A configuration becomes one `--round-<variable> <format>` pair per variable. The format is an argument, not a compile-time type, so one binary serves every configuration.",
       why="That is what `Caps(needs_rebuild=False)` declares to the search: no probe costs a rebuild."),
  dict(id="py-read", group="Parsimony (Python)", title="Reading the record, and refusing a silent no-op",
       status="tested", file="casestudies/hdcg/grid_eval.py", start=167, lang="py", code=lines("grid_eval.py",167,203),
       what="A missing JSON file is an error, not a divergence: a crashed driver must not be scored as 'this format failed'. Then every variable asked to round must report a non-zero change.",
       why="A hook on the wrong object gives a flat curve that looks exactly like 'this input needs no bits'. The check turns that into a stop."),
  dict(id="py-product", group="Parsimony (Python)", title="The product sweep the next study needs",
       status="proposed", file="(not written)", start=None, lang="py", code=PRODUCT,
       what="Two variables, every combination, one launch per gauge configuration.",
       why="The prefilter's floors are a heuristic, not a bound: a coarse format downstream can absorb upstream error. With two variables the full map is affordable and shows the interaction directly."),

  # ---------------------------------------------------------------- seam
  dict(id="seam-argv", group="The seam", title="What one probe sends",
       status="ran", file="casestudies/hdcg/jobs/run_sweep.slurm", start=24, lang="sh",
       code=lines("jobs/run_sweep.slurm",24,33),
       what="Today the parameters are command-line flags. The published 24-point ladder was produced by this shell loop, one `srun` per point on four A100s.",
       why="There is no input file today: the format list lives in the loop. That is fine for one variable and 12 formats, and it is the thing the proposed deck replaces."),
  dict(id="seam-deck", group="The seam", title="Proposed input deck",
       status="proposed", file="(not written)", start=None, lang="json", code=DECK,
       what="The next study's parameters in one file: the production solver's settings, three gauge configurations, two right-hand-side cases, and one ladder per variable swept as a product.",
       why="The operator and solver values shown are the stock HDCG test's, as placeholders until Joe confirms production's. The ladders start at v24 because that is the fp32 point production already runs, which makes it the sanity gate: v24 must reproduce production's iteration count exactly."),
  dict(id="seam-json", group="The seam", title="What comes back",
       status="ran", file="casestudies/hdcg/results/sw_v16_c0.json", start=None, lang="json", code=rec_text,
       what="One record per probe, written by rank 0. `iterations` is the rate; `true_resid` is judged against the exact operator; `delta` is the relative L2 change the rounding caused.",
       why="Every field that changes what the numbers mean is in the record (gauge, subspace, nbasis, Ls), so a cheap run can't later be mistaken for a physics one. Missing, and still to add: the rank count, `--mpi`, and the Grid commit."),

  # ---------------------------------------------------------------- our code
  dict(id="g-roundbits", group="Code we compile into the driver", title="The rounding function",
       status="ran", file="casestudies/hdcg/round_bits.h", start=50, lang="cpp", code=lines("round_bits.h",50,87),
       what="Round to nearest, ties to even, to k significand bits and the target format's exponent range. Not truncation: rounding toward zero would put a systematic bias on every entry.",
       why="Grid-free and callable on the device, so it can be tested where Grid isn't. It is bitwise-equal to parsimony's `formats.round_to` over 372 doubles, and on the A100 fp32 and bf16 came out identical to v24 and v8 to every digit."),
  dict(id="g-roundfield", group="Code we compile into the driver", title="Rounding a Grid field on the GPU",
       status="ran", file="casestudies/hdcg/round_field.h", start=34, lang="cpp", code=lines("round_field.h",34,72),
       what="Every double in every site object is snapped in place, one GPU thread per outer site. For a ComplexD field that rounds the real and imaginary part of every SIMD lane.",
       why="Returns the relative L2 change, the positive check the Python side insists on. It refuses single-precision fields by type name, because a ComplexF site is 8 bytes and would pass a size check. The production-matching study needs a float variant."),
  dict(id="g-parse", group="Code we compile into the driver", title="Reading the format flag",
       status="ran", file="casestudies/hdcg/Test_coarse_bits.cc", start=76, lang="cpp", code=lines("Test_coarse_bits.cc",76,84),
       what="`v53` means rounding off, and it is the default, so the unmodified run is always available as the reference.",
       why="An unrecognised name aborts instead of falling back. A typo in a sweep script must not quietly produce an fp64 point labelled bf16."),
  dict(id="g-twoops", group="Code we compile into the driver", title="Two operators: one to degrade, one to judge with",
       status="ran", file="casestudies/hdcg/Test_coarse_bits.cc", start=167, lang="cpp", code=lines("Test_coarse_bits.cc",167,172),
       what="Both coarse operators are built from the same aggregation, so they are identical before rounding.",
       why="The rounded operator converges accurately to its own, different, solution, so its own residual says nothing about accuracy. This is also half the setup cost; copying the stencil would halve it."),
  dict(id="g-hook", group="Code we compile into the driver", title="The hook: round the stored stencil",
       status="ran", file="casestudies/hdcg/Test_coarse_bits.cc", start=176, lang="cpp", code=lines("Test_coarse_bits.cc",176,198),
       what="This is the whole intervention: every one of the 81 stencil matrices, rounded in place, after the operator is built and before it is used.",
       why="Grid allocates `_Adag` only for a non-Hermitian operator. The first version indexed it unconditionally, read past an empty vector, and every rounded run segfaulted while the unrounded one worked, which pointed straight at the hook."),
  dict(id="g-solve", group="Code we compile into the driver", title="Solve, then judge with the exact operator",
       status="ran", file="casestudies/hdcg/Test_coarse_bits.cc", start=200, lang="cpp", code=lines("Test_coarse_bits.cc",200,228),
       what="Grid's own CG against the rounded operator, then the exact copy applied to the answer.",
       why="Two observables that answer different questions: iterations are the rate cost, true_resid the accuracy floor. In the run, the first never moved from 5 and the second tracked 2^-k."),
  dict(id="g-write", group="Code we compile into the driver", title="Writing the record",
       status="ran", file="casestudies/hdcg/Test_coarse_bits.cc", start=237, lang="cpp", code=lines("Test_coarse_bits.cc",237,254),
       what="Rank 0 writes one JSON object. `delta` is present only for a rounded variable, which is what lets the Python side tell 'not asked' from 'asked and did nothing'.",
       why="Written even when the solve fails: non-convergence is a data point, and only a missing file is a broken run."),
  dict(id="g-hdcg", group="Code we compile into the driver", title="The three HDCG hooks",
       status="blocked", file="casestudies/hdcg/Test_hdcg_parsimony.cc", start=254, lang="cpp",
       code=lines("Test_hdcg_parsimony.cc",254,269) + "\n  ...\n" + lines("Test_hdcg_parsimony.cc",287,311) + "\n  ...\n" + lines("Test_hdcg_parsimony.cc",362,375),
       lnums=list(range(254,270)) + [None] + list(range(287,312)) + [None] + list(range(362,376)),
       ranges="254–269, 287–311, 362–375",
       what="Subspace before coarsening, stencil after, deflation eigenvectors before import. The placement is the experiment: rounding the subspace first keeps the two-level method Galerkin-consistent, and rounding the stencil deliberately breaks it.",
       why="Compiles and runs up to a segfault in `MultiGeneralCoarsenedMatrix`'s constructor, which is Grid's deprecated class, not these lines. No hook has executed: hooks 2 and 3 sit after the crash, and hook 1, which sits before it, was never switched on in a run."),

  # ---------------------------------------------------------------- Grid library
  dict(id="p-perf", group="Changes to the Grid library", title="Patch 1: a build break on current NERSC defaults",
       status="ran", file="casestudies/hdcg/grid-patches/0001-perfcount-rdtsc-shim.patch", start=None, lang="diff",
       code=(H/"grid-patches/0001-perfcount-rdtsc-shim.patch").read_text().rstrip(),
       what="Grid replaces `__rdtsc` with a no-op under CUDA to keep `x86intrin.h` away from nvcc, but cudatoolkit 13.2 declares the real one first: 'cannot overload functions distinguished by return type alone'.",
       why="Both uses are host-side performance counters in Stat.cc, never in a kernel, so the fix is to use the real intrinsic. It does not touch numerics. cudatoolkit 12.4 is no escape: it refuses gcc-native/14, the Perlmutter default."),
  dict(id="p-adef", group="Changes to the Grid library", title="Patch 2: expose TwoLevelCGmrhs's iteration count",
       status="ran", file="casestudies/hdcg/grid-patches/0002-twolevelcgmrhs-expose-iterations.patch", start=None, lang="diff",
       code=(H/"grid-patches/0002-twolevelcgmrhs-expose-iterations.patch").read_text().rstrip(),
       what="The multi-RHS HDCG solver printed its iteration count and true residual but kept them. This adds the same three members ConjugateGradient already has, filled on both exits.",
       why="Without it an automated study has to scrape stdout. Both patches apply cleanly to 865c338d, and these two are the only changes to Grid's library: all the rounding lives in our driver."),

  # ---------------------------------------------------------------- amortizing
  dict(id="a-today", group="Amortizing setup", title="Today: one configuration per process",
       status="ran", file="casestudies/hdcg/results/*.json", start=None, lang="text",
       code="""per point, median of 24 (4x A100, one node)

  setup   53.6 s   aggregation + 2x CoarsenOperator
  solve    0.017 s CG, 5 iterations
  srun     ~3.8 s  launch and MPI start
  ------  ------
  wall    57.4 s

24 points          22.9 min      1.53 GPU-hours
solve / compute     0.05 %""",
       what="No. Grid does not loop today. Every point is a separate process that rebuilds the identical operator, rounds it one way, and measures for 17 ms.",
       why="The solve is 0.05% of the compute, so a search pays setup times evaluations and almost nothing else. Moving the loop into the driver is the one optimisation that matters."),
  dict(id="a-loop", group="Amortizing setup", title="Proposed: the loop inside the driver",
       status="proposed", file="(not written)", start=None, lang="cpp", code=LOOP,
       what="Set up once, keep an exact copy, then per point: restore, round, re-import, solve, record. For the fine operator the stored data is the gauge links, and `ImportGauge` re-derives the doubled links from the rounded copy.",
       why="Rounding is in place and destructive, which is why the restore is needed. 36 points would cost about one setup instead of 36. `ImportGauge` exists on WilsonFermion and WilsonFermion5D at 865c338d."),
  dict(id="a-vec", group="Amortizing setup", title="Proposed: rounding the vectors, not just the links",
       status="proposed", file="(not written)", start=None, lang="cpp", code=VECOP,
       what="Links are rounded once. Vectors are written on every operator application, so rounding them needs a hook that fires each time: an operator adaptor that stores each product narrow.",
       why="For domain-wall fermions one 4D link serves all Ls slices, so vectors are most of the mat-vec traffic (estimate ~960 B of spinors against ~72 B of links per 5D site at Ls = 16). This adaptor rounds only operator outputs, not the smoother's other updates, so it is a partial model; and the per-call rounding needs a fast path without the two-lattice L2 check."),
]

TRACE = ["py-search", "py-command", "seam-argv", "g-parse", "g-twoops", "g-hook",
         "g-roundfield", "g-roundbits", "g-solve", "g-write", "seam-json", "py-read"]

STATS = dict(
    rb=count("round_bits.h"), rf=count("round_field.h"),
    hook=198-176+1,
)

tpl = pathlib.Path(__file__).with_name("hooks_template.html").read_text()
out = tpl.replace("/*__ITEMS__*/[]", json.dumps(ITEMS)) \
         .replace("/*__TRACE__*/[]", json.dumps(TRACE)) \
         .replace("__RB__", str(STATS["rb"])).replace("__RF__", str(STATS["rf"])) \
         .replace("__HOOK__", str(STATS["hook"]))
# Standalone copy: the skeleton the artifact host would otherwise supply.
HEAD = ('<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>[hidden]{display:none!important}</style></head><body>\n')
(HERE.parent / "grid-precision-hooks.html").write_text(HEAD + out + "\n</body></html>\n")
if "--artifact" in __import__("sys").argv:                 # body only, for publishing
    (HERE / "grid-precision-hooks.artifact.html").write_text(out)
print("items", len(ITEMS), "trace", len(TRACE), STATS)
