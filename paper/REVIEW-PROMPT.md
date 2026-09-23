# Writing-quality review pass — prompt for a fresh session

Paste the section below into a new Claude Code session started in
`~/Codes/parsimony`, on Fable.

---

You are doing a **writing-quality pass** on `paper/parsimony.tex`, a ~18-page
arXiv draft. The technical content, all measurements, and the argument are
settled and **must not change**. Do not rerun experiments, do not alter numbers,
do not restructure sections. If you believe a claim is technically wrong, stop
and say so rather than editing it.

Build with `make` in `paper/` (pdflatex + bibtex). It must still compile with
zero undefined references when you are done.

## What to fix

**1. Vague words where a technical term is available.** The author's specific
complaint is words like "reach" standing in for something precise. There are
**nine occurrences of the reach/reaches/reached/reaching family**; each needs the
exact term or a rewrite. But only where the precise term is genuinely correct —
do not invent precision the measurement does not have.

Others of the class to sweep for: "hit", "get to", "handle", "deal with",
"affect", "impact", "leverage", "utilize", "performant", "significant" (unless
statistical), "dramatically", "vastly", "nicely". Prefer the domain word:
*converge to*, *saturate*, *attain*, *is bounded by*, *dominates*, *amortizes*,
*bandwidth-bound*, *latency-bound*, *quantizes onto*, *underflows*.

**2. Ambiguous sentences.** Every sentence must have exactly one reading. Look
hardest at:
- Dangling or ambiguous antecedents — "this", "it", "that", "which" whose
  referent is a clause rather than a noun, or could be either of two nouns.
- Ratios and comparisons without a stated direction. This paper is full of
  speedups and slowdowns, and "1.49× slower" / "block ÷ scalar" / "0.81×" are
  easy to misread. Every ratio must make clear **what is divided by what** and
  **which direction is better**. Check each one against the surrounding table.
- Scope of negation and of "only".
- Attachment of trailing clauses ("..., which is why ...") to the wrong noun.

**3. The author's general writing rules** (from `~/.claude/CLAUDE.md` and PETSc's
`AGENTS.md`; note `~/.claude/styles.md` does **not** exist, do not go looking):
- Lead with the result, then what verifies it, then the limitations.
- Clear, grammatical, concise prose for peer mathematicians and engineers.
- Explain unfamiliar terms; preserve exact API names, commands, and diagnostics
  verbatim (`MATBAIJKOKKOS`, `-log_view_gpu_time`, `seqbaijkokkos`, `VecTDot`).
- Remove repetition and boilerplate. No process narration, no edit history, no
  account of how the work unfolded.
- Cut hedging that carries no information, but **keep** every stated limitation —
  those are load-bearing, not hedges.

**4. Mechanical consistency.**
- One convention for multiplication signs, ranges (en dashes), and units.
- `\texttt{}` for every format name, option and type; consistent across the paper.
- Spelling is currently consistent British ("behaviour", 4 occurrences, no
  American variants). Keep it that way; do not convert.
- Table captions self-contained; every table and figure referenced in the text.

## What NOT to do

- Do not soften or remove the negative results. The paper deliberately reports a
  failed prediction (the Jacobian degrading to Picard rather than diverging),
  three experimental-design errors, a scalar-CSR case where reduced precision
  buys nothing, and two corrections the author caught. These are the paper's
  credibility and several are its best material.
- Do not remove the *Use of AI* section's two recorded corrections.
- Do not "improve" the abstract's structure or the section order.
- Do not touch `refs.bib` entries marked `[unverified]` — flag them in your
  summary instead.

## Deliverable

1. Edit `paper/parsimony.tex` in place.
2. Rebuild; confirm zero undefined references and report the page count.
3. Run `.venv/bin/pytest -q` — it must stay at 77 passing (you should not have
   touched code, so this is a guard).
4. Commit with a message following the author's convention: lead with the result,
   roughly 100 words of body, bullet list only if there is more than one
   significant change. Sign
   `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
5. Push to `origin main`.
6. Report: the count of vague-term substitutions, the count of ambiguity fixes,
   any sentence you could not disambiguate without a technical decision from the
   author, and any `[unverified]` citation that should be checked before
   circulation.

The paper is about to go to co-authors for review, so the bar is: a reader who
knows numerical software but not this project should get every sentence right on
one pass.
