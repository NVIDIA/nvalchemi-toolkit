# Speaker notes — Domain Decomposition (nvalchemi)

Audience: computational chemists, **not** parallelism experts. Talk about *what it enables*
(bigger, faster, correct simulations); use the analogies; skip jargon. 10 slides, ~15 min.
Advance → / space · back ← · `f` fullscreen.

**1 · Title.** "The domain-decomposition work — running MLIP simulations across many GPUs. One
design serves many models, and it's provably the same answer as one GPU."

**2 · Problem + design.** Two walls on one GPU: **memory** (big systems don't fit) and **time**
(MD is thousands of sequential steps). Fix = give each GPU a slice of atoms. Then the stack
(right): the model writes **zero parallel code** — one `distribution_spec`; the framework owns the
rest. Two ways to split: by **space** (Halo) or by **atom-list** (Graph-parallel). Punchline: the
hard part is staying bit-for-bit correct *while atoms move every step*.

**3 · Halo (MACE).** Walk the 5 steps vs the picture; green dashed ring = borrowed "ghost" shell.
**Punchline:** the shell is *refreshed* between layers, so it stays a constant thin width — never
grows with depth. Small shell → low memory → fit huge systems.

**4 · Graph-parallel (UMA).** Walk the 4 steps. Contrast with halo: **no geometry** — even split of
the atom list + an all-gather each layer so message passing is exact. Great for dense systems
(where a halo shell would cover almost the whole box — which is why UMA uses this).

**5 · Bring your own — everything.** THE design slide. Emphasize this is a *central tenet*: every
layer is swappable and declared, not hard-wired.
- **Model:** any PyTorch MPNN distributes with a small spec, no rewrite (we ship MPNN + graph-
  transformer examples).
- **Strategy:** Halo + Graph-parallel behind one interface; add a new scheme without touching
  models or dynamics.
- **Hook:** custom per-step logic (monitoring/control) that runs correctly across all GPUs.
- **Dynamics:** any integrator (NVE/NVT/NPT/NPH/FIRE); and *compose* models (potential + long-
  range electrostatics) in a pipeline.
- Bottom strip = the feature surface + the correctness guarantee (~1e-6 vs one GPU). Don't dwell —
  it's the "and it all actually works" line.

**6 · MACE full MD (NVT) — capacity.** The honest end-to-end MD picture (only MACE slide — the
forward-pass slide was removed because its per-atom trend contradicts the MD reality; see below).
Left: actual MD-step times vs system size, per GPU count — 1 GPU stops at ~37k; more GPUs reach far
larger. Right: **8 GPUs run 158k-atom MD** (~4.3× the single-GPU limit). **Key framing:** the full
MD loop adds real per-step work (neighbor-list rebuild + halo exchange + atom migration +
consolidation), so at these box sizes MACE-halo is a **capacity play** — it runs MD that won't fit
one GPU; the per-step *speedup* regime is much larger N (halo efficiency ~ shell-width / box-edge).
Don't oversell speed here — **sell capacity.** `[YOUR PLOT NOTE]` (Data note: the w2/30k point is
bimodal in the raw log; I use its steady-state value ~915 ms.)

*Why no forward slide:* a forward-only benchmark (model + forces + halo, no NL-rebuild/migration/
consolidation) shows per-atom cost *falling* with GPUs, which flatly contradicts the MD numbers
above. The two measure different things; showing both invites a "which is it?" objection. MACE's
honest story is capacity.

**7 · UMA speed & capacity.** The hero speed plot: graph-parallel UMA scales almost ideally —
4,394 atoms is **4.5× faster on 8 GPUs**. Capacity grows ~9× (1→8 GPUs). `[YOUR PLOT NOTE]`

**8 · UMA efficiency & memory.** µs/step/atom falls with size + GPUs; memory per GPU ~halves each
doubling. `[YOUR PLOT NOTE]`

**9 · vs fairchem (the two-part story).**
- **Left card — raw forward:** our distributed UMA *matches fairchem's own graph-parallel* kernels —
  same compute, ~zero wrapper overhead. So we're not comparing a different/faster model.
- **Right — full NVT MD at 8 GPUs (apples-to-apples, both full loops):** fairchem ms vs our ms vs
  speedup (fairchem ÷ ours, >1 = we're faster). Tie at 2k (0.89×), then we pull ahead as N grows:
  **1.11× → 1.20× → 1.25×** through 8k–24k, holding ~1.24× at 31k. fairchem's ASE loop pays a
  per-step conversion + graph-rebuild cost our integrated loop avoids.
- One-liner: "Same raw compute — our integrated loop wins as systems grow."
- (Note: this is NVT-vs-NVT. Comparing our *forward* to their MD step would look even better —
  ~1.3–1.5× — but that's not a fair workload match, so we don't lead with it.)

**10 · Multi-node (16 GPU / 2 nodes, InfiniBand).** Both stacks scale across nodes. Below ~40k
atoms fairchem's leaner cross-node comm leads; **at ≥40k we overtake by 1.1–1.3×** — same crossover
shape as single-node, shifted right by cross-node comm, and this is the regime production runs live
in. Ours is also steadier at the top (fairchem had two transient NCCL hiccups at 65k/71k). `[YOUR
PLOT NOTE]` Data caveats if pressed: fairchem's 39k point is a bit anomalous (re-run pending); IB is
required (TCP-only was ~2–3× slower). Known lever to move the crossover left: reduce-scatter the
force reduction (a reviewer flagged the same thing).

**11 · Takeaways.** Re-hit the five bullets; close.

---

## Q&A backstops
- **Exact?** Yes — validated per model vs single-GPU; force diff ~1e-6 eV/Å (FP/TF32 noise). Halo is
  the correctness reference.
- **UMA with halo?** Poor fit for dense boxes (ghost shell ≈ whole system) and can drift at
  small-N/high-GPU; graph-parallel is the intended UMA strategy and stays exact.
- **vs data parallel?** Data parallel replicates the whole model and splits *batches*. We split a
  *single system's atoms* — that's what lets one giant structure span GPUs.
- **Multi-dimensional?** Built on a device mesh (the standard primitive); today a single domain axis
  + pipeline model-composition, designed to extend to multiple axes.
- **Metrics:** "µs/step/atom" = wall-time per step ÷ atom count — lower is better; it strips out
  system size so you can compare efficiency across sizes and GPU counts.
- **Hardware:** H100 80GB, 1–8 GPUs, one node. MACE = cuEquivariance + torch.compile; UMA = fairchem
  "turbo" (compiled). MACE slide 6 = forward pass, slide 7 = NVT MD; UMA plots are NVT MD steps.
- **Why MACE MD looks like it "anti-scales" at small N:** halo efficiency ≈ shell-width / box-edge;
  at small boxes the borrowed shell is a large fraction of the region, so DD overhead dominates. It
  becomes a speedup at very large N (~1–2M atoms). At the sizes shown, DD's value is *capacity*.
