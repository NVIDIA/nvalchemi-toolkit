<!-- markdownlint-disable MD014 MD033 -->

(distributed_design_overview)=

# Distributed ML Potentials: Design Overview

A guided tour of the distributed framework. Reads top-to-bottom as a
30-minute talk; each section is a slide group anchored by a figure
and a code block. Cross-links into the deeper user-guide chapters
where appropriate.

| Section | Question it answers |
|---|---|
| 1. Motivation | Why does naïve domain decomposition break for MPNNs? |
| 2. ShardTensor | What primitive lets distribution stay invisible to the model? |
| 3. Specs | How does the framework know *what* to do at each op? |
| 4. MACE end-to-end | What does "halo MPNN" actually look like? |
| 5. Composition | How do MACE + Ewald run in the same pipeline? |
| 6. Warp / Triton kernels | How do opaque kernels participate? |
| 7. Validation + BYO | How does a new model author go from zero to production? |

---

## 1. Motivation: domain decomposition meets message passing

### 1.1 The starting point: classical DD works for short-range pair potentials

Spatial decomposition is the standard way to scale molecular dynamics:
each rank owns a region of the simulation cell, computes forces for
its atoms, and exchanges a thin shell of "ghost" atoms with neighbors
for pair interactions whose cutoff crosses the boundary. For a pair
potential like Lennard-Jones, the math is local and the comms are
cheap.

```{graphviz}
:caption: Classical halo decomposition for a short-range pair potential. The global atom array is split into rank-owned contiguous slices; each rank materialises a thin shell of remote atoms within ``cutoff`` of its boundary (dashed). Pairs that cross the boundary are evaluated locally on either rank, no message is in flight at force-eval time.
:align: center

digraph halo_classical {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];

    global_view [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="16"><FONT POINT-SIZE="10"><B>global atom array (16 atoms)</B></FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8">0</TD><TD BGCOLOR="#9fc5e8">1</TD><TD BGCOLOR="#9fc5e8">2</TD><TD BGCOLOR="#9fc5e8">3</TD>
          <TD BGCOLOR="#9fc5e8">4</TD><TD BGCOLOR="#9fc5e8">5</TD><TD BGCOLOR="#9fc5e8">6</TD><TD BGCOLOR="#9fc5e8">7</TD>
          <TD BGCOLOR="#f6b26b">8</TD><TD BGCOLOR="#f6b26b">9</TD><TD BGCOLOR="#f6b26b">10</TD><TD BGCOLOR="#f6b26b">11</TD>
          <TD BGCOLOR="#f6b26b">12</TD><TD BGCOLOR="#f6b26b">13</TD><TD BGCOLOR="#f6b26b">14</TD><TD BGCOLOR="#f6b26b">15</TD>
        </TR>
        <TR>
          <TD COLSPAN="8"><FONT POINT-SIZE="10">rank 0 owns</FONT></TD>
          <TD COLSPAN="8"><FONT POINT-SIZE="10">rank 1 owns</FONT></TD>
        </TR>
      </TABLE>
    >];

    rank0_view [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="12"><FONT POINT-SIZE="10"><B>rank 0's local view</B>: 8 owned + 4 shell rows</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8">0</TD><TD BGCOLOR="#9fc5e8">1</TD><TD BGCOLOR="#9fc5e8">2</TD><TD BGCOLOR="#9fc5e8">3</TD>
          <TD BGCOLOR="#9fc5e8">4</TD><TD BGCOLOR="#9fc5e8">5</TD><TD BGCOLOR="#9fc5e8">6</TD><TD BGCOLOR="#9fc5e8">7</TD>
          <TD BGCOLOR="#cfe2cf" STYLE="dashed">8</TD><TD BGCOLOR="#cfe2cf" STYLE="dashed">9</TD><TD BGCOLOR="#cfe2cf" STYLE="dashed">10</TD><TD BGCOLOR="#cfe2cf" STYLE="dashed">11</TD>
        </TR>
        <TR>
          <TD COLSPAN="8"><FONT POINT-SIZE="10">my owned</FONT></TD>
          <TD COLSPAN="4"><FONT POINT-SIZE="10">shell (read-only)</FONT></TD>
        </TR>
      </TABLE>
    >];

    rank1_view [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="12"><FONT POINT-SIZE="10"><B>rank 1's local view</B>: 4 shell + 8 owned rows</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#cfe2cf" STYLE="dashed">4</TD><TD BGCOLOR="#cfe2cf" STYLE="dashed">5</TD><TD BGCOLOR="#cfe2cf" STYLE="dashed">6</TD><TD BGCOLOR="#cfe2cf" STYLE="dashed">7</TD>
          <TD BGCOLOR="#f6b26b">8</TD><TD BGCOLOR="#f6b26b">9</TD><TD BGCOLOR="#f6b26b">10</TD><TD BGCOLOR="#f6b26b">11</TD>
          <TD BGCOLOR="#f6b26b">12</TD><TD BGCOLOR="#f6b26b">13</TD><TD BGCOLOR="#f6b26b">14</TD><TD BGCOLOR="#f6b26b">15</TD>
        </TR>
        <TR>
          <TD COLSPAN="4"><FONT POINT-SIZE="10">shell (read-only)</FONT></TD>
          <TD COLSPAN="8"><FONT POINT-SIZE="10">my owned</FONT></TD>
        </TR>
      </TABLE>
    >];

    global_view -> rank0_view [label=<<FONT POINT-SIZE="10">slice + add shell</FONT>>];
    global_view -> rank1_view [label=<<FONT POINT-SIZE="10">slice + add shell</FONT>>];
}
```

```{code-block} python
:caption: The mental model: per-rank locality holds because pair interactions decay with distance.
# Single-process: O(N) atoms, O(N²) pairs (with cutoff: O(N))
for atom in batch:
    for neighbour in atom.within(cutoff):
        accumulate_force(atom, neighbour)

# Halo distributed: each rank does the same loop on its
# (owned + halo) atoms, only writing forces for owned.
```

### 1.2 What breaks for message-passing potentials

A scatter-heavy MPNN like MACE doesn't have one cutoff and one pair
sum. It has L message-passing layers, each scattering edge features
into per-atom features. After L layers, every atom's representation
depends on the L-hop neighbourhood — even atoms whose owned-rank is
*not* the receiver. Two failure modes:

```{graphviz}
:caption: Halo width grows with depth. Each layer's update at an owned atom reads its 1-hop neighbours, so an owned atom's value at layer ℓ depends on the ℓ-hop neighbourhood. To compute correct values for every owned atom, the shell must reach ℓ × cutoff out from the rank boundary at layer ℓ.
:align: center

digraph halo_growth {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];

    layer1 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="14"><FONT POINT-SIZE="10"><B>after layer 1</B>: shell of width 1·cutoff is enough</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8">0</TD><TD BGCOLOR="#9fc5e8">1</TD><TD BGCOLOR="#9fc5e8">2</TD><TD BGCOLOR="#9fc5e8">3</TD>
          <TD BGCOLOR="#9fc5e8">4</TD><TD BGCOLOR="#9fc5e8">5</TD><TD BGCOLOR="#9fc5e8">6</TD><TD BGCOLOR="#9fc5e8">7</TD>
          <TD BGCOLOR="#cfe2cf" STYLE="dashed">8</TD><TD BGCOLOR="#cfe2cf" STYLE="dashed">9</TD>
        </TR>
        <TR>
          <TD COLSPAN="8"><FONT POINT-SIZE="10">my owned (correct features)</FONT></TD>
          <TD COLSPAN="2"><FONT POINT-SIZE="10">1-hop shell</FONT></TD>
        </TR>
      </TABLE>
    >];

    layer2 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="14"><FONT POINT-SIZE="10"><B>after layer 2</B>: layer-1 features for shell atoms 8..9 must be correct, so shell must extend to atoms 10..11 too</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8">0</TD><TD BGCOLOR="#9fc5e8">1</TD><TD BGCOLOR="#9fc5e8">2</TD><TD BGCOLOR="#9fc5e8">3</TD>
          <TD BGCOLOR="#9fc5e8">4</TD><TD BGCOLOR="#9fc5e8">5</TD><TD BGCOLOR="#9fc5e8">6</TD><TD BGCOLOR="#9fc5e8">7</TD>
          <TD BGCOLOR="#cfe2cf" STYLE="dashed">8</TD><TD BGCOLOR="#cfe2cf" STYLE="dashed">9</TD>
          <TD BGCOLOR="#fff2cc" STYLE="dashed">10</TD><TD BGCOLOR="#fff2cc" STYLE="dashed">11</TD>
        </TR>
        <TR>
          <TD COLSPAN="8"><FONT POINT-SIZE="10">my owned</FONT></TD>
          <TD COLSPAN="2"><FONT POINT-SIZE="10">1-hop shell</FONT></TD>
          <TD COLSPAN="2"><FONT POINT-SIZE="10">2-hop shell</FONT></TD>
        </TR>
      </TABLE>
    >];

    layerL [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="14"><FONT POINT-SIZE="10"><B>after layer L</B>: shell width = L · cutoff (gets expensive fast)</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8">0</TD><TD BGCOLOR="#9fc5e8">1</TD><TD BGCOLOR="#9fc5e8">2</TD><TD BGCOLOR="#9fc5e8">3</TD>
          <TD BGCOLOR="#9fc5e8">4</TD><TD BGCOLOR="#9fc5e8">5</TD><TD BGCOLOR="#9fc5e8">6</TD><TD BGCOLOR="#9fc5e8">7</TD>
          <TD BGCOLOR="#cfe2cf" STYLE="dashed">8</TD><TD BGCOLOR="#cfe2cf" STYLE="dashed">9</TD>
          <TD BGCOLOR="#fff2cc" STYLE="dashed">10</TD><TD BGCOLOR="#fff2cc" STYLE="dashed">11</TD>
          <TD BGCOLOR="#f4cccc" STYLE="dashed">12</TD><TD BGCOLOR="#f4cccc" STYLE="dashed">…</TD>
        </TR>
        <TR>
          <TD COLSPAN="8"><FONT POINT-SIZE="10">my owned</FONT></TD>
          <TD COLSPAN="6"><FONT POINT-SIZE="10">shell grows with L</FONT></TD>
        </TR>
      </TABLE>
    >];

    layer1 -> layer2 [label=<<FONT POINT-SIZE="10">go to next layer</FONT>>];
    layer2 -> layerL [label=<<FONT POINT-SIZE="10">…</FONT>>];
}
```

The framework dodges this growth by *refreshing* the shell between
layers (cheap exchange of one row's worth of data per shell atom)
rather than expanding it. After layer 1 finishes, the shell rows hold
correct *layer-1* features; layer 2's 1-hop reach into them is still
correct.

The two natural extremes that don't work:

| Strategy | What happens |
|---|---|
| **All-gather every layer.** | Comms are O(N · L). Every rank sees the global tensor every layer. Parity with single-process but no scaling. |
| **Strict-local.** Drop edges crossing the rank boundary. | No comms, but every owned atom near the boundary has missing neighbours. Forces on boundary atoms are wrong; energy is wrong; trajectories diverge. |

### 1.3 Beyond MPNNs: the long-range tail

Modern ML potentials don't stop at MPNN. Several patterns make naïve
locality even harder:

| Pattern | Locality breaks because |
|---|---|
| **Charge equilibration / electrostatic embedding (AIMNet2)** | A per-system reduction in every layer. ``mol_sum(per-atom)`` is global. |
| **Reciprocal-space methods (Ewald, PME)** | The structure factor / charge mesh is global. FFT is global. |
| **Attention-based potentials** | All-pairs interactions, full softmax. |
| **Graph-rebuilding models (UMA / eSCN)** | The model constructs its own neighbor list inside ``forward`` — there's no place outside the model to define a halo. |
| **Stress via strain trick** | Differentiates a *replicated* per-graph energy through per-atom positions. |

A halo-only world can't host these without per-pattern surgery. We
need a primitive that lets each model declare *its* locality contract.

### 1.4 Where this lands

The framework supports three storage strategies. Each is a different
answer to the "where does each rank's per-atom tensor live, and what's
the row layout?" question.

```{graphviz}
:caption: Three ways to lay out a per-atom tensor across two ranks. Solid blocks are owned; dotted blocks are remote rows. **Halo storage** materialises a thin shell of remote owners' rows on each rank (read-only mirrors). **Sharded storage** stores only owned; cross-rank reads route over the wire. **Replicated storage** stores the full tensor on every rank and partitions logically — used when a model rebuilds its own NL inside ``forward`` and needs to see every position.
:align: center

digraph storage_modes {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];

    title_halo [label=<<B>HALO STORAGE</B><BR/>(MACE / NequIP / LJ / Ewald / PME)>];
    title_sharded [label=<<B>SHARDED STORAGE</B><BR/>(AIMNet2)>];
    title_replicated [label=<<B>REPLICATED STORAGE</B><BR/>(UMA / eSCN-family)>];

    halo_r0 [label=<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
          <TR>
            <TD BGCOLOR="#9fc5e8" WIDTH="60"> </TD>
            <TD BGCOLOR="#9fc5e8" WIDTH="60">rank 0's<BR/>owned</TD>
            <TD BGCOLOR="#9fc5e8" WIDTH="60"> </TD>
            <TD BGCOLOR="#cfe2cf" WIDTH="60" STYLE="dashed">rank 1's<BR/>shell</TD>
          </TR>
          <TR><TD COLSPAN="4"><FONT POINT-SIZE="10">rank 0's local view: 8 owned + 4 halo rows</FONT></TD></TR>
        </TABLE>
    >];

    halo_r1 [label=<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
          <TR>
            <TD BGCOLOR="#cfe2cf" WIDTH="60" STYLE="dashed">rank 0's<BR/>shell</TD>
            <TD BGCOLOR="#f6b26b" WIDTH="60"> </TD>
            <TD BGCOLOR="#f6b26b" WIDTH="60">rank 1's<BR/>owned</TD>
            <TD BGCOLOR="#f6b26b" WIDTH="60"> </TD>
          </TR>
          <TR><TD COLSPAN="4"><FONT POINT-SIZE="10">rank 1's local view: 4 halo + 8 owned rows</FONT></TD></TR>
        </TABLE>
    >];

    sharded_r0 [label=<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
          <TR>
            <TD BGCOLOR="#9fc5e8" WIDTH="60"> </TD>
            <TD BGCOLOR="#9fc5e8" WIDTH="60">rank 0's<BR/>owned</TD>
            <TD BGCOLOR="#9fc5e8" WIDTH="60"> </TD>
          </TR>
          <TR><TD COLSPAN="3"><FONT POINT-SIZE="10">rank 0's local view: 8 owned rows</FONT></TD></TR>
        </TABLE>
    >];

    sharded_r1 [label=<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
          <TR>
            <TD BGCOLOR="#f6b26b" WIDTH="60"> </TD>
            <TD BGCOLOR="#f6b26b" WIDTH="60">rank 1's<BR/>owned</TD>
            <TD BGCOLOR="#f6b26b" WIDTH="60"> </TD>
          </TR>
          <TR><TD COLSPAN="3"><FONT POINT-SIZE="10">rank 1's local view: 8 owned rows</FONT></TD></TR>
        </TABLE>
    >];

    repl_r0 [label=<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
          <TR>
            <TD BGCOLOR="#9fc5e8" WIDTH="60">rank 0's<BR/>partition</TD>
            <TD BGCOLOR="#9fc5e8" WIDTH="60"> </TD>
            <TD BGCOLOR="#cfd8dc" WIDTH="60" STYLE="dashed">rank 1's<BR/>partition</TD>
            <TD BGCOLOR="#cfd8dc" WIDTH="60" STYLE="dashed"> </TD>
          </TR>
          <TR><TD COLSPAN="4"><FONT POINT-SIZE="10">rank 0's local view: full 16 rows; node_partition = [0..7]</FONT></TD></TR>
        </TABLE>
    >];

    repl_r1 [label=<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
          <TR>
            <TD BGCOLOR="#cfd8dc" WIDTH="60" STYLE="dashed">rank 0's<BR/>partition</TD>
            <TD BGCOLOR="#cfd8dc" WIDTH="60" STYLE="dashed"> </TD>
            <TD BGCOLOR="#f6b26b" WIDTH="60">rank 1's<BR/>partition</TD>
            <TD BGCOLOR="#f6b26b" WIDTH="60"> </TD>
          </TR>
          <TR><TD COLSPAN="4"><FONT POINT-SIZE="10">rank 1's local view: full 16 rows; node_partition = [8..15]</FONT></TD></TR>
        </TABLE>
    >];

    title_halo -> halo_r0 [style=invis];
    halo_r0 -> halo_r1 [label=<<FONT POINT-SIZE="10">refresh shell from<BR/>each other's owners</FONT>> dir=both color="#666"];
    title_sharded -> sharded_r0 [style=invis];
    sharded_r0 -> sharded_r1 [label=<<FONT POINT-SIZE="10">cross-rank read or scatter<BR/>routes over the wire<BR/>(no shell stored)</FONT>> dir=both color="#666" style=dashed];
    title_replicated -> repl_r0 [style=invis];
    repl_r0 -> repl_r1 [label=<<FONT POINT-SIZE="10">per-MP-layer all_gather<BR/>of partition rows<BR/>(every rank sees the<BR/>full feature tensor)</FONT>> dir=both color="#666"];
}
```

Quick reference for picking a strategy:

| Strategy | Per-atom tensor row layout | When the model needs it | Per-step comm |
|---|---|---|---|
| **Halo** | ``[owned │ halo]`` — owned rows are unique, halo rows mirror remote owners. | Local-receptive-field MPNN where the cutoff fits in one halo width. The model can be handed an opaque ``(n_owned + n_halo, *F)`` view and produce correct outputs without knowing the partition. | One halo exchange per step (refresh shell rows). |
| **Sharded** | ``(n_owned, *F)`` — each rank stores only its owned rows. | Per-system reductions inside every layer (charge equilibration; ``mol_sum``). The cost of a per-layer halo refresh is dominated by the global reduction anyway, so saving on memory wins. | One ``all_to_all_v`` per cross-rank read or scatter (already needed for ``mol_sum``). |
| **Replicated** | Full ``(n_global, *F)`` on every rank; partition is logical (a ``node_partition`` index slice). | Models that build their own neighbor list inside ``forward`` and can't be handed a pre-padded view (``UMA``'s ``_generate_graph``, ``eSCN``-family). Memory is O(n_global) per rank, so capped by single-GPU budget. | Per-MP-layer feature ``all_gather`` (autograd-aware) plus per-system ``all_reduce`` for energy / forces / stress. |

The storage strategy is just the start. Within each strategy we still
need to pick scatter rules, gather rules, and per-output reductions.
Encoding those choices is the job of the *spec* (§3).

---

## 2. ShardTensor: a partition-aware Tensor subclass

### 2.1 The primitive

`ShardTensor` is a `torch.Tensor` subclass that carries metadata about
the partition, plus a registry of dispatch handlers that intercept
specific torch ops on it via `__torch_function__`. The same model code
runs single-process and distributed; the runtime decides what to do
based on the input's metadata.

```{graphviz}
:caption: A ShardTensor is a regular tensor block with a small bag of metadata attached. The data lives in the same buffer as a plain tensor (zero-copy). The metadata describes what the rows mean and which rules govern operations on them.
:align: center

digraph shardtensor_anatomy {
    rankdir=LR;
    node [shape=plaintext fontname="Helvetica"];

    tensor [label=<
      <TABLE BORDER="0" CELLBORDER="2" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="4"><B>tensor data</B> &nbsp;&nbsp; <FONT POINT-SIZE="10">(this rank's view, shape (12, 3))</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="65"> </TD>
          <TD BGCOLOR="#9fc5e8" WIDTH="65">owned</TD>
          <TD BGCOLOR="#9fc5e8" WIDTH="65"> </TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="80" STYLE="dashed">shell</TD>
        </TR>
        <TR><TD COLSPAN="4"><FONT POINT-SIZE="10">rows 0..7 are this rank's; rows 8..11 mirror neighbour rank's owned</FONT></TD></TR>
      </TABLE>
    >];

    metadata [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2" BGCOLOR="#fff2cc"><B>metadata bag</B></TD></TR>
        <TR><TD>how many rows are mine</TD><TD>8</TD></TR>
        <TR><TD>how many rows total (mine + shell)</TD><TD>12</TD></TR>
        <TR><TD>which rank am I</TD><TD>0</TD></TR>
        <TR><TD>where do shell rows come from</TD><TD>rows 8..11<BR/>↤ rank 1's owned</TD></TR>
        <TR><TD>how many systems<BR/>(for per-graph reductions)</TD><TD>1</TD></TR>
        <TR><TD VALIGN="TOP">partition rules<BR/>(see §3)</TD><TD ALIGN="LEFT">storage = halo<BR/>scatter rule = halo_correction<BR/>gather rule = halo_read<BR/>per-system reductions = on</TD></TR>
      </TABLE>
    >];

    tensor -> metadata [style=dashed label=<<FONT POINT-SIZE="10">carries...</FONT>>];
}
```

```{code-block} python
:caption: Construction. ``wrap`` is the canonical entry point — the framework calls it when promoting padded positions, and users call it for their own per-atom fields.
from nvalchemi.distributed.ops import ShardTensor

t = ShardTensor.wrap(
    halo_padded_positions,        # (n_padded, 3) plain tensor
    spec=SPEC_MPNN_HALO,           # tells handlers HOW to dispatch
    meta=halo_meta,                # owned/padded counts, routing
    config=halo_config,            # mesh + process group
    n_systems=1,                   # for per-system reductions
)
```

### 2.2 Dispatch via __torch_function__

When a torch op is called with at least one ShardTensor argument,
PyTorch invokes `ShardTensor.__torch_function__`. We walk a small
registry of `(op, predicate, handler)` tuples; the first matching
predicate wins. No match → `super().__torch_function__` falls back
to plain torch behaviour with metadata propagation.

```{graphviz}
:caption: Dispatch flowchart for a single torch op. Predicates inspect the inputs' shapes + spec; the matching handler runs cross-rank logic and re-promotes outputs.
:align: center

digraph dispatch {
    rankdir=TB; node [shape=box style="rounded,filled" fontname="Helvetica"];

    Op [label="op(shard_tensor, ...)" fillcolor="#dce6f1"];
    Pred [label="any registered handler\npredicate matches?"
          fillcolor="#fff2cc" shape=diamond];
    Handler [label="handler runs:\n• unwrap inputs\n• run cross-rank logic\n• promote outputs"
             fillcolor="#cfe2cf"];
    Fallback [label="super().__torch_function__\n(plain torch.Tensor path)"
              fillcolor="#f3f3f3"];
    Done [label="return result\n(metadata propagated)" fillcolor="#82b366"];

    Op -> Pred;
    Pred -> Handler [label="yes"];
    Pred -> Fallback [label="no"];
    Handler -> Done;
    Fallback -> Done;
}
```

```{code-block} python
:caption: A user-facing example. The wrapper code is identical to single-process; the dispatch machinery reads the partition off the tensor.
# Single-process or distributed — same line:
total_energy = torch.zeros(n_graphs, ...)
total_energy = total_energy.scatter_add_(0, batch_idx, atomic_energies)
#                                    ^
# If atomic_energies is a ShardTensor with system_reductions=True
# and accumulator shape == n_systems, the dispatch routes through
# ``per_system_reduce``: local scatter + cross-rank all_reduce.
# Otherwise: plain in-place scatter_add_.
```

### 2.3 Why "almost transparent"

There's a footgun the framework has to surface explicitly. PyTorch's
in-place ops (`t.scatter_add_(...)`) return `self`; the standard
idiom drops the return. Under cross-rank halo correction the handler
*can't* preserve in-place semantics — it returns a fresh tensor.
Wrapper authors must rebind the return:

```{code-block} python
:caption: The one rule wrapper authors need to internalise. Single-process: rebind is a no-op. Distributed: mandatory.
# WRONG (single-process: works; distributed: silently produces zeros)
G.scatter_add_(0, receivers, edge_feats)

# RIGHT
G = G.scatter_add_(0, receivers, edge_feats)
```

The validator's worker-error translator detects this exact pattern and
surfaces it as a one-line fix. See {ref}`distributed_byo_guide` for
the full catalogue of dispatch handlers and their predicates.

---

## 3. Specs: declaring what to do at each op

ShardTensor knows *how* the data is partitioned. The
{py:class}`~nvalchemi.distributed.spec.MLIPSpec` tells it *what to do*
at each op. The split is deliberate: ShardTensor stays
chemistry-free; specs encode model-specific reduction rules.

The spec is a small structure with three knobs. Each knob has a
visual interpretation on the tensor — that's what the rest of this
section walks through.

| Knob | Choices | What gets visualised |
|---|---|---|
| **scatter rule** | `halo_correction` / `local` / `distributed` | how an `t.scatter_add_(...)` call moves data |
| **gather rule** | `halo_read` / `local` / `distributed` | how an `index_select` call sees data |
| **per-system reductions** | `on` / `off` | how a per-graph energy `scatter_add` becomes a global sum |

Plus per-op transforms (§6) and per-output classifications (§7).

### 3.1 The scatter rule: where do partial messages go?

A scatter is "for each edge, write a contribution into the receiver's
row." When the receiver might be a halo row (a mirrored copy of a
remote rank's owned), the scatter rule decides whether (and how) to
account for that.

```{graphviz}
:caption: ``scatter = "halo_correction"`` — the canonical MPNN pattern. Each rank scatters its messages into both owned and shell rows. The shell partials get sent back to their owners and accumulated. Then owners' values are pushed back out to refresh shell copies for the next layer.
:align: center

digraph scatter_halo {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];
    edge [color="#666"];

    s0 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10">step 1: each rank does a local scatter — partials land in BOTH owned and shell rows</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="180">rank 0 owned<BR/><FONT POINT-SIZE="9">filled with this rank's edges</FONT></TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="120" STYLE="dashed">rank 1's shell on rank 0<BR/><FONT POINT-SIZE="9">partials destined for rank 1</FONT></TD>
        </TR>
      </TABLE>
    >];

    s1 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10">step 2: send shell partials back to owners; owners accumulate</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="180">rank 0 owned<BR/><FONT POINT-SIZE="9">unchanged</FONT></TD>
          <TD BGCOLOR="#fce5cd" WIDTH="120">rank 1 owned receives<BR/>contributions from rank 0</TD>
        </TR>
      </TABLE>
    >];

    s2 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10">step 3: owners broadcast their final values back into the shell so the next layer sees them</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="180">rank 0 owned</TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="120" STYLE="dashed">shell now mirrors<BR/>rank 1's final owned</TD>
        </TR>
      </TABLE>
    >];

    s0 -> s1 [label=<<FONT POINT-SIZE="10">send back<BR/>(reverse)</FONT>>];
    s1 -> s2 [label=<<FONT POINT-SIZE="10">refresh<BR/>(forward)</FONT>>];
}
```

```{graphviz}
:caption: ``scatter = "local"`` — pure per-rank scatter, no cross-rank exchange. Used when the accumulator is per-system (small) and the per-system all-reduce in step 2 of the next rule handles cross-rank correctness; or for a halo-unaware backbone whose edges already cover the global graph.
:align: center

digraph scatter_local {
    rankdir=LR;
    node [shape=plaintext fontname="Helvetica"];

    a0 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD BGCOLOR="#9fc5e8" WIDTH="120">rank 0 owned</TD><TD BGCOLOR="#cfe2cf" STYLE="dashed">shell</TD></TR>
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10">scatter_add_ writes locally; nothing crosses ranks</FONT></TD></TR>
      </TABLE>
    >];

    a1 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD BGCOLOR="#cfe2cf" STYLE="dashed">shell</TD><TD BGCOLOR="#f6b26b" WIDTH="120">rank 1 owned</TD></TR>
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10">same: pure local</FONT></TD></TR>
      </TABLE>
    >];

    a0 -> a1 [style=invis];
}
```

### 3.2 The gather rule: how does an index_select see data?

A gather is "for each input row index, fetch that row's data." When
the index falls in the shell region (i.e. asks for a remote rank's
row), the gather rule decides whether to serve from the local
mirror or to route a request to the owner.

```{graphviz}
:caption: ``gather = "halo_read"`` — the index ``9`` (in this rank's shell) is served from the local shell copy. No cross-rank traffic at gather time. Stays cheap because the shell is refreshed by the previous scatter's step 3.
:align: center

digraph gather_halo {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];

    tensor [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="40">0</TD>
          <TD BGCOLOR="#9fc5e8" WIDTH="40">…</TD>
          <TD BGCOLOR="#9fc5e8" WIDTH="40">7</TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="40" STYLE="dashed">8</TD>
          <TD BGCOLOR="#fff2cc" WIDTH="40"><B>9</B></TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="40" STYLE="dashed">10</TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="40" STYLE="dashed">11</TD>
        </TR>
        <TR><TD COLSPAN="7"><FONT POINT-SIZE="10">rank 0's view: indices 0..7 are owned, 8..11 are shell</FONT></TD></TR>
      </TABLE>
    >];

    request [label=<<FONT POINT-SIZE="11"><B>request: index_select [9]</B></FONT>>];
    served [label=<<FONT POINT-SIZE="10">served from local shell row 9<BR/>(no cross-rank message)</FONT>>];

    request -> tensor [label=<<FONT POINT-SIZE="10">read</FONT>>];
    tensor -> served [label=<<FONT POINT-SIZE="10">return</FONT>>];
}
```

### 3.3 Per-system reductions: per-rank scatter + cross-rank sum

The most common reduction in MLIPs is `total_energy.scatter_add_(0,
batch_idx, atomic_energies)` — collapsing per-atom energies into a
per-graph total. Under partitioning, no rank has all the atoms, so
the local scatter is a partial. `per_system_reduce` does the local
scatter, then sums the partials across ranks.

```{graphviz}
:caption: ``per_system_reduce`` — one primitive that combines a local per-system scatter with a cross-rank sum. The output is replicated globally on every rank, so any rank can read the final per-graph value.
:align: center

digraph per_system {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];

    inputs [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10">per-rank atomic energies (sliced to owned only)</FONT></TD></TR>
        <TR>
          <TD>rank 0:</TD>
          <TD BGCOLOR="#9fc5e8">e₀ e₁ e₂ e₃ e₄ e₅ e₆ e₇</TD>
        </TR>
        <TR>
          <TD>rank 1:</TD>
          <TD BGCOLOR="#f6b26b">e₈ e₉ e₁₀ e₁₁ e₁₂ e₁₃ e₁₄ e₁₅</TD>
        </TR>
      </TABLE>
    >];

    locals [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10">step 1: local scatter into per-graph slot</FONT></TD></TR>
        <TR><TD>rank 0:</TD><TD BGCOLOR="#9fc5e8"><B>Σ₀..₇</B></TD></TR>
        <TR><TD>rank 1:</TD><TD BGCOLOR="#f6b26b"><B>Σ₈..₁₅</B></TD></TR>
      </TABLE>
    >];

    global [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10">step 2: all_reduce sum — every rank holds the global total</FONT></TD></TR>
        <TR><TD>rank 0:</TD><TD BGCOLOR="#82b366" COLOR="white"><B>E_global</B></TD></TR>
        <TR><TD>rank 1:</TD><TD BGCOLOR="#82b366" COLOR="white"><B>E_global</B></TD></TR>
      </TABLE>
    >];

    inputs -> locals [label=<<FONT POINT-SIZE="10">scatter_add_ on owned slice</FONT>>];
    locals -> global [label=<<FONT POINT-SIZE="10">all_reduce(SUM)</FONT>>];
}
```

### 3.4 The complete spec

The decisions above all live on a small data structure that the
wrapper attaches via `distribution_spec`:

| Field | Purpose |
|---|---|
| `distribution.policy` | Storage layout: halo / sharded / local. Each carries its own scatter and gather rules. |
| `distribution.custom_ops` | Per-op declarations for opaque kernels that bypass `__torch_function__` (Warp, Triton). See §6. |
| `output_kinds` | One of `PER_NODE`, `PER_GRAPH`, `GLOBAL`, `UNKNOWN` per output. Drives final consolidation. |
| `owned_only_outputs` | Per-atom outputs that are already globally correct on each rank (e.g. PME reciprocal forces) — skip the back-exchange. |
| `all_reduce_outputs` | Per-rank-partial outputs that need a final SUM across ranks (e.g. strain-trick stress). |

```{code-block} python
:caption: The shipped presets cover the production model families. A wrapper author either picks one or composes a new ``MLIPSpec`` directly.
from nvalchemi.distributed.spec import (
    SPEC_MPNN_HALO,        # MACE, NequIP, Allegro, ORB (spatial halo)
    SPEC_MPNN_GP,          # MPNN node-partition graph-parallel
    SPEC_UMA_HALO,         # UMA / eSCN (spatial halo)
    SPEC_LJ_HALO,          # Lennard-Jones (Warp pair kernel)
    SPEC_EWALD_HALO,       # Ewald (real + reciprocal stages)
    SPEC_PME_HALO,         # PME (charge spread + FFT mesh)
    SPEC_DFTD3_HALO,       # DFT-D3 dispersion
)
```

```{code-block} python
:caption: The shipped presets cover the production model families.
from nvalchemi.distributed.spec import (
    SPEC_MPNN_HALO,        # MACE, NequIP, Allegro, ORB (spatial halo)
    SPEC_MPNN_GP,          # MPNN node-partition graph-parallel
    SPEC_UMA_HALO,         # UMA / eSCN (spatial halo)
    SPEC_LJ_HALO,          # Lennard-Jones (Warp pair kernel)
    SPEC_EWALD_HALO,       # Ewald (real + reciprocal stages)
    SPEC_PME_HALO,         # PME (charge spread + FFT mesh)
    SPEC_DFTD3_HALO,       # DFT-D3 dispersion
)
```

Authoring a spec for a new model is the topic of §6 + §7. For now,
note that **every spec parameterises the same dispatch machinery** —
the registry, the predicates, the handlers. The spec is the single
declaration point.

---

## 4. MACE end-to-end: what halo MPNN looks like

### 4.1 The forward pass, three steps

```{graphviz}
:caption: One MACE message-passing layer under halo storage, viewed as tensor states (rank 0 of 2). The features tensor enters the layer as ``(n_padded, F)`` with both owned and shell rows populated. Edge messages scatter into the receivers' rows, leaving partial accumulations in shell positions destined for rank 1. The framework's scatter rule sends those partials back, then refreshes the shell so the next layer reads correct values.
:align: center

digraph mace_layer {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];

    fpre [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>features entering layer ℓ</B>: shape (n_padded, F)</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="180">my owned (8 rows of features)</TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="120" STYLE="dashed">shell (4 rows mirroring rank 1)</TD>
        </TR>
      </TABLE>
    >];

    scatter [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>compute messages, scatter into receivers</B>: writes land in BOTH owned and shell rows</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="180">owned: full sum of incoming<BR/>edges where receiver is mine</TD>
          <TD BGCOLOR="#fff2cc" WIDTH="120">shell: partial sum of edges<BR/>where receiver lives on rank 1</TD>
        </TR>
      </TABLE>
    >];

    after_back [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>shell partials sent back to owners; rank 1's owners now hold the cross-rank contributions too</B></FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="180">owned: unchanged</TD>
          <TD BGCOLOR="#fce5cd" WIDTH="120">shell: emptied<BR/>(rank 1 has the data now)</TD>
        </TR>
      </TABLE>
    >];

    fpost [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>features leaving layer ℓ</B>: shape (n_padded, F), shell refreshed for layer ℓ+1's gather</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="180">owned: layer-(ℓ+1) features</TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="120" STYLE="dashed">shell: rank 1's<BR/>layer-(ℓ+1) features</TD>
        </TR>
      </TABLE>
    >];

    fpre -> scatter [label=<<FONT POINT-SIZE="10">edge messages</FONT>>];
    scatter -> after_back [label=<<FONT POINT-SIZE="10">send back<BR/>(reverse exchange)</FONT>>];
    after_back -> fpost [label=<<FONT POINT-SIZE="10">refresh shell<BR/>(forward exchange)</FONT>>];
}
```

```{code-block} python
:caption: What the wrapper actually writes. The halo-correction is implicit in ShardTensor dispatch — the wrapper has zero distribution code.
# Inside MACE InteractionBlock — a typical scatter pattern:
node_feats = node_feats.zero_()
node_feats = node_feats.scatter_add_(0, receivers, edge_messages)
#       ^                       ^
#       rebind handles          if edge_messages is a ShardTensor,
#       distributed return       the dispatch handler does
#                                halo_reverse + halo_forward
```

### 4.2 The full forward (one slide)

```{graphviz}
:caption: A complete forward pass viewed as tensor states (rank 0 of 2). Positions enter as a halo-padded ShardTensor; L message-passing layers each apply the §4.1 pattern; the final atomic energies get sliced and reduced into a globally replicated total energy. Forces fall out of an autograd backward through positions; the framework routes shell gradients back to their owners.
:align: center

digraph mace_full {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];

    pos [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>positions</B>: ShardTensor (n_padded, 3)</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="180">my owned (8 atoms)</TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="120" STYLE="dashed">shell (4 atoms, mirrored)</TD>
        </TR>
      </TABLE>
    >];

    feats [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>features after L MP layers</B>: ShardTensor (n_padded, F)</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="180">my owned (correct)</TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="120" STYLE="dashed">shell (mirrored, layer-L)</TD>
        </TR>
      </TABLE>
    >];

    atE [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>atomic energies</B>: head(features) → (n_padded,)</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="180">my owned: 8 per-atom energies</TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="120" STYLE="dashed">shell: 4 mirrors (will be dropped)</TD>
        </TR>
      </TABLE>
    >];

    pers [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>per-system reduce</B>: drop shell, scatter into per-graph slot, all_reduce</FONT></TD></TR>
        <TR><TD>local sum on rank 0:</TD><TD BGCOLOR="#9fc5e8"><B>Σ owned</B></TD></TR>
        <TR><TD>after all_reduce:</TD><TD BGCOLOR="#82b366"><FONT COLOR="white"><B>E_global (replicated on every rank)</B></FONT></TD></TR>
      </TABLE>
    >];

    bwd [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>autograd.grad(E_global, positions)</B>: produces ∂E/∂x on (n_padded, 3)</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="180">owned: this rank's contribution</TD>
          <TD BGCOLOR="#fff2cc" WIDTH="120">shell: this rank's<BR/>partial gradient at others' atoms</TD>
        </TR>
      </TABLE>
    >];

    forces [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>forces after consolidation</B>: shell gradients sent back to owners</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#82b366" WIDTH="180"><FONT COLOR="white">my owned forces<BR/>(complete: own + cross-rank pieces)</FONT></TD>
        </TR>
      </TABLE>
    >];

    pos -> feats [label=<<FONT POINT-SIZE="10">L message-passing layers<BR/>(each: scatter + send-back + refresh)</FONT>>];
    feats -> atE [label=<<FONT POINT-SIZE="10">read-out head</FONT>>];
    atE -> pers [label=<<FONT POINT-SIZE="10">scatter_add into (n_graphs,)<BR/>(per-system reduce)</FONT>>];
    pers -> bwd [label=<<FONT POINT-SIZE="10">.backward()</FONT>>];
    bwd -> forces [label=<<FONT POINT-SIZE="10">framework routes<BR/>shell gradients to owners</FONT>>];
}
```

### 4.3 Why this is short to write

```{code-block} python
:caption: A halo-MPNN wrapper has zero distribution-aware code. The framework promotes per-atom inputs to ShardTensor before calling the wrapper; ``__torch_function__`` propagates the partition through the wrapper's ops; consolidation handles the final per-output reduction.
class MACEWrapper(nn.Module, BaseModelMixin):

    @property
    def distribution_spec(self):
        return SPEC_MPNN_HALO   # halo correction + halo read + per-system reductions

    def adapt_input(self, data, **kwargs):
        # Drop neighbour-list sentinel rows. Single-process: drops the
        # genuine padding rows the NL builder emits. Distributed: also
        # drops halo-receiver rows the framework rewrote to the same
        # sentinel value at NL-build time. One line, both regimes.
        n_atoms = data.positions.shape[0]
        edge_index = data.neighbor_list.long().T
        valid = (edge_index[0] < n_atoms) & (edge_index[1] < n_atoms)
        return {
            "positions": data.positions,                # already a ShardTensor under DD
            "edge_index": edge_index[:, valid],
            "node_attrs": self._node_attrs(data),
            "shifts": ...,
        }

    def forward(self, data):
        return self.model(**self.adapt_input(data))
```

The framework handles every cross-rank thing: halo build, NL filter,
per-layer scatter/refresh, per-system reduction, force consolidation,
strain-trick stress (with the inner virial pass routed correctly).

---

## 4b. UMA end-to-end: replicated for graph-rebuilding models

Some models can't be handed a halo-padded view because they build
their own neighbor list inside ``forward``. UMA / eSCN-family models
take ``positions`` and emit ``edge_index`` via their internal
``radius_pbc`` kernel — there's no pre-forward seam to attach a halo
to. The replicated strategy answers that: every rank holds the full
positions tensor, the model's NL builder runs on the global view,
then a logical *node partition* (an index slice of
``arange(n_global)``) restricts which receivers each rank computes
messages for.

UMA was designed by fairchem with this pattern already wired in. Its
``escn_md._generate_graph`` already gates on
``gp_utils.initialized()``: under that branch, fairchem partitions
the receiver set, filters edges to ``target ∈ partition``, and slices
node features to the partition rows. ``escn_md_block`` already calls
``gather_from_model_parallel_region_sum_grad`` per MP layer to read
sender features for cross-partition edges. ``compute_forces_and_stress``
already calls ``reduce_from_model_parallel_region`` on the gradient
of the per-rank energy partial.

We don't reimplement that partition path — we redirect it. The
distributed spec installs a small set of ``PythonAdapter`` swaps on
``fairchem.core.common.gp_utils``:

```{code-block} python
:caption: SPEC_UMA_REPLICATED's third_party_helpers (built lazily inside UMAWrapper.distribution_spec). Each adapter swaps a fairchem accessor with a chemistry-side replacement that reads our mesh state from a per-thread ReplicatedMetadata.
helpers = (
    PythonAdapter("fairchem.core.common.gp_utils", "initialized",
                  patched_initialized),
    PythonAdapter("fairchem.core.common.gp_utils", "get_gp_group",
                  patched_get_gp_group),
    PythonAdapter("fairchem.core.common.gp_utils", "get_gp_rank",
                  patched_get_gp_rank),
    PythonAdapter("fairchem.core.common.gp_utils", "get_gp_world_size",
                  patched_get_gp_world_size),
    PythonAdapter("fairchem.core.common.gp_utils",
                  "reduce_from_model_parallel_region",
                  patched_reduce_from_model_parallel_region),
    PythonAdapter("fairchem.core.common.gp_utils",
                  "gather_from_model_parallel_region",
                  patched_gather_from_model_parallel_region),
    PythonAdapter("fairchem.core.common.gp_utils",
                  "gather_from_model_parallel_region_sum_grad",
                  patched_gather_from_model_parallel_region_sum_grad),
    # outputs.reduce_node_to_system patched at two binding sites
    # (escn_md re-imports the symbol at module-load time).
    PythonAdapter("fairchem.core.models.uma.outputs",
                  "reduce_node_to_system",
                  patched_reduce_node_to_system),
    PythonAdapter("fairchem.core.models.uma.escn_md",
                  "reduce_node_to_system",
                  patched_reduce_node_to_system),
)
```

The patched accessors all read a thread-local ``ReplicatedMetadata``
that ``DistributedModel._call_replicated_storage`` sets up at the
start of each forward and clears at the end. Outside a scoped call
the thread-local is ``None`` and the replacements behave as
fairchem's no-GP path (``initialized() == False``), so single-rank
inference is unaffected.

The patched reductions promote fp32 → fp64 inside ``all_reduce``
(matches the existing ``per_system_reduce`` pattern) so non-deterministic
NCCL fp32 SUM commit order doesn't show up at the ε_fp32 level. Per-element
error drops from ~1e-7 to ~1e-15 at modest cost.

One extra subtlety lives in ``patched_reduce_node_to_system``.
fairchem's original always all-reduces the per-system tensor when
``gp_utils.initialized()``. That's correct when ``node_values`` is a
per-rank partial (the ``compute_energy`` path), but wrong when it's
already global+replicated (the ``compute_forces_and_stress``
``pos_virial`` path, where ``node_values = grads[0] * pos`` and
``grads[0]`` was all-reduced two lines earlier). The unconditional
all_reduce there would multiply the global value by ``world_size``.
The replacement gates the second all_reduce on
``len(batch) == n_global``: if the input covers all atoms, it's
already aggregated globally — skip the reduce. If it covers only the
partition (``n_owned``), it's partial — run the reduce.

```{code-block} python
:caption: UMAWrapper has zero distribution-aware code. distribution_spec returns the replicated preset; everything else flows through the patched gp_utils path on the model side.
class UMAWrapper(nn.Module, BaseModelMixin):

    @property
    def distribution_spec(self):
        # Triggers @triton_op registration + builds the third_party_helpers
        # tuple that swaps fairchem's gp_utils accessors. Memoised by the
        # framework's adapter registry across forwards within a scope.
        return SPEC_UMA_REPLICATED   # populated lazily, see helpers above

    def forward(self, data):
        return self.predict_unit(data)   # fairchem does the rest
```

What the framework adds on top:

* ``ShardedBatch.from_batch(partition_mode="replicated")`` broadcasts
  the full atom data to every rank and stamps each rank's
  ``node_partition = arange(n_global).tensor_split(W)[rank]``.
* ``DistributedModel._call_replicated_storage`` constructs the
  per-thread ``ReplicatedMetadata``, installs it before the wrapper
  call, and consolidates outputs (which mostly come back already
  globally reduced inside the model — no extra all_reduce needed at
  consolidation time).

The replicated path also produces a clean memory-savings ratio under
multi-rank: even though every rank holds the full positions tensor,
per-rank MP-layer activations only span ``n_owned`` rows, so peak
memory under 2 ranks is consistently 0.55–0.90× single-rank memory
(better at larger N, where the activations dominate the peak over
the replicated positions). The compute speedup is more modest
(~1.20× forward, ~1.55× NVT at 2 ranks) because the per-MP-layer
``all_gather`` is in the critical path.

---

## 5. Composing models: pipelines that mix strategies

Real workflows compose models. Energy = MACE (short-range MPNN) +
Ewald (long-range electrostatics). Different sub-models can want
different storage strategies and different specs.

```{graphviz}
:caption: A two-block pipeline: MACE (short-range MPNN) + Ewald (long-range electrostatics). Halo construction happens once on the input; both blocks read the same padded tensor. Each block produces a globally replicated total energy and per-rank-owned forces; the pipeline sums them.
:align: center

digraph pipeline {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];

    input [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>input</B>: per-rank owned atoms (8 each, ShardedBatch)</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="120">rank 0 owned</TD>
          <TD BGCOLOR="#f6b26b" WIDTH="120">rank 1 owned</TD>
        </TR>
      </TABLE>
    >];

    padded [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>padded batch</B>: built once, reused by both blocks</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="120">rank 0: owned + shell</TD>
          <TD BGCOLOR="#f6b26b" WIDTH="120">rank 1: shell + owned</TD>
        </TR>
      </TABLE>
    >];

    mace [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2" BGCOLOR="#dce6f1"><B>MACE block</B>: short-range MPNN<BR/><FONT POINT-SIZE="10">spec = SPEC_MPNN_HALO</FONT></TD></TR>
        <TR><TD>energy:</TD><TD BGCOLOR="#82b366"><FONT COLOR="white">E_MACE (replicated)</FONT></TD></TR>
        <TR><TD>forces:</TD><TD BGCOLOR="#9fc5e8">F_MACE (per-rank owned)</TD></TR>
      </TABLE>
    >];

    ewald [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2" BGCOLOR="#fff2cc"><B>Ewald block</B>: long-range electrostatics<BR/><FONT POINT-SIZE="10">spec = SPEC_EWALD_HALO + OpAdapter for partial S(k)</FONT></TD></TR>
        <TR><TD>energy:</TD><TD BGCOLOR="#82b366"><FONT COLOR="white">E_Ewald (replicated)</FONT></TD></TR>
        <TR><TD>forces:</TD><TD BGCOLOR="#9fc5e8">F_Ewald (per-rank owned)</TD></TR>
      </TABLE>
    >];

    out [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#82b366"><FONT COLOR="white"><B>energy</B> = E_MACE + E_Ewald (replicated)<BR/><B>forces</B> = F_MACE + F_Ewald (per-rank owned)</FONT></TD></TR>
      </TABLE>
    >];

    input -> padded [label=<<FONT POINT-SIZE="10">build halo (once)</FONT>>];
    padded -> mace;
    padded -> ewald;
    mace -> out;
    ewald -> out;
}
```

```{code-block} python
:caption: Pipeline construction is one line per block plus a top-level wrap.
from nvalchemi.models import PipelineModelWrapper
from nvalchemi.distributed import DistributedPipelineModel

pipeline = PipelineModelWrapper([
    MACEWrapper.from_checkpoint("medium-0b2"),
    EwaldModelWrapper(cutoff=10.0),
])

dist_model = DistributedPipelineModel(pipeline, domain_config)
energy_dict = dist_model(sharded_batch)
# energy_dict["energy"]  ← MACE + Ewald summed, globally replicated
# energy_dict["forces"]  ← per-rank owned, autograd-derived
```

Composition rules at the seam:

| Sub-model A | Sub-model B | Pipeline strategy |
|---|---|---|
| Halo | Halo | Halo (single padded_batch, both blocks read it) |
| Halo | Sharded | Sharded (most permissive) |
| Sharded | Sharded | Sharded |
| Local | anything | the other one |

The merge rule is implemented in `MLIPSpec._merge_strategies` — same
discriminated-union pattern as the Strategy classes themselves.

---

## 6. Wrapping Warp / Triton kernels

### 6.1 The boundary problem

`ShardTensor.__torch_function__` only fires on ops PyTorch dispatches
through the public Python API. Warp / Triton kernels reach into
tensor data via `wp.from_torch(t)` / Triton's pointer protocol —
both strip the subclass before reading. The kernel sees a plain
buffer and writes a plain buffer; ShardTensor never gets a chance to
intervene.

```{graphviz}
:caption: An OpAdapter wraps the kernel boundary. Inputs enter as ShardTensors; the adapter pre-shapes them per the wrapper's declared transforms (e.g. slice to owned only); the kernel runs on plain tensors; outputs get post-shaped (e.g. shell-rows-back-to-owners) and re-promoted to ShardTensor for the rest of the model.
:align: center

digraph kernel_boundary {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];

    inp [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>input</B>: ShardTensor (n_padded, 3)</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="120">owned</TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="80" STYLE="dashed">shell</TD>
        </TR>
      </TABLE>
    >];

    pre [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>pre-shape</B>: e.g. slice to owned-only<BR/>(arg_transforms = SliceOwned)</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="120">owned (plain tensor)</TD>
        </TR>
      </TABLE>
    >];

    kern [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#f4cccc"><B>kernel runs</B><BR/><FONT POINT-SIZE="10">Warp / Triton / custom_op<BR/>plain tensors only — ShardTensor is invisible inside</FONT></TD></TR>
      </TABLE>
    >];

    post [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>post-shape</B>: e.g. send shell partials to owners,<BR/>refresh shell from owners<BR/>(output_transforms = ScatterOutputs)</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="120">owned (corrected)</TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="80" STYLE="dashed">shell (refreshed)</TD>
        </TR>
      </TABLE>
    >];

    out [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2"><FONT POINT-SIZE="10"><B>output</B>: ShardTensor again, ready for the next op</FONT></TD></TR>
        <TR>
          <TD BGCOLOR="#9fc5e8" WIDTH="120">owned</TD>
          <TD BGCOLOR="#cfe2cf" WIDTH="80" STYLE="dashed">shell</TD>
        </TR>
      </TABLE>
    >];

    inp -> pre [label=<<FONT POINT-SIZE="10">unwrap + transform</FONT>>];
    pre -> kern [label=<<FONT POINT-SIZE="10">launch</FONT>>];
    kern -> post [label=<<FONT POINT-SIZE="10">cross-rank correction</FONT>>];
    post -> out [label=<<FONT POINT-SIZE="10">re-promote to ShardTensor</FONT>>];
}
```

### 6.2 The transform vocabulary

Every input / output transform is a small dataclass marker. The
framework's `wrap_custom_op` interprets them at call time.

| Position | Transform | What it does |
|---|---|---|
| input | `GatherInputs` | halo-pad an owned-shape input to `(n_padded, *F)` |
| input | `GatherInputsFull` | sharded analogue: full-gather to `(n_global+1, *F)` |
| input | `SliceOwned` | slice halo-padded input to `(n_owned, *F)` |
| output | `ScatterOutputs` | halo_reverse + halo_forward on a per-atom output |
| output | `AllReduceSum` | cross-rank SUM (autograd-symmetric) |
| output | `SliceOutputsOwned` | slice global-shape output back to owned-only |

### 6.3 Worked example

```{code-block} python
:caption: A Warp pair-energy kernel wrapped through ``OpAdapter``. (Excerpted from ``examples/distributed/05_byo_graph_transformer.py``.)
@wp.kernel
def _gaussian_pair_kernel(...):
    ...

@torch.library.custom_op("tutorial::gaussian_pair_energy", mutates_args=())
def gaussian_pair_energy(edge_index, positions, epsilon, sigma, cutoff):
    energy_per_atom = torch.zeros(...)
    wp.launch(_gaussian_pair_kernel, ...)
    return energy_per_atom

# Spec declares the boundary semantics.
spec = MLIPSpec(
    distribution=DistributionSpec(
        policy=HaloStoragePolicy(),
        custom_ops=(
            OpAdapter(
                op=torch.ops.tutorial.gaussian_pair_energy.default,
                arg_transforms={},                       # halo-padded inputs OK as-is
                output_transforms={0: ScatterOutputs()}, # output[0] is per-atom: halo-correct
            ),
        ),
    ),
    output_kinds={"energy": OutputKind.PER_GRAPH, ...},
)
```

The OpAdapter is the *only* distribution-aware code in the wrapper.
The kernel itself stays single-process; the spec parameterises the
cross-rank wrap.

---

## 7. Validation + Bring-Your-Own-Model

### 7.1 The flow

`trace_and_validate` is the BYO author's only required entry point.
A single call: build a sample, point at the model factory, get back a
verdict + a working spec.

```{graphviz}
:caption: ``trace_and_validate`` flow. A single reference run captures the truth; ``world_size`` workers re-run the same factory with a candidate spec; diffs that exceed tolerance trigger the auto-fix engine, which proposes a spec mutation and retries.
:align: center

digraph validate {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];

    Factory [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#dce6f1"><B>model_factory()</B><BR/><FONT POINT-SIZE="10">a callable that returns a fresh wrapper —<BR/>called once for the reference, once per spawned worker</FONT></TD></TR>
      </TABLE>
    >];

    Ref [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#9fc5e8"><B>reference run</B><BR/><FONT POINT-SIZE="10">single-process forward on the sample batch — produces the<BR/>per-output truth tensors plus an op-trace and helper-trace</FONT></TD></TR>
      </TABLE>
    >];

    Spec0 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#cfe2cf"><B>initial candidate spec</B><BR/><FONT POINT-SIZE="10">use ``wrapper.distribution_spec`` if declared,<BR/>otherwise infer a sensible halo default</FONT></TD></TR>
      </TABLE>
    >];

    Spawn [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#fff2cc"><B>spawn ``world_size`` workers</B><BR/><FONT POINT-SIZE="10">each runs the wrapper through the framework with the candidate spec</FONT></TD></TR>
      </TABLE>
    >];

    Diff [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#fff2cc"><B>compare to reference</B><BR/><FONT POINT-SIZE="10">per-output abs/rel diff, op firings, halo completeness,<BR/>helper-diagnostic gaps</FONT></TD></TR>
      </TABLE>
    >];

    Pass [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#82b366"><FONT COLOR="white"><B>report.ok = True</B><BR/><FONT POINT-SIZE="10">save spec, ship alongside checkpoint</FONT></FONT></TD></TR>
      </TABLE>
    >];

    AutoFix [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#f4cccc"><B>auto-fix rule engine</B><BR/><FONT POINT-SIZE="10">try a known mutation:<BR/>• swap halo correction → local<BR/>• promote per-graph autograd output to all-reduce<BR/>• drop a redundant all-reduce</FONT></TD></TR>
      </TABLE>
    >];

    Translate [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#f4cccc"><B>error translator</B><BR/><FONT POINT-SIZE="10">rewrite generic torch errors as<BR/>framework-specific hints (e.g. dropped scatter return)</FONT></TD></TR>
      </TABLE>
    >];

    Factory -> Ref -> Spec0;
    Spec0 -> Spawn -> Diff;
    Diff -> Pass [label=<<FONT POINT-SIZE="10">diff &lt; tolerance</FONT>>];
    Diff -> AutoFix [label=<<FONT POINT-SIZE="10">diff &gt; tolerance</FONT>>];
    Diff -> Translate [label=<<FONT POINT-SIZE="10">worker raised</FONT>>];
    AutoFix -> Spawn [label=<<FONT POINT-SIZE="10">retry with mutated spec</FONT>>];
}
```

### 7.2 What the report carries

```{code-block} python
:caption: The actionable surface. Either ``report.ok`` is True and ``report.spec`` is ready to save, or ``report.next_action`` tells you exactly what's wrong.
report = trace_and_validate(model_factory, sample_batch, world_size=2)

if report.ok:
    report.spec.save("my_model_spec.json")
else:
    report.log_summary(logger)
    # Output includes:
    # - validation status + auto-fix applied
    # - per-output abs/rel diffs vs single-process
    # - dispatch-handler firings (so you can see what the multi-rank
    #   run actually exercised)
    # - halo-completeness verdict
    # - helper-diagnostic gaps from watched third-party packages
    # - "Diagnosis:" hint when an error pattern is recognised
    #   (e.g. dropped scatter_add_ return, missing OpAdapter, etc.)
```

### 7.3 The intended user path

```{graphviz}
:caption: The BYO arc — the same five steps regardless of whether the model is pure PyTorch (example 04) or has a Warp kernel (example 05). Most users finish at step 5 without ever touching step 4.
:align: center

digraph byo {
    rankdir=TB;
    node [shape=plaintext fontname="Helvetica"];

    S1 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#dce6f1"><B>1. write the single-process wrapper</B><BR/><FONT POINT-SIZE="10">no distribution code, no spec — just BaseModelMixin</FONT></TD></TR>
      </TABLE>
    >];

    S2 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#9fc5e8"><B>2. call trace_and_validate</B><BR/><FONT POINT-SIZE="10">no arguments beyond ``model_factory`` + sample batch — auto-fix<BR/>discovers the spec for typical halo MPNNs and per-rank-partial outputs</FONT></TD></TR>
      </TABLE>
    >];

    S3 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#fff2cc"><B>3. read report.log_summary(logger)</B><BR/><FONT POINT-SIZE="10">if it passed, the report shows the residual diff vs single-process;<BR/>if it failed, the diagnostic points at the root cause in plain English</FONT></TD></TR>
      </TABLE>
    >];

    S4 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#cfe2cf"><B>4. (rare) author an OpAdapter</B><BR/><FONT POINT-SIZE="10">only needed if the model embeds a Warp / Triton kernel:<BR/>declare the input pre-shape and output post-shape on the spec</FONT></TD></TR>
      </TABLE>
    >];

    S5 [label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#82b366"><FONT COLOR="white"><B>5. spec.save("model_spec.json")</B><BR/><FONT POINT-SIZE="10">ship alongside the checkpoint;<BR/>production loads it as ``MLIPSpec.load(...)``</FONT></FONT></TD></TR>
      </TABLE>
    >];

    S1 -> S2 -> S3;
    S3 -> S4 [label=<<FONT POINT-SIZE="10">opaque kernel?</FONT>>];
    S3 -> S5 [label=<<FONT POINT-SIZE="10">otherwise</FONT>>];
    S4 -> S2 [label=<<FONT POINT-SIZE="10">re-validate</FONT>>];
}
```

```{code-block} python
:caption: The end-to-end happy path is six lines. (Excerpted from ``examples/distributed/04_byo_pytorch_mpnn.py``.)
def model_factory():
    torch.manual_seed(123)
    return BPWrapper(BPModel(feat_dim=32, cutoff=5.0)).cuda()

report = trace_and_validate(model_factory, sample_batch, world_size=2)
report.log_summary(logger)            # validation PASSED in 1 attempt
report.spec.save("bp_model_spec.json")
# Production:
#     spec = MLIPSpec.load("bp_model_spec.json")
#     dist = DistributedModel(BPWrapper(BPModel()), domain_cfg, spec=spec)
```

---

## What's not in this overview

* **Performance numbers** — see `examples/distributed/benchmark_*.py` and the
  scaling tables those produce. The benchmarks measure per-step wall
  clock, halo-build amortisation, and weak/strong scaling on argon /
  sodium chloride / silica supercells.
* **Checkpoint compatibility** — covered in {doc}`distributed_byo`.
* **The full handler registry** — every predicate + handler is
  documented in {doc}`distributed_shardtensor` (this overview only
  walks the dispatch flow at the conceptual level).
* **Failure-mode catalogue** — common spec mistakes and the
  diagnostics that catch them are in {doc}`distributed_byo` §
  "Common failure modes".

For a runnable end-to-end build, work through the two BYO examples in
order:

| Example | Adds |
|---|---|
| `examples/distributed/04_byo_pytorch_mpnn.py` | The minimal pure-PyTorch path. |
| `examples/distributed/05_byo_graph_transformer.py` | The Warp-kernel path with a hand-authored `OpAdapter`. |
