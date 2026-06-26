# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Domain-decomposed composition of models (``DistributedPipelineModel``).

Runs a :class:`~nvalchemi.models.pipeline.PipelineModelWrapper` (e.g.
MACE + DFT-D3, AIMNet2 + PME) under domain decomposition, giving each
sub-model its own right-sized halo over one shared owned partition rather
than forcing every sub-model onto the largest cutoff's ghost region.

The caller builds one ``ShardedBatch`` at the max cutoff (one owned
partition); the composite rebuilds each sub-model's halo over it via
``ShardedBatch.invalidate_padded_view`` between models, so the owned set is
never recomputed. It mirrors
:class:`~nvalchemi.distributed.distributed_model.DistributedModel`'s
context-manager + ``__call__(sharded_batch)`` contract, so it is drop-in for
``BaseDynamics``.

Three group kinds:

* **Direct-force** (``use_autograd=False``): sub-models whose forces are
  direct per-atom kernel outputs (DFT-D3, Lennard-Jones, Ewald / PME with
  ``hybrid_forces=True``) compose by summing owned-aligned per-atom energies /
  forces / stresses — no cross-model autograd.

* **Shared-autograd** (``use_autograd=True``, e.g. MACE energy → ``-dE/dr``):
  the group force is ``-d(ΣE_m)/dr`` over the summed energy. With one shared
  owned partition and no cross-model coupling it decomposes exactly into
  ``Σ_m (-dE_m/dr_owned)``, so each sub-model runs its own autograd forward
  (forces enabled) and the owned-aligned results are summed — identical to a
  single shared ``positions`` leaf with one ``backward()``, while reusing each
  model's eager / compile paths.

* **Wired cross-model fields**: a consumer's energy depends on a per-atom field
  the producer makes (e.g. PME's energy on AIMNet2's ``charges``), so the two
  models share one autograd graph and can't run independently. See
  :meth:`_run_wired_autograd_group`.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nvalchemi.distributed.config import DomainConfig
    from nvalchemi.distributed.sharded_batch import ShardedBatch
    from nvalchemi.models.pipeline import PipelineModelWrapper

__all__ = ["DistributedPipelineModel"]


class DistributedPipelineModel:
    """Domain decomposition of a composed (pipeline) model.

    Parameters
    ----------
    pipeline : PipelineModelWrapper
        The composed model. Its groups / steps supply the ordered sub-models;
        each sub-model's ``model_config.neighbor_config.cutoff`` sets that
        model's ghost width.
    domain_config : DomainConfig
        Base config carrying the mesh + skin. Per-model configs are derived
        from it by overriding ``cutoff`` with each sub-model's cutoff. The
        caller should build the shared :class:`ShardedBatch` at (at least) the
        **max** sub-model cutoff so the one owned partition's cells hold every
        model's ghost layer.

    Notes
    -----
    The composite does not build the ``ShardedBatch`` — the caller does (once,
    at the max cutoff), exactly as for a single :class:`DistributedModel`. The
    composite only orchestrates per-model halos over it and sums the results.
    """

    def __init__(
        self,
        pipeline: "PipelineModelWrapper",
        domain_config: "DomainConfig",
        *,
        compile: bool = False,
        compile_kwargs: "dict | None" = None,
    ) -> None:
        from nvalchemi.models.pipeline import PipelineModelWrapper  # noqa: PLC0415

        if not isinstance(pipeline, PipelineModelWrapper):
            raise TypeError(
                "DistributedPipelineModel expects a PipelineModelWrapper; for a "
                "single model use DistributedModel."
            )

        self.pipeline = pipeline
        self.domain_config = domain_config
        self.additive_keys = pipeline.additive_keys
        # Only sub-models with an autograd-force compiled path (MACE, AIMNet2)
        # compile; kernel-force models (DFTD3 / PME / Ewald / LJ) and the
        # composite glue always run eager.
        self._compile = bool(compile)
        self._compile_kwargs = compile_kwargs
        self._closed = False
        # Per-group plans: "per_step" (no cross-step field dependency — run each
        # sub-model independently and sum) or "wired" (a later step consumes a
        # per-atom field an earlier step produces, e.g. PME needs AIMNet2's
        # charges — one coupled autograd graph). Per-step plans hold persistent
        # DistributedModel instances so compiled graphs survive across MD steps.
        self._group_plans: list[dict[str, Any]] = [
            self._plan_group(g) for g in pipeline.groups
        ]

    # Fields always present on a Batch, so never a cross-step wired dependency.
    _BATCH_FIELDS = frozenset(
        {"positions", "atomic_numbers", "atomic_masses", "cell", "pbc", "energy", "forces"}
    )

    def _model_cfg(self, step: Any) -> "DomainConfig":
        """Per-model :class:`DomainConfig` (this sub-model's ghost width)."""
        return self.domain_config.model_copy(
            update={"cutoff": step.model.model_config.neighbor_config.cutoff}
        )

    @staticmethod
    def _compile_capable(step: Any) -> bool:
        """Whether a sub-model declares an autograd-force compiled path."""
        spec = getattr(step.model, "distribution_spec", None)
        cp = getattr(spec, "compile", None)
        return bool(cp is not None and getattr(cp, "forces_via_autograd", False))

    def _make_dist_model(self, step: Any, cfg: "DomainConfig") -> Any:
        """Build a (possibly compiled) persistent :class:`DistributedModel`."""
        from nvalchemi.distributed.distributed_model import (  # noqa: PLC0415
            DistributedModel,
        )

        compiled = self._compile and self._compile_capable(step)
        return DistributedModel(
            step.model,
            cfg,
            compile=compiled,
            compile_kwargs=self._compile_kwargs if compiled else None,
        ), compiled

    def _plan_group(self, group: Any) -> dict[str, Any]:
        """Classify a pipeline group as per-step or wired.

        A *wired* group has a step whose ``required_inputs`` (excluding always-
        present batch fields) is produced — after any ``PipelineStep.wire``
        rename — by an earlier step in the same group. Only one such
        producer->consumer field over a two-step group is supported; anything
        richer raises ``NotImplementedError``.
        """
        produced: dict[str, tuple[Any, str]] = {}
        wired: list[tuple[Any, str, Any, str]] = []
        for step in group.steps:
            needed = set(step.model.model_config.required_inputs) - self._BATCH_FIELDS
            for f in needed:
                if f in produced:
                    p_step, p_out = produced[f]
                    wired.append((p_step, p_out, step, f))
            for out_key in step.model.model_config.outputs:
                produced[step.wire.get(out_key, out_key)] = (step, out_key)

        if not wired:
            steps = []
            for s in group.steps:
                cfg = self._model_cfg(s)
                dm, compiled = self._make_dist_model(s, cfg)
                steps.append(
                    {
                        "step": s,
                        "use_autograd": group.use_autograd,
                        "dm": dm,
                        "compiled": compiled,
                    }
                )
            return {"kind": "per_step", "steps": steps}

        if not group.use_autograd:
            raise NotImplementedError(
                "DistributedPipelineModel supports wired cross-model fields only "
                "in shared-autograd groups (use_autograd=True); the charge "
                "pathway (e.g. AIMNet2 -> PME) needs the combined-energy autograd."
            )
        if len(wired) != 1 or len(group.steps) != 2:
            raise NotImplementedError(
                "DistributedPipelineModel C3 supports exactly one producer->"
                "consumer wired field over a two-step group; got "
                f"{len(wired)} wired field(s) over {len(group.steps)} steps."
            )
        producer, out_key, consumer, field = wired[0]
        return {
            "kind": "wired",
            "producer": producer,
            "consumer": consumer,
            "producer_out_key": out_key,
            "field": field,
            "producer_cfg": self._model_cfg(producer),
            "consumer_cfg": self._model_cfg(consumer),
        }

    # ------------------------------------------------------------------
    # Context-manager contract (mirrors DistributedModel for drop-in use)
    # ------------------------------------------------------------------

    def __enter__(self) -> "DistributedPipelineModel":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Tear down the persistent per-model :class:`DistributedModel` instances
        (restoring their adapters / compiled state). Idempotent."""
        if self._closed:
            return
        for plan in self._group_plans:
            for item in plan.get("steps", ()):
                item["dm"].close()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: S110
            pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, sharded: "ShardedBatch") -> dict[str, Any]:
        """Run each sub-model over the shared owned partition; sum the results.

        For every sub-model: drop the previous model's ghost layer
        (``invalidate_padded_view``) so the halo is rebuilt at *this* model's
        ghost width, run it through a :class:`DistributedModel`, and accumulate
        its owned-aligned outputs. All sub-models see the same owned atoms, so
        per-atom forces line up by owned index and per-system energies /
        stresses add directly.

        For a **shared-autograd** sub-model the per-model ``active_outputs`` is
        temporarily widened to include the group's derivative keys (``forces`` /
        ``stress``) so the sub-model's own forward emits them via autograd;
        summing those is exactly the group's ``-d(ΣE_m)/dr``.

        Parameters
        ----------
        sharded : ShardedBatch
            The shared partition, built once by the caller at the max cutoff.

        Returns
        -------
        dict[str, Any]
            Summed outputs (``energy`` / ``forces`` / ``stress`` over
            ``additive_keys``); any non-additive key takes the first
            sub-model that produced it.
        """
        per_model: list[dict[str, Any]] = []
        for plan in self._group_plans:
            if plan["kind"] == "wired":
                per_model.append(self._run_wired_autograd_group(sharded, plan))
                continue
            for item in plan["steps"]:
                step = item["step"]
                # Each sub-model rebuilds its own halo over the shared owned set.
                sharded.invalidate_padded_view()
                mc = step.model.model_config
                saved_active = mc.active_outputs
                # A compiled sub-model already emits forces via its compiled
                # energy-autograd path, so only widen active_outputs when eager.
                if item["use_autograd"] and not item["compiled"]:
                    mc.active_outputs = self._autograd_active_outputs(step)
                try:
                    per_model.append(item["dm"](sharded))
                finally:
                    mc.active_outputs = saved_active

        return self._combine(per_model)

    def _run_wired_autograd_group(
        self, sharded: "ShardedBatch", plan: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a wired producer->consumer shared-autograd group.

        The consumer's energy depends on a per-atom field the producer computes
        (e.g. PME's energy on AIMNet2's ``charges``), so the two models form one
        coupled autograd graph. The total force is::

            F = -dE_prod/dr               (producer's own energy)
                - (dE_cons/dfield)(dfield/dr)   (the cross-model chain)
                - dE_cons/dr|_field        (consumer's direct kernel force)

        Realized as: run the producer for energy + field only (no forces, so its
        graph is retained), with grad-bearing positions via a
        ``compute_forces=True`` halo. Run the consumer with its ghost field
        gathered from the producer's owned values through the autograd-aware
        ``wired_fields`` exchange, taking its direct kernel forces and the field
        gradient ``dE_cons/dfield`` on the producer's owned atoms. One backward
        through the producer over ``E_prod + <field, dE_cons/dfield>`` yields the
        first two force terms; that gradient is sent back to the owning ranks and
        the per-rank replication from the two all-reduced energies is divided
        out; the consumer's owned kernel forces are then added.
        """
        import torch  # noqa: PLC0415

        from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
            halo_reverse_exchange,
        )
        from nvalchemi.distributed.distributed_model import (  # noqa: PLC0415
            DistributedModel,
        )
        from nvalchemi.distributed.helpers import to_local  # noqa: PLC0415
        from nvalchemi.distributed.particle_halo import (  # noqa: PLC0415
            halo_exchange,
        )

        producer = plan["producer"].model
        consumer = plan["consumer"].model
        out_key = plan["producer_out_key"]
        field = plan["field"]

        prod_mc = producer.model_config
        saved_prod = prod_mc.active_outputs
        prod_mc.active_outputs = {"energy", out_key}
        try:
            sharded.invalidate_padded_view()
            with DistributedModel(producer, plan["producer_cfg"]) as pdm:
                # Build the producer halo with grad-bearing positions so the
                # energy / field graph reaches them, even though the producer
                # emits no forces of its own.
                pdm._ensure_initialized(sharded)
                halo_exchange(sharded, pdm._halo_config, compute_forces=True)
                prod_out = pdm(sharded)
                e_prod = prod_out["energy"]
                owned_field = prod_out[out_key]
                prod_pos_leaf = sharded.padded_batch.positions
                prod_meta = sharded.halo_meta
                prod_halo_cfg = pdm._halo_config
                world_size = pdm._world_size or 1

                # Consumer: direct kernel force + energy differentiable in the
                # wired field, whose ghost values are gathered (autograd-aware)
                # from the producer's owned values.
                sharded.invalidate_padded_view()
                cons_mc = consumer.model_config
                saved_cons = cons_mc.active_outputs
                cons_mc.active_outputs = {"energy", "forces"}
                try:
                    with DistributedModel(consumer, plan["consumer_cfg"]) as cdm:
                        cons_out = cdm(sharded, wired_fields={field: owned_field})
                        e_cons = cons_out["energy"]
                        f_cons_direct = cons_out.get("forces")
                        (de_dfield,) = torch.autograd.grad(
                            [e_cons.sum()],
                            [owned_field],
                            retain_graph=True,
                            allow_unused=True,
                        )
                finally:
                    cons_mc.active_outputs = saved_cons

                # One backward through the producer for -dE_prod/dr and the chain
                # -(dE_cons/dfield)(dfield/dr) together.
                surrogate = e_prod.sum()
                if de_dfield is not None:
                    surrogate = surrogate + (owned_field * de_dfield.detach()).sum()
                (g_pos,) = torch.autograd.grad(
                    [surrogate],
                    [prod_pos_leaf],
                    retain_graph=False,
                    allow_unused=True,
                )
        finally:
            prod_mc.active_outputs = saved_prod

        forces = None
        if g_pos is not None:
            owned_grad = halo_reverse_exchange(
                to_local(g_pos), prod_meta, prod_halo_cfg
            )
            forces = -(owned_grad / world_size)
        if f_cons_direct is not None:
            forces = f_cons_direct if forces is None else forces + f_cons_direct

        out: dict[str, Any] = OrderedDict()
        out["energy"] = (e_prod + e_cons).detach()
        if forces is not None:
            out["forces"] = forces.detach()
        return out

    def _autograd_active_outputs(self, step: Any) -> set[str]:
        """Widen a shared-autograd sub-model's ``active_outputs`` to emit the
        group's derivatives via its own autograd.

        Where the single-process pipeline strips ``forces`` / ``stress`` from a
        sub-model and computes them once from the summed energy, the distributed
        composite instead has each sub-model produce them, so the owned-aligned
        per-model forces sum to the group force. Only keys the pipeline produces
        *and* the sub-model can emit are added; ``energy`` is always kept.
        """
        base = set(step.model.model_config.active_outputs) | {"energy"}
        wanted = {"forces", "stress"} & set(self.pipeline.model_config.active_outputs)
        producible = set(step.model.model_config.outputs)
        return base | (wanted & producible)

    def _combine(self, per_model: list[dict[str, Any]]) -> dict[str, Any]:
        """Sum owned-aligned additive outputs across sub-models."""
        out: dict[str, Any] = OrderedDict()
        seen: list[str] = []
        for result in per_model:
            for key in result:
                if key not in seen:
                    seen.append(key)
        for key in seen:
            vals = [r[key] for r in per_model if key in r and r[key] is not None]
            if not vals:
                continue
            if key in self.additive_keys:
                acc = vals[0]
                for v in vals[1:]:
                    acc = acc + v
                out[key] = acc
            else:
                out[key] = vals[0]
        return out
