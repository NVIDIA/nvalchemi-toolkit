Intermediate Examples
=====================

These examples assume familiarity with the basic tier and introduce
the storage layer, performance monitoring, and more complex pipeline
patterns. Training examples are labeled explicitly in this list; if the
collection grows further, split training workflows into a dedicated examples
section so inference, dynamics, and training entry points remain easy to scan.

**01 — Multi-Stage Pipeline**: FusedStage composition, LoggingHook CSV output,
step-budget migration, fused hooks for global status monitoring.

**02 — Trajectory I/O**: Writing trajectories to Zarr, reading back with
DataLoader, round-trip validation.

**03 — NPT MD**: Pressure-controlled dynamics with the MTK barostat,
LJ stress computation, cell fluctuation monitoring.

**04 — Inflight Batching**: SizeAwareSampler, Mode 2 FusedStage run (batch=None),
system_id tracking, ConvergedSnapshotHook collecting results.

**05 — Safety and Monitoring**: NaNDetectorHook, MaxForceClampHook,
EnergyDriftMonitorHook, ProfilerHook — defensive MD patterns.

**06 — DDP MLP Training**: DDPHook with a simple MLP, dummy AtomicData,
single-node ``torchrun`` launch, and ``auto``/``gloo``/``nccl`` backend
selection.

**07 — Rich Training Reporting**: Live Rich dashboard driven by synthetic
training losses, validation metrics, progress counters, and learning-rate
scheduler values.
