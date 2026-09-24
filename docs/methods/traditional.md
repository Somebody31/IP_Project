# Traditional method track

Use this document for the selected classical dehazing method, its source/reference, assumptions, parameters, and implementation decisions. Record deviations from the source method and how its outputs enter the shared evaluation.

## Dark Channel Prior (DCP)

The first traditional baseline is the Dark Channel Prior. It estimates the atmospheric light and transmission from a foggy RGB image, refines the transmission with a guided filter, and recovers the scene radiance. The implementation is in `src/ip_dehaze/traditional/dcp.py`; fixed benchmark settings are in `configs/traditional/dcp.yaml`.

The paired benchmark is `scripts/traditional/benchmark_dcp.py`. Its initial run evaluates the 52-scene refined validation subset at all three beta levels against matching clear Cityscapes targets, comparing untreated fog, DCP, and the clear reference. See `docs/protocol.md`. Results are not assumed to improve over the untreated baseline; use the generated per-run metrics to make that determination. Set `refined_only: false` in the config for a separately identified full-split validation run.
