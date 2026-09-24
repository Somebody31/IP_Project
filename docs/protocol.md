# Evaluation protocol

**Status: not yet frozen.** Agree on this protocol before using validation results to make experiment choices or evaluating the final test split.

Record the following here:

- Dataset versions, paths, split definitions, and any permitted preprocessing.
- Traditional and machine-learning methods to compare, including the untreated hazy input baseline.
- Shared evaluation inputs and metrics, with exact implementation/settings for each metric.
- Which split is used for development and method selection; reserve test data for final scoring.
- Experiment seeds, model/checkpoint selection rules, and result reporting requirements.
- Per-run provenance: configuration, source revision, dataset/input hashes or manifest, software versions, and completion status.

Keep the two method tracks separate in implementation and run records. Compare their results using the same evaluation setup where applicable. Do not report results as a paper reproduction until the protocol has been checked against `paper_notes.md` and the source paper.

## Initial traditional-method benchmark

The initial DCP benchmark uses the 52 validation scenes in `data/foggy_cityscapes/foggy_trainval_refined_filenames.txt`, with all three beta variants (expected 156 pairs). Each Foggy Cityscapes image is paired with the clear Cityscapes image from the same split, city, and image ID. It compares the clear image with itself (reference), the untreated foggy input, and DCP output against the clear target. Report PSNR and SSIM by beta and pooled, plus scene-clustered 95% bootstrap intervals for paired DCP-minus-hazy metric differences. This is a refined-subset validation result, not a full-validation result. Reserve `test` for a later final evaluation.
