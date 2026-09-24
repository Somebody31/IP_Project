# Data inventory

Keep original dataset paths and splits unchanged. Do not commit or redistribute dataset images or archives without checking their terms. Record checksums and processing details for each experiment; store generated data under `data/derived/` and document how to regenerate it.

## Available datasets

| Dataset | Current path | Notes |
|---|---|---|
| Cityscapes clear images | `data/cityscapes_clear/` | Clear images organized under `leftImg8bit/{train,val,test}`. Preserve the accompanying license and dataset documentation. |
| Foggy Cityscapes | `data/foggy_cityscapes/` | Foggy images under `leftImg8bit_foggy/{train,val,test}`; filenames encode fog density. Preserve its license, README, and refined filename list. |
| Transmittance maps | `data/transmittance/` | Contains a source archive and extracted maps. Record which files are source inputs and which are derived for each run. |
| RTTS export | `rtts/` | YOLO-format `train`, `valid`, and `test` splits with `images/` and `labels/`; `rtts/data.yaml` lists five classes. Its README attributes this export to Roboflow and states CC BY 4.0. Preserve attribution and verify terms before redistribution. |

## Dataset handling

- Preserve the upstream split and folder structure; do not mix train, validation, or test data.
- Keep dataset-provided license and attribution files with their data.
- Treat caches and preprocessed copies as generated files, not original data.
- Before an experiment, record dataset version/source, split counts, preprocessing, and checksums or a suitable file manifest in that run's record.
- Cityscapes materials indicate non-commercial terms; check the included license documents before use or sharing. Do not assume derived images can be redistributed.
- Do not commit large raw images or archives to source control. Keep code, data manifests, and instructions separate from data storage.
