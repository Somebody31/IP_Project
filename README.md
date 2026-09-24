# Dehazing project

This project compares two approaches to image dehazing:

1. **Traditional** — classical image-processing methods.
2. **Machine learning** — learned dehazing models.

Both tracks use the shared datasets and evaluation setup, but keep their code, settings, and experiment runs separate. The reference paper is `assets/2506.15837v1.pdf`; the project should not claim a reproduction until its methods and evaluation have been checked against the paper.

## Project map

- `data/`, `rtts/` — existing datasets; see `data/README.md`.
- `assets/`, `decks/` — reference paper, slide template, and existing presentations.
- `src/ip_dehaze/traditional/` — traditional-method code.
- `src/ip_dehaze/machine_learning/` — machine-learning code.
- `src/ip_dehaze/data/`, `src/ip_dehaze/evaluation/` — shared data and evaluation code.
- `configs/`, `scripts/`, `tests/` — settings, entry points, and tests for each track.
- `runs/`, `results/`, `outputs/` — run records, result summaries, and generated artifacts.
- `docs/` — paper notes, protocol, method notes, decisions, and reports.

## Workflow

1. Verify dataset provenance, licenses, and split integrity.
2. Read the paper and write down its method and evaluation details in `docs/paper_notes.md`.
3. Agree on the comparison protocol in `docs/protocol.md` before experiments.
4. Establish a traditional baseline and a machine-learning baseline.
5. Evaluate both using the same setup and retain a record for every run.

Raw data and generated model outputs are not intended for ordinary source control. See `.gitignore` and `data/README.md` before adding or sharing data.
