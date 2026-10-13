# CLI usage and outputs

Run commands from the repository root. Install the pinned environment in `requirements.txt`. Python 3.14.3 on Windows was used for validation; see the root README for limits. Paths containing spaces need quotes.

When installed as a skill alone, use the `requirements.txt` in that skill folder and invoke `scripts/<script>.py` from that folder. The following paths assume a cloned repository instead.

```shell
python skills/dual-color-live-imaging/scripts/make_synthetic_demo.py --output demo/synthetic.ome.tif
python skills/dual-color-live-imaging/scripts/audit_acquisition.py --input demo/synthetic.ome.tif --out demo/audit
python skills/dual-color-live-imaging/scripts/run_pipeline.py --input demo/synthetic.ome.tif --out demo/analysis --structure any --min-structure-um 10 --min-skeleton-um 20 --min-branch-um 5 --extend-um 0 --n-null 20
python skills/dual-color-live-imaging/scripts/run_onset_order.py --results demo/analysis/results
python skills/dual-color-live-imaging/scripts/make_report_figures.py --results demo/analysis/results --out demo/figures
```

These parameters are a **synthetic smoke-test example**, not recommended settings for biological data. Use a fresh output directory for another run; existing results are protected. All scripts expose `--help`.

For actual data, audit before analysis:

```shell
python skills/dual-color-live-imaging/scripts/audit_acquisition.py --input "input/recording.ome.tif" --out output/audit --channels 0 1
python skills/dual-color-live-imaging/scripts/run_pipeline.py --input "input/recording.ome.tif" --out output/analysis --channels 0 1 --names A B --roi-json input/rois.json --n-null 1000
```

An input directory is scanned for `.czi`, `.tif`, `.tiff`, or restrict with `--glob`. Stems must be unique. TIFF `--series` selects a series. Only when independently verified, add `--axes TCZYX`, `--pixel-size-um 0.4`, `--z-step-um 0.8`, or `--frame-interval-s 2`; they override metadata for **every** selected file. These illustrative numbers must not be assumed for real data. For 2D, no Z calibration is required.

Optional preprocessing: `--bleach-correct` enables multiplicative flattening; `--channel-shift-px DY DX` applies an independently calibrated shift to B in row/column pixels. It is the shift to apply, not the measured B displacement (which has the opposite sign). Check retained overlap and interpolated intensity effects.

ROI JSON uses vertices in `(x,y)` order. Empty or missing exact per-file mappings fail; they never silently fall back to the full image. Overlapping ROIs are permitted but must not be treated as independent observations.

```json
{
  "crop_offset_row": 0,
  "crop_offset_col": 0,
  "rois": {"region_1": [[15, 35], [145, 35], [145, 60], [15, 60]]}
}
```

For multiple files, use `{"per_file":{"exact_file_stem":{...same object...}}}`. A filename `recording.ome.tif` has stem `recording.ome`.

| Output | Content |
| --- | --- |
| `qc_acquisition.csv`, `acquisition.json` | Sampling, dimensions, diagnostics, actual relative times and calibration provenance |
| `params.json` | CLI settings, seed and input/output paths |
| `structures.csv` | Shape-filter acceptance/rejection; empty shape log for `--structure any` |
| `colocalisation.csv` | Counts, fractions, uniform-mask null summaries, descriptive shell ratios |
| `tracks.csv` | Stationary linking, actual observed spans and censoring flags; tracks with fewer than two detections omitted |
| `displacement.csv` | Estimated profile shifts and actual interval durations |
| `masks.npz` | Projections, valid mask, analysis masks, exclusions and applied bleaching factors |
| `clusters.npz` | Detections, full raw traces at selected sites, calibration and actual times |
| `kymographs.npz` | Profiles, skeleton paths, distances and actual times |
| `onset_order_summary.csv`, `onset_order_sites.csv` | Descriptive detected/count-matched/raw onset delays and site-level records |
| Figure folder | PNG/PDF plots and per-structure descriptive means |

NPZ arrays use reserved `|` separators in keys and are loaded without pickle. Results can contain acquisition names and local paths. They are excluded from Git by default; review and anonymize before sharing your own outputs. This repository ships no experimental image data, manuscript or credentials.
