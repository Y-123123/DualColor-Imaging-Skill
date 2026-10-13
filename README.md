# Dual-Color Live Imaging Skill

[中文说明](README.zh-CN.md) · [Methodology](skills/dual-color-live-imaging/references/methodology.md) · [CLI reference](skills/dual-color-live-imaging/references/usage.md)

A reusable Codex / Claude Code skill and Python toolkit for auditing two-channel fluorescence time-lapse microscopy of elongated structures. It produces acquisition QC, one-to-one puncta matching, an explicitly defined spatial null, stationary-site tracking, kymographs and descriptive onset-order checks.

**Version 1.0.0:** quantitative metadata is required; biological interpretation remains experiment-specific. This public package contains no experimental images, credentials or private acquisition records.

## Install the skill

```shell
git clone https://github.com/Y-123123/DualColor-Imaging-Skill.git
cd DualColor-Imaging-Skill
```

Copy the whole `skills/dual-color-live-imaging` folder, including scripts and references, to the appropriate personal skill directory:

| Agent | Destination |
| --- | --- |
| Codex | `~/.agents/skills/dual-color-live-imaging/` |
| Claude Code | `~/.claude/skills/dual-color-live-imaging/` |

Back up an existing skill of the same name before replacing it. Restart the agent if it has already loaded the skill inventory. Invoke by name, or ask for a dual-color time-lapse audit. The tools also run directly without an agent or API key.

Example request: “Use dual-color-live-imaging to audit this OME-TIFF. Confirm axes, physical calibration and channel timing before analysis; inspect masks and report descriptive overlap and onset delays with the relevant limitations.”

## Python environment

Python **3.12 or newer** is required by the pinned image readers. The completed checks used **Windows, Python 3.14.3** and the exact dependency versions in `requirements.txt`; other Python/OS combinations have not been validated here.

```shell
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on PowerShell, or `source .venv/bin/activate` on macOS/Linux, then:

```shell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Try without experimental data

```shell
python skills/dual-color-live-imaging/scripts/make_synthetic_demo.py --output demo/synthetic.ome.tif
python skills/dual-color-live-imaging/scripts/audit_acquisition.py --input demo/synthetic.ome.tif --out demo/audit
python skills/dual-color-live-imaging/scripts/run_pipeline.py --input demo/synthetic.ome.tif --out demo/analysis --structure any --min-structure-um 10 --min-skeleton-um 20 --min-branch-um 5 --extend-um 0 --n-null 20
python skills/dual-color-live-imaging/scripts/run_onset_order.py --results demo/analysis/results
python skills/dual-color-live-imaging/scripts/make_report_figures.py --results demo/analysis/results --out demo/figures
```

The generated OME-TIFF is wholly synthetic. These settings exercise execution; choose and validate different parameters for your assay. Use new output folders for repeat runs. See the [CLI reference](skills/dual-color-live-imaging/references/usage.md) for real-data workflows, metadata overrides, channel shifts, ROIs and output definitions.

## Scope and limits

- CZI and TIFF/OME-TIFF inputs with labeled axes; square XY pixels; explicit physical/time calibration; 2D or Z stacks; one selected scene/series. ND2/LIF requires an independently checked export.
- Rigid XY registration, elongated-structure segmentation and local-maxima detection require visual QC. An ROI intersects the segmentation. `--structure any` still uses a tubular filter.
- Bleaching correction is opt-in. Independent channel calibration is required for a supplied chromatic shift. Relative-intensity strata are not positivity controls.
- Spatial null quantiles and conditional empirical probabilities do not establish biological significance. The null does not preserve channel A clustering.
- Observed track spans are censored, stationary-linking summaries. Profile shifts do not prove cargo transport; temporal order does not prove recruitment. Sites and frames are nested observations, not independent biological replicates.
- Arrays are loaded into memory as float32. Extremely large acquisitions and high-precision quantitative intensities need another validated workflow.

The [methodology](skills/dual-color-live-imaging/references/methodology.md) describes these assumptions and primary references in detail. Inspect raw data, masks, detections and sensitivity to parameters before using results in a manuscript.

## Validation and provenance

The test suite checks axis permutation, units, irregular OME timestamps, unknown calibration rejection, anisotropic-pixel rejection, mocked CZI metadata adaptation, registration/shift direction, absent-channel handling, maximum-cardinality matching, seeded nulls, count selection, onset censoring/times, displacement sign and missing-ROI rejection. It also executes synthetic generation, acquisition audit, pipeline, onset and PNG/PDF report creation, and checks output protection.

Real CZI acquisition files, biological accuracy, cross-platform execution, manual figure interpretation and agent behavior across models were **not** validated by these tests. This release is an audited exploratory implementation, not a validated microscopy standard.

Built from an existing local imaging-analysis workflow, reorganized and revised for public reuse. No unpublished experimental findings or datasets are included. Third-party dependencies are installed separately under their own licenses; none are vendored. MIT license applies to this repository's code and skill materials.

## Share results deliberately

Generated outputs may include acquisition filenames and local paths. `.gitignore` excludes common data/result/manuscript formats, but it cannot anonymize arbitrary files. Review your own outputs before publishing them. Releases include source and fictional/synthetic generators, not generated data.
