# Methodology and interpretation

## Input and calibration

OME physical sizes and times are converted to micrometres and seconds. Complete OME `Plane DeltaT` values for reference channel 0 / plane 0 take precedence over a regular `TimeIncrement`. Different channel/plane acquisition times are not aligned or interpolated by this software. CZI scaling is read in metres and timestamp attachments are used only when their count matches the time axis. Unsupported or missing quantitative calibration produces an error rather than a guessed value.

TIFF resolution is physical only when its unit, or a declared ImageJ unit, has physical meaning. Anisotropic XY pixels, multiple CZI scenes, ND2/LIF input and negative/non-finite intensity arrays are unsupported. Work on a documented export/resampling when needed. `--axes` changes labels only: it cannot reconstruct lost metadata. Source files are read without modification; working arrays are float32 and held in memory. Large acquisitions may exceed RAM, and integer precision above float32 capacity is not retained.

## Registration and masks

Rigid XY shifts are estimated from the summed-channel maximum projection relative to frame 0 and applied identically to all channels and Z planes. Changing structures/intensities can bias this estimate; the procedure does not correct nonrigid motion, Z motion or rolling/sequential acquisition. Inspect stable landmarks or bead controls. Common valid XY pixels exclude shifted borders.

The analysis mask uses a multiscale tubular filter, hysteresis/raw percentiles, morphological cleanup, a skeleton graph, and optional compact-object rejection. Several cleanup/band sizes are specified in pixels inside the implementation; they do not form a fully scale-invariant segmentation method. Thin branches, disconnected components and closed loops may be omitted or traced incorrectly. Inspect the masks and paths at the actual magnification. An ROI intersects this estimated mask; it does not replace segmentation or define a manually calibrated background.

Relative log2(A/B) strata are exploratory. The largest-gap automatic split does not prove bimodality, positivity or a biological control population. Exposure, gain, background and fluorophore brightness affect ratios. Regions and acquisition files remain separately labeled in outputs.

## Detection and spatial matching

White top-hat intensities are thresholded using candidate background median + `bg_k * 1.4826 * MAD` from the same frame/channel. Confirm this background is appropriate. Local maxima are not fitted PSFs or identified particles. Peak suppression is voxel-based; coarse/anisotropic Z sampling and noise affect detection. The intensity helper measures a local-box maximum, not integrated calibrated photon counts.

Matching maximizes the number of one-to-one pairs within separate XY and Z tolerances, then minimizes normalized distance. Tolerances require optical resolution, registration uncertainty and an assay-specific rationale. Denominators: pairs/A, pairs/B and pairs/min(A,B). The paired fraction is undefined when either channel has no detections; those entries are NaN rather than zero.

The spatial null fixes B, mask and detection counts while placing A uniformly at distinct mask voxels. It does not retain A's clustering, gradients or temporal structure. Simulation 2.5/97.5 percentiles are null-distribution quantiles, not a confidence interval for biological effect size. `p_empirical = (1 + number of null fractions >= observed)/(1 + simulations)` is conditional on this null; it is not a replicate-level test, is uncorrected for multiple comparisons and must not be used to label biological significance. Null zero means yield undefined enrichment, not infinite evidence. More simulations improve Monte Carlo precision but cannot repair a wrong null hypothesis.

Shell ratios compare 3D mask intensity with a local shell from a maximum projection; their denominator is intentionally projection-based and is not a volumetrically matched concentration estimate. Halo, projection, autofluorescence and mask choices can change them.

## Persistence, profiles and onset

Stationary-site nearest-neighbor linking has a 1.5 um XY limit and may bridge one missing frame. It is order-dependent and cannot reliably resolve collisions, splitting, fusing or fast transport. Exported spans use actual timestamps; boundary flags identify censoring. Density flags are user-selected QC thresholds, not physical resolution guarantees.

Skeleton paths are directed from a geometrically thick region to a tip. This does not establish soma/distal identity. Branch paths can share their proximal portion; do not sum their lengths or observations as independent units. Cross-correlation uses interpolated profiles, requires sufficient overlap and can confuse intensity changes with displacement. Scale separation is a smoothing heuristic, not identification of shaft/cargo compartments.

Onset sites merge detections within spatial tolerances across time. Nearby/moving spots may merge into one site, and nondetected sites are absent. Count matching selects the brightest `min(count_A,count_B)` detections **per frame**; it addresses a count imbalance but not brightness, SNR or segmentation bias. All-frame raw traces are measured at fixed, detection-selected positions before bleaching correction, using local maxima. Their first half-range crossing is an exploratory summary sensitive to noise, outliers, baseline and observation duration. It is not threshold-free recruitment kinetics. Frame-zero onsets are left-censored and excluded from delay summaries. Sites observed in only one channel do not enter paired onset summaries. A positive delta means B is later than A; delays use actual timestamps, including irregular sampling.

## Evidence and reporting

Use negative/single-label controls, independent channel calibration, appropriate stable landmarks, representative annotation and sensitivity analysis on matching/detection parameters. Record independent cultures/animals/participants, sampling structure, exclusions and analysis decisions before making inference. No bundled script fits a hierarchical model or performs biological replicate inference.

Useful primary references:

- [OME-TIFF model specification](https://docs.openmicroscopy.org/ome-model/latest/ome-tiff/specification.html) for file structure and metadata.
- [tifffile documentation](https://www.cgohlke.com/docs/tifffile/) and [czifile source](https://github.com/cgohlke/czifile) for reader behavior.
- [Dunn, Kamocka & McDonald (2011), A practical guide to evaluating colocalization in biological microscopy](https://doi.org/10.1152/ajpcell.00462.2010) for interpretation and controls.

These references inform interpretation; this package is not a validated implementation of every method in those publications.
