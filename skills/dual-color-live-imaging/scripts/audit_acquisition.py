"""Acquisition audit with explicit calibration and conservative interpretation."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import dcli_lib as L
from image_io import read_image, add_input_arguments, input_options, collect_inputs, validate_channels


def audit_one(path, ch_a=0, ch_b=1, **options):
    acq, data = read_image(path, **options)
    validate_channels(acq, [ch_a, ch_b])
    T, C, Z, Y, X = data.shape
    shifts = L.estimate_drift(data)
    registered = L.apply_drift(data, shifts)
    valid = L.valid_mask((Y, X), shifts)
    _, factors, raw = L.bleach_factors(registered, valid, strict=False)
    offset, _, spread = L.channel_offset_um(registered, ch_a, ch_b, acq.px_um)
    frame_drift = np.hypot(*np.diff(shifts, axis=0).T)
    saturation = float(100 * np.mean(data >= 2 ** acq.bit_depth - 1)) if acq.bit_depth else np.nan
    row = dict(file=Path(path).name, T=T, C=C, Z=Z, Y=Y, X=X,
               analysis_dimension='2D' if Z == 1 else '3D', dt_s=acq.dt_s,
               min_dt_s=float(np.min(np.diff(acq.times_s))) if T > 1 else np.nan,
               max_dt_s=float(np.max(np.diff(acq.times_s))) if T > 1 else np.nan,
               duration_min=acq.duration_min, px_um=acq.px_um,
               dz_um=acq.dz_um if Z > 1 else np.nan, bit_depth=acq.bit_depth,
               pct_zero_voxels=float(100 * np.mean(data == 0)), pct_saturated=saturation,
               p99_9=float(np.percentile(data, 99.9)),
               drift_total_um=float(np.hypot(*shifts[-1]) * acq.px_um),
               drift_px_per_frame=float(np.median(frame_drift)) if T > 1 else np.nan,
               axial_centroid_range_um=float(np.ptp(L.axial_centroid(data)) * acq.dz_um) if Z > 1 else np.nan,
               image_channel_offset_um=offset, image_channel_offset_px=offset / acq.px_um,
               channel_offset_tile_spread_px=spread,
               signal_change_A_pct=float(100 * (raw[-1, ch_a] / raw[0, ch_a] - 1)) if raw[0,ch_a]>0 else np.nan,
               signal_change_B_pct=float(100 * (raw[-1, ch_b] / raw[0, ch_b] - 1)) if raw[0,ch_b]>0 else np.nan,
               max_bleach_correction_factor=float(np.nanmax(factors)) if np.isfinite(factors).any() else np.nan,
               notes='; '.join(acq.notes))
    return row, acq


def verdict(row, acq):
    output = [f"{row['analysis_dimension']}; {row['T']} frames; XY={row['px_um']:.4g} um/pixel",
              f"dt={row['dt_s']:.4g} s; duration={row['duration_min']:.4g} min",
              'Image-derived channel offsets are diagnostics, not bead-based chromatic calibration.',
              'Temporal/spatial sampling alone does not establish transport resolvability.']
    if abs(row['signal_change_A_pct']) > 15 or abs(row['signal_change_B_pct']) > 15:
        output.append('Large intensity change: distinguish bleaching, biology, focus and exposure before correction.')
    if row['pct_saturated'] > 0:
        output.append('Saturated voxels detected; quantify their location and impact.')
    return output + acq.notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--channels', type=int, nargs=2, default=[0, 1])
    parser.add_argument('--glob')
    add_input_arguments(parser)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'qc_acquisition.csv'
    if target.exists():
        parser.error('QC output already exists; choose a new --out directory.')
    rows = []
    for path in collect_inputs(args.input, args.glob):
        row, acq = audit_one(path, *args.channels, **input_options(args))
        rows.append(row)
        print(path.name + "\n" + "\n".join(verdict(row, acq)))
    pd.DataFrame(rows).to_csv(target, index=False)
    print('Wrote', target)


if __name__ == '__main__':
    main()
