#!/usr/bin/env python
"""
Steps 1-6 of the dual-color-live-imaging skill.

    python run_pipeline.py --input <dir-or-file> --out <outdir> \
        [--channels 0 1] [--names A B] [--structure tubular|any]

Writes to <outdir>/results:
    qc_acquisition.csv        acquisition audit
    structures.csv            which regions were kept/rejected and why
    colocalisation.csv        per frame per structure: counts, matched fraction,
                              null distribution, empirical p, shell ratios
    tracks.csv                one row per tracked cluster (persistence)
    displacement.csv          per-frame along-path displacement + scale split
    kymographs.npz            distance x time matrices for every branch
    masks.npz                 analysis masks, excluded blobs, projections
"""
from __future__ import annotations

import argparse, sys, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import dcli_lib as L
from image_io import read_image, add_input_arguments, input_options, collect_inputs, validate_channels
from audit_acquisition import audit_one
from run_onset_order import build_sites


def analyse_file(path, args, rng):
    acq, data = read_image(path, **input_options(args))
    validate_channels(acq, args.channels)
    px, dz = acq.px_um, acq.dz_um
    ca, cb = args.channels
    na, nb = args.names
    print(f"\n{'='*74}\n{Path(path).name}\n  dt={acq.dt_s:.2f}s  T={acq.shape[0]}  "
          f"Z={acq.shape[2]}  px={px:.4f}um  dz={dz:.2f}um  {acq.bit_depth}-bit",
          flush=True)

    if acq.shape[0] < 2:
        raise ValueError('The time-lapse pipeline needs at least two frames; use the audit for static images.')
    T, C, Z, Y, X = data.shape

    # every length parameter is specified in um and converted here, so that the same
    # settings work at any zoom / pixel size
    def to_px(um, lo=1):
        return max(int(round(um / acq.px_um)), lo)

    sigmas = tuple(max(s / acq.px_um, 0.7) for s in args.tube_sigmas_um)
    print(f"  px-converted: tube sigmas {tuple(round(s,2) for s in sigmas)}, "
          f"tophat {to_px(args.tophat_um)}, blob-open {to_px(args.blob_open_um)}, "
          f"min skeleton {to_px(args.min_skeleton_um)} px", flush=True)

    shifts = L.estimate_drift(data)
    reg = L.apply_drift(data, shifts)
    vm = L.valid_mask((Y, X), shifts)

    off_um, off_yx, off_spread = L.channel_offset_um(reg, ca, cb, acq.px_um)
    off_px = off_um / acq.px_um
    if args.channel_shift_px is not None:
        shift = np.array(args.channel_shift_px, dtype=float)
        reg = L.apply_channel_offset(reg, cb, shift)
        vm &= L.valid_mask((Y, X), np.array([[0., 0.], shift]))
        print(f"  applied independently calibrated B-channel shift {shift.tolist()} px")
    elif off_px > args.max_channel_offset_px:
        print(f"  NOTE image-derived offset {off_px:.2f} px (tile spread {off_spread:.2f}); "
              "check independent chromatic calibration. Biological differences can mimic an offset.")

    raw_registered = reg.copy()
    if args.bleach_correct:
        reg, fac, raw = L.bleach_factors(reg, vm)
    else:
        _, fac, raw = L.bleach_factors(reg, vm, strict=False)
        fac = np.ones_like(fac)
    reg *= vm
    hat = L.tophat(reg, radius_px=to_px(args.tophat_um)) * vm
    mip = reg.max(axis=2)
    tavg = mip.mean(axis=0)

    # ---- optional user-drawn ROIs -----------------------------------------
    # When ROIs are supplied each one is segmented and reported SEPARATELY; regions
    # are never pooled, because pooling averages away exactly the between-structure
    # differences the ROIs were drawn to test.
    rois = L.load_rois(args.roi_json, (Y, X), Path(path).stem) if args.roi_json else None
    if rois:
        print(f"  {len(rois)} ROI(s): " +
              ", ".join(f"{k} ({v.sum()} px)" for k, v in rois.items()), flush=True)

    # ---- structures --------------------------------------------------------
    seg = L.segment_tubular(tavg[ca] + tavg[cb], valid=vm,
                            seed_pct=args.seed_pct, grow_pct=args.grow_pct,
                            raw_pct=args.raw_pct, sigmas=sigmas)
    if args.structure == "tubular":
        comps, slens, blobs, shape_log = L.tubular_components(
            seg, min_skel_px=to_px(args.min_skeleton_um), min_aspect=args.min_aspect, max_solidity=args.max_solidity,
            open_radius_px=to_px(args.blob_open_um), verbose=True)
    else:
        comps = [seg]
        slens = [int(np.sum(L.skeletonize(seg)))]
        blobs = np.zeros_like(seg)
        shape_log = []
    print(f"  kept {len(comps)} structure(s): "
          f"{[round(s * px, 1) for s in slens]} um (skeleton-pixel approximation); "
          f"{100 * blobs.sum() / max(seg.sum(), 1):.0f}% excluded as compact",
          flush=True)
    if not comps:
        return None

    # ---- exploratory relative-intensity stratification (not label positivity)
    # If ROIs were supplied, each retained component is first intersected with each ROI
    # and the pieces are kept apart under the ROI's name.
    if rois:
        pieces = []
        for roi_name, roi in rois.items():
            for k, comp in enumerate(comps):
                sub = comp & roi
                if sub.sum() < 40:
                    continue
                pieces.append((f"{roi_name}_s{k}", sub))
    else:
        pieces = [(f"s{k}", comp) for k, comp in enumerate(comps)]

    structures = {}
    for tag, comp in pieces:
        groups, split = L.split_by_channel_ratio(comp, tavg[ca], tavg[cb],
                                                 split=args.ratio_split)
        for cls, segs in groups.items():
            if not segs:
                continue
            m = np.zeros(comp.shape, bool)
            for s in segs:
                m |= L.band_mask(s["path"], comp.shape, halfwidth=3)
            m &= ~blobs
            if rois:
                m &= rois[tag.rsplit("_s", 1)[0]]
            length_um = sum(s["length"] for s in segs) * px
            if length_um < args.min_structure_um:
                continue
            label_cls = cls
            key = f"{tag}_{label_cls}" if not rois else (
                tag if len(groups) == 1 else f"{tag}_{label_cls}")
            if not m.any():
                continue
            structures[key] = dict(mask=m, cls=label_cls, length_um=length_um,
                                   split=split, roi=tag if rois else None)
        if split is not None:
            print(f"    {tag}: bimodal channel ratio, split at "
                  f"log2 = {split:.2f}", flush=True)

    if not structures:
        raise ValueError('No structures meet the analysis criteria; revise ROI/segmentation parameters.')
    # ---- one absolute threshold per frame+channel from cell-free background
    any_s = np.zeros((Y, X), bool)
    for s in structures.values():
        any_s |= s["mask"]
    bg2d = (vm & ~ndi.binary_dilation(any_s, iterations=8)
            & ~ndi.binary_dilation(blobs, iterations=6))
    THR = np.array([[L.background_threshold(hat[t, c], bg2d, k=args.bg_k)
                     for c in range(C)] for t in range(T)])

    # floor at 2 px: with min_distance = 1 every local maximum of the noise becomes a
    # cluster, which is how a coarsely sampled stack yields impossible densities
    sep_px = max(to_px(args.cluster_sep_um), 2)
    if to_px(args.cluster_sep_um) < 2:
        print(f"  NOTE cluster separation {args.cluster_sep_um} um is under 2 px at "
              f"{acq.px_um:.2f} um/px; raised to 2 px", flush=True)
    coloc, trk, disp, kymo, clusters = [], [], [], {}, {}

    for sid, S in structures.items():
        m2 = S["mask"]
        ref = reg.mean(axis=0)[ca] + reg.mean(axis=0)[cb]
        m3 = L.structure_mask_3d(ref, m2)
        shellA = L.shell_ratio(reg[:, ca].mean(0), m2, m3, vm)
        shellB = L.shell_ratio(reg[:, cb].mean(0), m2, m3, vm)

        ptsA, ptsB = [], []
        for t in range(T):
            pA = L.detect_clusters_3d(hat[t, ca], m3, THR[t, ca], sep_px)
            pB = L.detect_clusters_3d(hat[t, cb], m3, THR[t, cb], sep_px)
            ptsA.append(pA); ptsB.append(pB)
            pr = L.match_clusters(pA, pB, px, dz, args.match_xy_um, args.match_z_um)
            denom = min(len(pA), len(pB))
            obs = len(pr) / denom if denom else np.nan
            null = L.null_match_fraction(pA, pB, m3, px, dz, n_iter=args.n_null,
                                         rng=rng, max_xy_um=args.match_xy_um,
                                         max_z_um=args.match_z_um)
            coloc.append(dict(
                file=Path(path).name, structure=sid, structure_class=S["cls"],
                length_um=S["length_um"], frame=t, time_min=acq.times_s[t] / 60,
                n_A=len(pA), n_B=len(pB), n_pairs=len(pr),
                frac_A_with_B=len(pr) / len(pA) if len(pA) else np.nan,
                frac_B_with_A=len(pr) / len(pB) if len(pB) else np.nan,
                matched_fraction=obs,
                dens_pairs_per_10um=10 * len(pr) / S["length_um"],
                null_mean=float(np.mean(null)) if denom else np.nan,
                null_p2_5=float(np.percentile(null, 2.5)) if denom else np.nan,
                null_p97_5=float(np.percentile(null, 97.5)) if denom else np.nan,
                enrichment=float(obs / np.nanmean(null)) if denom and np.nanmean(null) > 0 else np.nan,
                p_empirical=float((1 + np.sum(null >= obs)) / (1 + len(null)))
                if denom else np.nan,
                A_over_shell=shellA, B_over_shell=shellB))

        sites = build_sites(ptsA, ptsB, px, dz, args.match_xy_um, args.match_z_um)
        prefix = f"{Path(path).stem}|{sid}"
        positions = np.array([s["pos"] / np.array([dz, px, px]) for s in sites]).reshape(-1, 3)
        clusters[prefix + "|site_positions"] = positions
        clusters[prefix + "|raw_traces_A"] = np.array([
            L.cluster_intensity(raw_registered[t, ca], positions) for t in range(T)])
        clusters[prefix + "|raw_traces_B"] = np.array([
            L.cluster_intensity(raw_registered[t, cb], positions) for t in range(T)])
        clusters[prefix + "|times_s"] = acq.times_s
        # keep the raw detections + local intensities so that onset order can be
        # tested downstream without re-running detection
        for t in range(T):
            for lbl, pts, c in ((na, ptsA[t], ca), (nb, ptsB[t], cb)):
                key = f"{Path(path).stem}|{sid}|{lbl}|t{t}"
                clusters[key] = np.asarray(pts, np.float32)
                if len(pts):
                    clusters[key + "|IA"] = L.cluster_intensity(reg[t, ca], pts)
                    clusters[key + "|IB"] = L.cluster_intensity(reg[t, cb], pts)
        clusters[f"{Path(path).stem}|{sid}|meta"] = np.array(
            [acq.px_um, dz, acq.dt_s, T, S["length_um"]], np.float32)
        for t in range(T):
            for lbl, c in ((na, ca), (nb, cb)):
                clusters[f"{Path(path).stem}|{sid}|{lbl}|thr{t}"] = np.array(
                    [THR[t, c]], np.float32)

        # persistence
        pairsets = [{(i, j) for i, j, _ in
                     L.match_clusters(ptsA[t], ptsB[t], px, dz,
                                      args.match_xy_um, args.match_z_um)}
                    for t in range(T)]
        for lbl, tracks, which in ((nb, L.link_tracks(ptsB, px, dz), 1),
                                   (na, L.link_tracks(ptsA, px, dz), 0)):
            for tid, tr in enumerate(tracks):
                if len(tr) < 2:
                    continue
                fr = sorted(tr)
                nco = sum(1 for t in fr if any(tr[t] == p[which] for p in pairsets[t]))
                trk.append(dict(
                    file=Path(path).name, structure=sid, structure_class=S["cls"],
                    channel=lbl, track=tid, first_frame=fr[0], last_frame=fr[-1],
                    n_frames=len(fr), span_frames=fr[-1] - fr[0] + 1,
                    persistence_min=(acq.times_s[fr[-1]] - acq.times_s[fr[0]]) / 60,
                    left_censored=fr[0] == 0, right_censored=fr[-1] == T - 1,
                    coloc_fraction=nco / len(fr), is_coloc=nco / len(fr) >= 0.5))

        # kymographs + displacement
        paths, _ = L.branch_paths(m2, min_len_px=to_px(args.min_branch_um))
        for bi, bp in enumerate(paths, start=1):
            sm = L.smooth_path(bp["pix"])
            ext = L.extend_path(sm, to_px(args.extend_um), shape=(Y, X))
            rs, s_px = L.resample(ext, 0.5)
            s_um = s_px * px
            kA = np.array([L.sample_along(mip[t, ca], rs) for t in range(T)])
            kB = np.array([L.sample_along(mip[t, cb], rs) for t in range(T)])
            key = f"{Path(path).stem}|{sid}|B{bi}"
            kymo[f"{key}|{na}"] = kA
            kymo[f"{key}|{nb}"] = kB
            kymo[f"{key}|s_um"] = s_um
            kymo[f"{key}|path"] = sm
            kymo[f"{key}|times_s"] = acq.times_s

            ds = float(np.median(np.diff(s_um)))
            d = sm[-1] - sm[0]
            ang = float(np.degrees(np.arctan2(d[0], d[1])))
            for t in range(T - 1):
                dA, rA = L.xcorr_shift(kA[t], kA[t + 1], ds)
                dB, rB = L.xcorr_shift(kB[t], kB[t + 1], ds)
                loA0, hiA0 = L.split_scales(kA[t], ds)
                loA1, hiA1 = L.split_scales(kA[t + 1], ds)
                loB0, hiB0 = L.split_scales(kB[t], ds)
                loB1, hiB1 = L.split_scales(kB[t + 1], ds)
                disp.append(dict(
                    file=Path(path).name, structure=sid, structure_class=S["cls"],
                    branch=f"B{bi}", frame=t, branch_angle_deg=ang,
                    disp_A_um=dA, r_A=rA, disp_B_um=dB, r_B=rB,
                    dt_s=acq.times_s[t + 1] - acq.times_s[t],
                    disp_A_um_per_min=dA / ((acq.times_s[t + 1] - acq.times_s[t]) / 60),
                    disp_B_um_per_min=dB / ((acq.times_s[t + 1] - acq.times_s[t]) / 60),
                    disp_A_shaft_um=L.xcorr_shift(loA0, loA1, ds)[0],
                    disp_A_cluster_um=L.xcorr_shift(hiA0, hiA1, ds)[0],
                    disp_B_shaft_um=L.xcorr_shift(loB0, loB1, ds)[0],
                    disp_B_cluster_um=L.xcorr_shift(hiB0, hiB1, ds)[0]))
        dA = 10 * np.mean([len(p) for p in ptsA]) / S['length_um']
        dB = 10 * np.mean([len(p) for p in ptsB]) / S['length_um']
        flag = ""
        if max(dA, dB) > args.max_plausible_density:
            flag = (f"  <-- OVER-DETECTION: >{args.max_plausible_density:.0f} clusters "
                    f"per 10 um exceeds the user-selected QC limit; inspect detections, "
                    f"sampling and background before interpreting cluster counts")
        print(f"    {sid}: {S['length_um']:.0f} um, {len(paths)} branches, "
              f"clusters/frame {np.mean([len(p) for p in ptsA]):.0f}/"
              f"{np.mean([len(p) for p in ptsB]):.0f} "
              f"({dA:.1f}/{dB:.1f} per 10 um){flag}", flush=True)

    masks = {f"{Path(path).stem}|bleach_factors": fac,
             f"{Path(path).stem}|valid": vm,
             f"{Path(path).stem}|excluded_blobs": blobs,
             f"{Path(path).stem}|tmean_A": tavg[ca],
             f"{Path(path).stem}|tmean_B": tavg[cb]}
    for sid, S in structures.items():
        masks[f"{Path(path).stem}|{sid}|mask"] = S["mask"]
        masks[f"{Path(path).stem}|{sid}|class"] = np.array([S["cls"]])

    shape_rows = [{**s, "file": Path(path).name} for s in shape_log]
    return coloc, trk, disp, kymo, masks, shape_rows, clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default=None)
    ap.add_argument("--channels", nargs=2, type=int, default=[0, 1])
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--structure", choices=["tubular", "any"], default="tubular")
    # segmentation -- all lengths in micrometres, converted to pixels per file
    ap.add_argument("--tube-sigmas-um", nargs="+", type=float,
                    default=[0.6, 1.0, 1.5, 2.1])
    ap.add_argument("--seed-pct", type=float, default=99.2)
    ap.add_argument("--grow-pct", type=float, default=95.5)
    ap.add_argument("--raw-pct", type=float, default=85.0)
    ap.add_argument("--min-skeleton-um", type=float, default=100.0)
    ap.add_argument("--min-aspect", type=float, default=25.0,
                    help="skeleton length / mean local diameter; separates tubes from blobs")
    ap.add_argument("--max-solidity", type=float, default=0.95,
                    help="lenient by design: solidity fails for straight unbranched "
                         "structures, aspect ratio is the real discriminator")
    ap.add_argument("--blob-open-um", type=float, default=3.75,
                    help="radius of the disk that must fit inside a blob but not "
                         "inside the structure of interest")
    ap.add_argument("--min-structure-um", type=float, default=30.0)
    ap.add_argument("--ratio-split", type=float, default=None)
    ap.add_argument("--max-channel-offset-px", type=float, default=0.5)
    ap.add_argument('--bleach-correct', action='store_true',
                    help='Explicit multiplicative correction; inspect recorded factors and raw signal first.')
    ap.add_argument('--channel-shift-px', nargs=2, type=float, metavar=('DY','DX'),
                    help='Independently calibrated shift to apply to B, in pixel units (row, column).')
    # detection / matching
    ap.add_argument("--tophat-um", type=float, default=2.5)
    ap.add_argument("--bg-k", type=float, default=8.0)
    ap.add_argument("--cluster-sep-um", type=float, default=1.0)
    ap.add_argument("--match-xy-um", type=float, default=1.0)
    ap.add_argument("--match-z-um", type=float, default=2.5)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--max-plausible-density", type=float, default=20.0,
                    help="clusters per 10 um above which detection is flagged as noise")
    # paths
    ap.add_argument("--min-branch-um", type=float, default=10.0)
    ap.add_argument("--extend-um", type=float, default=18.0)
    ap.add_argument("--roi-json", default=None,
                    help="JSON of hand-drawn ROI polygons; each ROI is analysed separately")
    ap.add_argument("--seed", type=int, default=2026)
    add_input_arguments(ap)
    args = ap.parse_args()
    if args.channel_shift_px is not None and not np.isfinite(args.channel_shift_px).all():
        ap.error('Channel shifts must be finite')
    if args.n_null < 1:
        ap.error('--n-null must be positive')
    if len(set(args.names)) != 2 or any('|' in name or not name for name in args.names):
        ap.error('Use two different nonempty names without |')
    positive = ['tophat_um', 'bg_k', 'cluster_sep_um', 'match_xy_um', 'match_z_um',
                'min_skeleton_um', 'min_structure_um', 'min_branch_um', 'min_aspect',
                'blob_open_um', 'max_channel_offset_px', 'max_plausible_density']
    if any(not np.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in positive):
        ap.error('Distances, scales, and detection parameters must be finite and positive')
    if any(not np.isfinite(v) or v <= 0 for v in args.tube_sigmas_um) or not np.isfinite(args.extend_um) or args.extend_um < 0:
        ap.error('Tube scales must be positive; extension must be nonnegative')
    if not 0 <= args.raw_pct <= 100 or not 0 <= args.grow_pct <= args.seed_pct <= 100:
        ap.error('Percentiles must be between 0 and 100, with grow <= seed')
    if not np.isfinite(args.max_solidity) or not 0 < args.max_solidity <= 1:
        ap.error('--max-solidity must be in (0, 1]')
    if args.ratio_split is not None and not np.isfinite(args.ratio_split):
        ap.error('--ratio-split must be finite')

    p = Path(args.input)
    files = collect_inputs(p, args.glob)
    out = Path(args.out) / "results"
    if out.exists() and any(out.iterdir()):
        raise ValueError('Results directory is not empty; choose a new --out directory.')
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    coloc, trk, disp, shape_rows = [], [], [], []
    kymo, masks, clusters = {}, {}, {}
    qc_rows, acquisition_records = [], []
    for f in files:
        row, acq = audit_one(f, *args.channels, **input_options(args))
        qc_rows.append(row)
        acquisition_records.append(dict(file=f.name, axes=acq.source_axes,
            shape=acq.shape, times_s=acq.times_s.tolist(), metadata_sources=acq.metadata_sources, notes=acq.notes))
        r = analyse_file(f, args, rng)
        if r is None:
            continue
        coloc += r[0]; trk += r[1]; disp += r[2]
        kymo.update(r[3]); masks.update(r[4]); shape_rows += r[5]
        clusters.update(r[6])

    if not coloc:
        raise ValueError('No analyzable structures were found; no statistical result was generated.')
    pd.DataFrame(qc_rows).to_csv(out / 'qc_acquisition.csv', index=False)
    (out / 'acquisition.json').write_text(json.dumps(acquisition_records, indent=2), encoding='utf-8')
    pd.DataFrame(coloc).to_csv(out / "colocalisation.csv", index=False)
    pd.DataFrame(trk, columns=['file','structure','structure_class','channel','track',
        'first_frame','last_frame','n_frames','span_frames','persistence_min',
        'left_censored','right_censored','coloc_fraction','is_coloc']).to_csv(out / 'tracks.csv', index=False)
    pd.DataFrame(disp, columns=['file','structure','structure_class','branch','frame',
        'branch_angle_deg','disp_A_um','r_A','disp_B_um','r_B','dt_s',
        'disp_A_um_per_min','disp_B_um_per_min','disp_A_shaft_um','disp_A_cluster_um',
        'disp_B_shaft_um','disp_B_cluster_um']).to_csv(out / 'displacement.csv', index=False)
    pd.DataFrame(shape_rows, columns=sorted({k for row in shape_rows for k in row}) or ['file','status']).to_csv(out / "structures.csv", index=False)
    np.savez_compressed(out / "kymographs.npz", **kymo)
    np.savez_compressed(out / "masks.npz", **masks)
    np.savez_compressed(out / "clusters.npz", **clusters)
    (out / "params.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
