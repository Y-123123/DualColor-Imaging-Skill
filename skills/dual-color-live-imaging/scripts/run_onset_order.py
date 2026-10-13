"""Descriptive onset order with per-frame count matching and all-frame raw traces.

Sites are detection-conditioned and nested within recordings. No per-site
inferential p values or causal recruitment verdicts are produced.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def build_sites(ptsA, ptsB, px, dz, max_xy, max_z):
    """Group detections from both channels and all frames into spatial sites.

    A site is a location; it records, per channel, the frames at which a cluster was
    detected there and the intensity of both channels at that detection.
    """
    sites = []          # each: dict(pos, A={frame: idx}, B={frame: idx})
    order = [(t, "A", i, p) for t, P in enumerate(ptsA) for i, p in enumerate(P)]
    order += [(t, "B", i, p) for t, P in enumerate(ptsB) for i, p in enumerate(P)]
    order.sort(key=lambda r: r[0])
    for t, ch, i, p in order:
        phys = np.array([p[0] * dz, p[1] * px, p[2] * px])
        best, bd = None, np.inf
        for s in sites:
            d_xy = np.hypot(*(s["pos"][1:] - phys[1:]))
            d_z = abs(s["pos"][0] - phys[0])
            if d_xy <= max_xy and d_z <= max_z and d_xy < bd:
                best, bd = s, d_xy
        if best is None:
            sites.append(dict(pos=phys, A={}, B={}, n=1))
        else:
            best[ch].setdefault(t, i)
            best["pos"] = (best["pos"] * best["n"] + phys) / (best["n"] + 1)
            best["n"] += 1
            continue
        sites[-1][ch][t] = i
    return sites



def onset_table(sites):
    rows = []
    for index, site in enumerate(sites):
        aa, bb = sorted(site['A']), sorted(site['B'])
        if aa and bb:
            rows.append(dict(site=index, onset_A=aa[0], onset_B=bb[0],
                             delta_frames=bb[0] - aa[0],
                             left_censored=aa[0] == 0 or bb[0] == 0,
                             y_um=site['pos'][1], x_um=site['pos'][2]))
    return pd.DataFrame(rows, columns=['site','onset_A','onset_B','delta_frames',
                                      'left_censored','y_um','x_um'])


def rank_matched_points(pts, intensity, keep_n):
    if keep_n < 0 or len(intensity) != len(pts):
        raise ValueError('Invalid intensity/count matching inputs.')
    indices = np.argsort(intensity, kind='stable')[::-1][:keep_n]
    return np.asarray(pts)[indices]


def halfmax_onset(trace):
    trace = np.asarray(trace, dtype=float)
    if not np.isfinite(trace).all() or trace.size < 2 or np.ptp(trace) <= 0:
        return np.nan
    threshold = trace.min() + .5 * np.ptp(trace)
    hits = np.flatnonzero(trace >= threshold)
    return float(hits[0]) if len(hits) else np.nan


def describe(table, label, times):
    usable = table.loc[table.left_censored.eq(False)]
    frames = usable.delta_frames.to_numpy(dtype=float)
    seconds = np.array([times[int(b)] - times[int(a)]
                        for a, b in zip(usable.onset_A, usable.onset_B)])
    return dict(method=label, n_sites=len(frames), n_left_censored=int(table.left_censored.sum()),
                median_delta_frames=float(np.median(frames)) if len(frames) else np.nan,
                median_delta_s=float(np.median(seconds)) if len(seconds) else np.nan,
                q25_delta_s=float(np.percentile(seconds,25)) if len(seconds) else np.nan,
                q75_delta_s=float(np.percentile(seconds,75)) if len(seconds) else np.nan,
                A_first=int(np.sum(frames>0)), B_first=int(np.sum(frames<0)), same_frame=int(np.sum(frames==0)),
                interpretation='descriptive; sites are nested; sequence does not establish recruitment')


def load_clusters(npz, prefix, label, T):
    pts, intensities = [], []
    suffix = '|IA' if label == 'A' else '|IB'
    # Archive channel labels are supplied separately by main.
    for t in range(T):
        key = prefix + f'|t{t}'
        points = np.asarray(npz[key], dtype=float)
        intensity = np.asarray(npz[key+suffix]) if key+suffix in npz.files else np.zeros(len(points))
        pts.append(points); intensities.append(intensity)
    return pts, intensities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    results = Path(args.results)
    parameters = json.loads((results/'params.json').read_text(encoding='utf-8'))
    name_a, name_b = parameters['names']
    summaries, detail = [], []
    with np.load(results/'clusters.npz', allow_pickle=False) as npz:
        for meta in sorted(k for k in npz.files if k.endswith('|meta')):
            stem, structure, _ = meta.split('|')
            prefix = f'{stem}|{structure}'
            px, dz, _, T, _ = map(float, npz[meta]); T = int(T)
            times = np.asarray(npz[prefix+'|times_s'], dtype=float)
            if len(times)!=T or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
                raise ValueError('Invalid timestamps in result archive.')
            points_a, ia = load_clusters(npz, prefix+'|'+name_a, 'A', T)
            points_b, ib = load_clusters(npz, prefix+'|'+name_b, 'B', T)
            sites = build_sites(points_a, points_b, px, dz,
                                parameters['match_xy_um'], parameters['match_z_um'])
            observed = onset_table(sites)
            keep = [min(len(a),len(b)) for a,b in zip(points_a,points_b)]
            matched = onset_table(build_sites(
                [rank_matched_points(p,i,k) for p,i,k in zip(points_a,ia,keep)],
                [rank_matched_points(p,i,k) for p,i,k in zip(points_b,ib,keep)],
                px,dz,parameters['match_xy_um'],parameters['match_z_um']))
            traces_a = np.asarray(npz[prefix+'|raw_traces_A'])
            traces_b = np.asarray(npz[prefix+'|raw_traces_B'])
            if traces_a.shape!=(T,len(sites)) or traces_b.shape!=traces_a.shape:
                raise ValueError('Raw all-frame trace schema mismatch; rerun the current pipeline.')
            raw_rows = []
            for index, site in enumerate(sites):
                if not site['A'] or not site['B']:
                    continue
                a,b = halfmax_onset(traces_a[:,index]), halfmax_onset(traces_b[:,index])
                if np.isfinite(a) and np.isfinite(b):
                    raw_rows.append(dict(site=index,onset_A=a,onset_B=b,delta_frames=b-a,
                        left_censored=a==0 or b==0 or 0 in site['A'] or 0 in site['B'],
                        y_um=site['pos'][1],x_um=site['pos'][2]))
            raw = pd.DataFrame(raw_rows, columns=observed.columns)
            for label, table in [('detected',observed),('per-frame-count-matched',matched),
                                 ('all-frame-raw-half-range',raw)]:
                record = describe(table,label,times)
                record.update(file=stem,structure=structure)
                summaries.append(record)
                table=table.copy(); table['file']=stem; table['structure']=structure; table['method']=label
                detail.append(table)
    out=Path(args.out) if args.out else results
    out.mkdir(parents=True,exist_ok=True)
    if any((out/name).exists() for name in ('onset_order_summary.csv','onset_order_sites.csv')):
        raise ValueError('Onset output exists; choose a new --out directory.')
    pd.DataFrame(summaries).to_csv(out/'onset_order_summary.csv',index=False)
    pd.concat(detail,ignore_index=True).to_csv(out/'onset_order_sites.csv',index=False)
    print('Wrote descriptive onset summaries. Review brightness, censoring and site-selection bias before interpretation.')


if __name__=='__main__':
    main()
