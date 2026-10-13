"""
dual-color-live-imaging: shared library.

Dataset-agnostic implementation of the method described in SKILL.md.  Everything
that depends on the experiment (pixel size, interval, channel identity) is read
from the file metadata or passed in explicitly -- nothing is hard-coded.

Axis convention throughout: (T, C, Z, Y, X).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import networkx as nx
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import (skeletonize, remove_small_objects,
                                remove_small_holes, closing, opening, disk)




# --------------------------------------------------------------------------- #
# metadata + loading
# --------------------------------------------------------------------------- #
from image_io import Acquisition, read_image

def read_metadata(path, **options):
    return read_image(path, **options)[0]

def load(path, **options):
    return read_image(path, **options)[1]


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
def estimate_drift(data: np.ndarray, upsample: int = 20) -> np.ndarray:
    """Rigid XY shift of each frame relative to frame 0, from the summed-channel MIP."""
    mip = data.max(axis=2).sum(axis=1)
    ref = mip[0]
    if not np.isfinite(data).all() or ref.std() < 1e-9:
        raise ValueError('No spatial contrast for drift registration.')
    if any(m.std() < 1e-9 for m in mip):
        raise ValueError('A frame has no spatial contrast for drift registration.')
    return np.array([phase_cross_correlation(ref, m, upsample_factor=upsample,
                                             normalization=None)[0] for m in mip])


def apply_drift(data: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Apply the frame shift identically to every channel and z-plane, so that the
    channels are never displaced relative to one another."""
    out = np.empty_like(data)
    for t in range(data.shape[0]):
        for c in range(data.shape[1]):
            for z in range(data.shape[2]):
                out[t, c, z] = ndi.shift(data[t, c, z], shifts[t], order=1,
                                         mode="constant", cval=0.0)
    return out


def valid_mask(shape_yx, shifts, pad=3) -> np.ndarray:
    m = np.zeros(shape_yx, bool)
    y0 = int(np.ceil(max(0, shifts[:, 0].max()))) + pad
    y1 = shape_yx[0] + int(np.floor(min(0, shifts[:, 0].min()))) - pad
    x0 = int(np.ceil(max(0, shifts[:, 1].max()))) + pad
    x1 = shape_yx[1] + int(np.floor(min(0, shifts[:, 1].min()))) - pad
    if y1 <= y0 or x1 <= x0:
        raise ValueError('Registration leaves no common field of view.')
    m[y0:y1, x0:x1] = True
    return m


def bleach_factors(data, mask=None, pct=95.0, strict=True):
    """Multiplicative per-frame, per-channel correction that flattens the top-`pct`
    structural signal.  Returns (corrected, factors[T, C], raw_signal[T, C])."""
    T, C = data.shape[:2]
    out, fac, raw = data.copy(), np.ones((T, C)), np.zeros((T, C))
    for c in range(C):
        ref = None
        for t in range(T):
            v = data[t, c]
            v = v[:, mask] if mask is not None else v
            if not v.size:
                raise ValueError('No valid pixels for bleaching estimation.')
            s = v[v >= np.percentile(v, pct)].mean()
            if not np.isfinite(s):
                raise ValueError('Non-finite signal for bleaching estimation.')
            raw[t, c] = s
            if t == 0:
                ref = s
            if s <= 0 or ref <= 0:
                if strict:
                    raise ValueError('Nonpositive reference signal; bleaching correction is undefined.')
                fac[t, c] = np.nan
                continue
            fac[t, c] = ref / max(s, 1e-9)
            out[t, c] *= fac[t, c]
    return out, fac, raw


def tophat(data, radius_px=4, presmooth=0.8):
    se = disk(radius_px)
    out = np.empty_like(data)
    for t in range(data.shape[0]):
        for c in range(data.shape[1]):
            for z in range(data.shape[2]):
                out[t, c, z] = ndi.white_tophat(
                    ndi.gaussian_filter(data[t, c, z], presmooth), footprint=se)
    return out


def channel_offset_um(data, ca, cb, px_um, upsample=50, n_tiles=3):
    """Chromatic / channel-registration offset, with a reliability check.

    Cross-correlating a field dominated by a single near-1D structure is
    unconstrained along that structure's axis and can return a large spurious shift.
    The offset is therefore also estimated on n_tiles x n_tiles sub-tiles; a large
    spread across tiles means the whole-field number cannot be trusted.
    Returns (offset_um, shift_yx, spread_px).
    """
    mip = data.max(axis=2)
    A, B = mip[:, ca].mean(0), mip[:, cb].mean(0)
    if A.std() < 1e-9 or B.std() < 1e-9:
        return float('nan'), np.full(2, np.nan), float('nan')
    s, _, _ = phase_cross_correlation(A, B, upsample_factor=upsample,
                                      normalization=None)
    Y, X = A.shape
    hy, hx = Y // n_tiles, X // n_tiles
    tiles = []
    for i in range(n_tiles):
        for j in range(n_tiles):
            a = A[i * hy:(i + 1) * hy, j * hx:(j + 1) * hx]
            b = B[i * hy:(i + 1) * hy, j * hx:(j + 1) * hx]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            if np.percentile(a, 99.5) <= np.percentile(a, 50) * 1.2:
                continue                      # tile carries no structure
            t, _, _ = phase_cross_correlation(a, b, upsample_factor=upsample,
                                              normalization=None)
            tiles.append(t)
    spread = float(np.median(np.hypot(*(np.array(tiles) - s).T))) if len(tiles) >= 3 \
        else float("nan")
    return float(np.hypot(*s) * px_um), s, spread


def apply_channel_offset(data, cb, shift_yx):
    """Shift channel `cb` onto channel `ca` by `shift_yx` (as returned above)."""
    out = data.copy()
    for t in range(data.shape[0]):
        for z in range(data.shape[2]):
            out[t, cb, z] = ndi.shift(data[t, cb, z], shift_yx, order=1,
                                      mode="constant", cval=0.0)
    return out


def axial_centroid(data) -> np.ndarray:
    """Intensity-weighted z centroid per frame; focus and biology can both affect it."""
    out = []
    for t in range(data.shape[0]):
        w = data[t].sum(axis=(0, 2, 3))
        w = np.clip(w - w.min(), 0, None)
        out.append((w * np.arange(len(w))).sum() / max(w.sum(), 1e-9))
    return np.array(out)


# --------------------------------------------------------------------------- #
# structure segmentation + shape filtering
# --------------------------------------------------------------------------- #
def load_rois(json_path, shape_yx, file_stem=None):
    """Read hand-drawn ROI polygons and rasterise them to full-frame masks.

    The JSON may describe polygons either in full-frame coordinates or in the
    coordinates of a crop (give `crop_offset_row` / `crop_offset_col`), which is what
    you get when the ROIs were drawn on a published figure panel.  Structure:

        {"crop_offset_row": 0, "crop_offset_col": 0,
         "rois": {"region_1": [[x, y], ...], "region_2": [[x, y], ...]}}

    A top-level "per_file" mapping of file stem -> the same object is also accepted.
    """
    import json
    from matplotlib.path import Path as _MplPath
    cfg = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if "per_file" in cfg:
        if file_stem not in cfg["per_file"]:
            raise ValueError('ROI per_file mapping must contain the exact input stem.')
        cfg = cfg['per_file'][file_stem]
    r0 = int(cfg.get("crop_offset_row", 0))
    c0 = int(cfg.get("crop_offset_col", 0))
    H, W = shape_yx
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.column_stack([xx.ravel() - c0, yy.ravel() - r0])
    out = {}
    for name, poly in cfg["rois"].items():
        if not name or '|' in name:
            raise ValueError('ROI names must be nonempty and cannot contain |.')
        p = np.asarray(poly, float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError('Each ROI needs at least three finite (x, y) vertices.')
        m = _MplPath(np.vstack([p, p[:1]])).contains_points(pts).reshape(H, W)
        if not m.any():
            raise ValueError('An ROI contains no pixels in the selected field.')
        out[name] = m
    if not out:
        raise ValueError('ROI file must define at least one polygon.')
    return out


def tubeness(img, sigmas=(1.0, 1.6, 2.4, 3.4)):
    v = sato(img.astype(np.float32), sigmas=sigmas, black_ridges=False, mode="reflect")
    return v / max(v.max(), 1e-12)


def segment_tubular(tavg, valid=None, seed_pct=99.2, grow_pct=95.5, raw_pct=85.0,
                    sigmas=(1.0, 1.6, 2.4, 3.4), min_size=60):
    """Hysteresis threshold on a multiscale tubeness map -- bright ridge cores seed the
    mask and grow along the much fainter distal branches."""
    tb = tubeness(tavg, sigmas)
    if valid is not None:
        tb = tb * valid
    pool = tb[tb > 0]
    if not pool.size:
        raise ValueError('No positive tubular response for segmentation.')
    m = apply_hysteresis_threshold(tb, np.percentile(pool, grow_pct),
                                   np.percentile(pool, seed_pct))
    raw = tavg * valid if valid is not None else tavg
    m &= raw > np.percentile(raw[raw > 0], raw_pct)
    m = closing(m, disk(2))
    m = remove_small_holes(m, max_size=63)
    return remove_small_objects(m, max_size=min_size - 1)


def strip_blobs(mask, open_radius_px=6, min_core_px=40, rim_px=2, bridge_px=2):
    """Excise compact objects, including any fused to a tubular structure.

    A disk of `open_radius_px` fits only inside soma-sized objects, not inside a
    neurite.  Surviving cores are removed with a collar, then a small closing
    reconnects any branch that was severed.  Returns (thin_mask, removed_mask).
    """
    core = opening(mask, disk(open_radius_px))
    if core.any():
        lab = label(core)
        keep = np.zeros_like(core)
        for i in range(1, lab.max() + 1):
            c = lab == i
            if c.sum() >= min_core_px:
                keep |= c
        core = keep
    if not core.any():
        return mask.copy(), np.zeros_like(mask)
    removed = ndi.binary_dilation(core, disk(rim_px)) & mask
    thin = mask & ~removed
    if bridge_px:
        thin = closing(thin, disk(bridge_px)) & mask & ~core
    return thin, removed


def shape_stats(comp, dt=None):
    """Shape descriptors of one connected component.

    `aspect` = skeleton length / mean local diameter is the discriminator that
    helps distinguish elongated from compact objects. Solidity and aspect thresholds
    are experiment-dependent geometric heuristics, not cell-type identification.
    """
    sk = int(skeletonize(comp).sum())
    rp = regionprops(comp.astype(np.uint8))[0]
    st = dict(skel=sk, area=int(rp.area), solidity=float(rp.solidity),
              skel_per_area=sk / max(rp.area, 1), eccentricity=float(rp.eccentricity),
              aspect=float("nan"))
    if dt is not None and comp.any():
        r = dt[comp]
        st["r_mean"] = float(r.mean())
        st["r_p90"] = float(np.percentile(r, 90))
        st["r_max"] = float(r.max())
        st["aspect"] = float(sk / (2 * max(r.mean(), 1e-9)))
    return st


def tubular_components(mask, min_skel_px=160, min_aspect=25.0, max_solidity=0.95,
                       open_radius_px=6, verbose=False):
    """Two-stage filter: drop whole compact objects, then excise fused blobs.

    Stage 1 keeps objects that are long AND elongated (see `shape_stats`).  Stage 2
    excises soma-sized objects welded onto a retained structure.  Fragments left by
    stage 2 are re-grouped with their parent so a structure is analysed whole.

    Returns (components, skeleton_lengths, removed_mask, log).
    """
    log = []
    dt_all = ndi.distance_transform_edt(mask)
    lab0 = label(mask)
    pre = np.zeros_like(mask)
    for i in range(1, lab0.max() + 1):
        comp = lab0 == i
        st = shape_stats(comp, dt_all)
        ok = (st["skel"] >= min_skel_px and st["aspect"] >= min_aspect
              and st["solidity"] <= max_solidity)
        st.update(id=int(i), stage="object", kept=bool(ok))
        log.append(st)
        if ok:
            pre |= comp
        elif verbose and st["skel"] >= 10:
            why = ("short" if st["skel"] < min_skel_px else
                   "compact" if st["aspect"] < min_aspect else "solid")
            print(f"    object {i}: skel={st['skel']} aspect={st['aspect']:.1f} "
                  f"solidity={st['solidity']:.2f} -> dropped ({why})")

    if not pre.any():
        return [], [], np.zeros_like(mask), log

    thin, removed = strip_blobs(pre, open_radius_px=open_radius_px)
    thin = remove_small_objects(thin, max_size=24)

    dt = ndi.distance_transform_edt(pre)
    parent, lab = label(pre), label(thin)
    groups = {}
    for i in range(1, lab.max() + 1):
        c = lab == i
        p = int(np.bincount(parent[c]).argmax())
        groups.setdefault(p, np.zeros_like(thin))
        groups[p] |= c

    out = []
    for p, comp in groups.items():
        st = shape_stats(comp, dt)
        ok = (st["skel"] >= 0.5 * min_skel_px and st["aspect"] >= min_aspect)
        st.update(id=int(p), stage="group", kept=bool(ok))
        log.append(st)
        if ok:
            out.append((st["skel"], comp))
        elif verbose:
            print(f"    group {p}: skel={st['skel']} aspect={st['aspect']:.1f} "
                  f"-> dropped after blob excision")
    out.sort(key=lambda t: -t[0])
    return [c for _, c in out], [n for n, _ in out], removed, log


def split_by_channel_ratio(comp, imgA, imgB, split=None, halfwidth=2, min_len_px=8):
    """Optional descriptive log2(A/B) split; ratios do not establish label positivity.

    An automatic largest-gap threshold is exploratory and must be inspected.
    """
    skel, G, coords = skeleton_graph(comp)
    segs = []
    for u, v, d in G.edges(data=True):
        if d["length"] < min_len_px:
            continue
        bm = band_mask(d["path"], imgA.shape, halfwidth)
        a, b = imgA[bm].mean(), imgB[bm].mean()
        segs.append(dict(u=u, v=v, path=d["path"].astype(float),
                         length=d["length"], ratio=float(np.log2(max(a, 1e-9) / max(b, 1e-9)))))
    if not segs:
        return {"unsplit": []}, None
    r = np.sort([s["ratio"] for s in segs])
    if split is None and len(r) > 4:
        gaps = np.diff(r)
        k = int(np.argmax(gaps))
        split = (r[k] + r[k + 1]) / 2 if gaps[k] > 0.6 else None
    if split is None:
        return {"unsplit": segs}, None
    return ({"ratio-high": [s for s in segs if s["ratio"] >= split],
             "ratio-low": [s for s in segs if s["ratio"] < split]}, float(split))


# --------------------------------------------------------------------------- #
# skeleton graph + branch paths
# --------------------------------------------------------------------------- #
def band_mask(path_yx, shape, halfwidth=2):
    m = np.zeros(shape, bool)
    p = np.asarray(path_yx)
    m[np.clip(p[:, 0].astype(int), 0, shape[0] - 1),
      np.clip(p[:, 1].astype(int), 0, shape[1] - 1)] = True
    return ndi.binary_dilation(m, ndi.generate_binary_structure(2, 2),
                               iterations=halfwidth)


_OFF = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_graph(mask):
    """Skeletonise; return (skel, MultiGraph with 'path'/'length' on edges, node coords)."""
    skel = skeletonize(mask)
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8),
                      mode="constant") * skel - skel
    node_mask = skel & ((nb == 1) | (nb >= 3))
    nlab, n = ndi.label(node_mask, structure=np.ones((3, 3)))
    coords = (np.array(ndi.center_of_mass(node_mask, nlab, range(1, n + 1)))
              if n else np.zeros((0, 2)))
    G = nx.MultiGraph()
    G.add_nodes_from(range(1, n + 1))
    skelset = set(map(tuple, np.argwhere(skel)))
    seen = set()
    for i in range(1, n + 1):
        for s in map(tuple, np.argwhere(nlab == i)):
            for d in _OFF:
                nxt = (s[0] + d[0], s[1] + d[1])
                if nxt not in skelset or nlab[nxt] == i or (s, nxt) in seen:
                    continue
                path, prev, cur = [s, nxt], s, nxt
                seen.add((s, nxt))
                while nlab[cur] == 0:
                    cand = [(cur[0] + dd[0], cur[1] + dd[1]) for dd in _OFF]
                    cand = [c for c in cand
                            if c in skelset and c != prev and c not in path[:-1]]
                    if not cand:
                        break
                    cand.sort(key=lambda c: abs(c[0] - cur[0]) + abs(c[1] - cur[1]))
                    prev, cur = cur, cand[0]
                    path.append(cur)
                    seen.add((prev, cur))
                if nlab[cur] and nlab[cur] != i:
                    # block the reverse traversal of this edge, otherwise the walk
                    # started from the far node adds it a second time and every
                    # length is doubled
                    if len(path) >= 2:
                        seen.add((path[-1], path[-2]))
                    p = np.array(path)
                    G.add_edge(i, int(nlab[cur]), path=p, length=polyline_len(p))
    return skel, G, coords


def polyline_len(path):
    d = np.diff(np.asarray(path, float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def branch_paths(mask, min_len_px=16):
    """Ordered root-to-tip pixel paths.  Root = node in the thickest part of the mask."""
    skel, G, coords = skeleton_graph(mask)
    if G.number_of_edges() == 0 or not len(coords):
        return [], None
    dt = ndi.distance_transform_edt(mask)
    root = int(np.argmax([dt[int(round(c[0])), int(round(c[1]))] for c in coords])) + 1
    H = nx.Graph()
    for u, v, d in G.subgraph(nx.node_connected_component(G, root)).edges(data=True):
        if not H.has_edge(u, v) or H[u][v]["length"] > d["length"]:
            H.add_edge(u, v, **d)
    out = []
    for tip in [nd for nd in H.nodes if H.degree(nd) == 1 and nd != root]:
        try:
            nodes = nx.shortest_path(H, root, tip, weight="length")
        except nx.NetworkXNoPath:
            continue
        pix = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            p = H[a][b]["path"]
            reference = np.asarray(pix[-1]) if pix else coords[a - 1]
            if np.hypot(*(p[0] - reference)) > np.hypot(*(p[-1] - reference)):
                p = p[::-1]
            pix.extend(p.tolist() if not pix else p.tolist()[1:])
        pix = np.array(pix, float)
        if polyline_len(pix) >= min_len_px:
            out.append(dict(pix=pix, length_px=polyline_len(pix), tip=tip, root=root))
    out.sort(key=lambda d: -d["length_px"])
    return out, root


def smooth_path(p, win=7):
    p = np.asarray(p, float)
    if len(p) < win:
        return p
    k = np.ones(win) / win
    pad = win // 2
    return np.column_stack([
        np.convolve(np.pad(p[:, i], (pad, pad), mode="edge"), k, "valid")[:len(p)]
        for i in (0, 1)])


def extend_path(p, extra_px, shape=None):
    """Extrapolate the distal end so growth beyond the time-averaged skeleton is visible."""
    p = np.asarray(p, float)
    n = min(12, len(p) - 1)
    d = p[-1] - p[-1 - n]
    d = d / max(np.hypot(*d), 1e-9)
    ext = p[-1] + np.outer(np.arange(1, int(extra_px) + 1), d)
    if shape is not None:
        ext = ext[(ext[:, 0] >= 0) & (ext[:, 0] < shape[0] - 1) &
                  (ext[:, 1] >= 0) & (ext[:, 1] < shape[1] - 1)]
    return np.vstack([p, ext])


def resample(path, spacing_px=0.5):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    s = np.linspace(0, d[-1], max(int(d[-1] / spacing_px) + 1, 2))
    return np.column_stack([np.interp(s, d, path[:, 0]),
                            np.interp(s, d, path[:, 1])]), s


def sample_along(img, path, halfwidth=2, reduce="max"):
    """Sample along a path, taking max (or mean) across a perpendicular band -- a
    single-pixel line is dominated by noise."""
    p = np.asarray(path, float)
    g = np.gradient(p, axis=0)
    nrm = np.column_stack([-g[:, 1], g[:, 0]])
    nrm /= np.maximum(np.hypot(nrm[:, 0], nrm[:, 1])[:, None], 1e-9)
    stack = np.array([ndi.map_coordinates(img, (p + o * nrm).T, order=1, mode="nearest")
                      for o in range(-halfwidth, halfwidth + 1)])
    return stack.max(axis=0) if reduce == "max" else stack.mean(axis=0)


# --------------------------------------------------------------------------- #
# 3-D cluster detection, matching, null model
# --------------------------------------------------------------------------- #
def background_threshold(vol_hat, bg_mask2d, k=8.0):
    """Threshold from candidate background of the SAME frame and channel.

    A mask-relative percentile would manufacture clusters in a channel carrying only
    noise, which is how a negative control silently turns positive.
    """
    pool = vol_hat[:, bg_mask2d].ravel()
    if pool.size < 20 or not np.isfinite(pool).all():
        raise ValueError('Insufficient valid cell-free background; revise ROIs or acquisition.')
    med = np.median(pool)
    return float(med + k * 1.4826 * np.median(np.abs(pool - med)))


def structure_mask_3d(vol_ref, mask2d, frac=0.35, dilate_z=1):
    """Keep only the z-planes the structure occupies, so the null model is not given
    a full cylinder of empty space to scatter points into."""
    zmax = vol_ref.max(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(zmax > 0, vol_ref / zmax, 0)
    m3 = (rel >= frac) & mask2d[None]
    ys, xs = np.nonzero(mask2d)
    m3[vol_ref.argmax(axis=0)[ys, xs], ys, xs] = True
    if dilate_z:
        m3 = ndi.binary_dilation(m3, np.ones((3, 1, 1), bool), iterations=dilate_z) \
             & mask2d[None]
    return m3


def detect_clusters_3d(vol_hat, mask3d, thr, min_sep_px=2):
    if mask3d.sum() < 20:
        return np.zeros((0, 3))
    return peak_local_max(vol_hat * mask3d, min_distance=max(int(min_sep_px), 1),
                          threshold_abs=thr, labels=mask3d,
                          exclude_border=False).astype(float)


def cluster_intensity(vol, pts, rad_px=2, rad_z=1):
    """Peak intensity in a small box around each (z, y, x) detection."""
    Z, Y, X = vol.shape
    out = np.zeros(len(pts))
    for i, (z, y, x) in enumerate(np.asarray(pts, float)):
        z0, z1 = max(int(z) - rad_z, 0), min(int(z) + rad_z + 1, Z)
        y0, y1 = max(int(y) - rad_px, 0), min(int(y) + rad_px + 1, Y)
        x0, x1 = max(int(x) - rad_px, 0), min(int(x) + rad_px + 1, X)
        out[i] = vol[z0:z1, y0:y1, x0:x1].max()
    return out


def _phys(pts, px_um, dz_um):
    pts = np.asarray(pts, float)
    return (pts * np.array([dz_um, px_um, px_um])) if len(pts) else pts.reshape(0, 3)


def match_clusters(ptsA, ptsB, px_um, dz_um, max_xy_um=1.0, max_z_um=2.5):
    """Maximum-cardinality one-to-one matching, then minimum normalized distance."""
    from scipy.optimize import linear_sum_assignment
    if any(not np.isfinite(v) or v <= 0 for v in
           (px_um, dz_um, max_xy_um, max_z_um)):
        raise ValueError('Calibration and matching tolerances must be positive.')
    if not len(ptsA) or not len(ptsB):
        return []
    A, B = _phys(ptsA, px_um, dz_um), _phys(ptsB, px_um, dz_um)
    dxy = np.hypot(A[:, 1, None] - B[None, :, 1], A[:, 2, None] - B[None, :, 2])
    dz = np.abs(A[:, 0, None] - B[None, :, 0])
    valid = (dxy <= max_xy_um) & (dz <= max_z_um)
    distance = np.hypot(dxy / max_xy_um, dz / max_z_um)
    penalty = (min(len(A), len(B)) + 1) * 4.
    ii, jj = linear_sum_assignment(np.where(valid, distance, penalty))
    return [(int(i), int(j), float(dxy[i, j])) for i, j in zip(ii, jj) if valid[i, j]]


def null_match_fraction(ptsA, ptsB, mask3d, px_um, dz_um, n_iter=200, rng=None, **kw):
    """Conditional uniform-location null within a fixed structure mask.

    Keeps object counts and channel B fixed; does not preserve A's spatial clustering.
    This is one explicit null hypothesis, not a universal or superior null model.
    """
    if n_iter < 1:
        raise ValueError('n_iter must be positive.')
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = np.argwhere(mask3d)
    if not len(idx) or not len(ptsA) or not len(ptsB):
        return np.array([np.nan])
    if len(ptsA) > len(idx):
        raise ValueError('More objects than distinct mask voxels.')
    denom = min(len(ptsA), len(ptsB))
    return np.array([
        len(match_clusters(idx[rng.choice(len(idx), len(ptsA), replace=False)].astype(float),
                           ptsB, px_um, dz_um, **kw)) / denom
        for _ in range(n_iter)])


def shell_ratio(vol_mean, mask2d, m3, valid, inner=3, outer=7):
    """Structure content relative to the neuropil shell immediately around it.
    A value < 1 means the structure holds less than its own local background,
    a descriptive comparison; halos, background and expression can confound it."""
    shell = (ndi.binary_dilation(mask2d, iterations=outer)
             & ~ndi.binary_dilation(mask2d, iterations=inner) & valid)
    if not shell.any() or not m3.any():
        return float("nan")
    return float(vol_mean[m3].mean() / max(vol_mean.max(axis=0)[shell].mean(), 1e-9))


# --------------------------------------------------------------------------- #
# tracking (stationary structures only, at coarse intervals)
# --------------------------------------------------------------------------- #
def link_tracks(per_frame_pts, px_um, dz_um, max_link_um=1.5, max_gap=1):
    """Nearest-neighbour linking.  At intervals too coarse for transport this measures
    persistence of stationary structures and nothing else."""
    tracks, last = [], []
    for t, pts in enumerate(per_frame_pts):
        P = _phys(pts, px_um, dz_um)
        claimed = set()
        alive = sorted((i for i in range(len(tracks)) if t - last[i][0] <= max_gap + 1),
                       key=lambda i: -last[i][0])
        for tid in alive:
            if not len(P):
                continue
            lp = last[tid][1]
            d = np.hypot(P[:, 1] - lp[1], P[:, 2] - lp[2])
            d = np.where(np.abs(P[:, 0] - lp[0]) <= dz_um * 1.5, d, np.inf)
            if claimed:
                d[list(claimed)] = np.inf
            j = int(np.argmin(d))
            if d[j] <= max_link_um:
                tracks[tid][t] = j
                claimed.add(j)
                last[tid] = [t, P[j]]
        for j in range(len(P)):
            if j not in claimed:
                tracks.append({t: j})
                last.append([t, P[j]])
    return tracks


# --------------------------------------------------------------------------- #
# along-path displacement + discriminating controls
# --------------------------------------------------------------------------- #
def xcorr_shift(a, b, ds_um, max_lag_um=12.0):
    """Lag (um) at which profile `b` best matches `a`; positive = away from the root."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan, np.nan
    if len(a) != len(b) or ds_um <= 0 or not np.isfinite(ds_um):
        raise ValueError('Profiles must have the same length and positive spacing.')
    limit = min(int(max_lag_um / ds_um), max(len(a) - 12, 0))
    lags = np.arange(-limit, limit + 1)
    r = np.full(len(lags), np.nan)
    for i, L in enumerate(lags):
        x, y = (a[L:], b[:len(b) - L] if L else b) if L >= 0 else (a[:L], b[-L:])
        m = min(len(x), len(y))
        if m < 12:
            continue
        x, y = x[:m], y[:m]
        if x.std() > 1e-9 and y.std() > 1e-9:
            r[i] = float(np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std()))
    if np.all(np.isnan(r)):
        return np.nan, np.nan
    k = int(np.nanargmax(r))
    off = 0.0
    if 0 < k < len(r) - 1 and np.isfinite(r[k - 1]) and np.isfinite(r[k + 1]):
        den = r[k - 1] - 2 * r[k] + r[k + 1]
        off = np.clip(0.5 * (r[k - 1] - r[k + 1]) / den, -1, 1) if abs(den) > 1e-12 else 0
    return float(-(lags[k] + off) * ds_um), float(r[k])


def split_scales(profile, ds_um, cut_um=5.0):
    """Smooth 'shaft' component and punctate 'cluster' component of a profile.
    This contrast separation is descriptive; it does not identify cargo or distinguish
    tissue deformation from transport on its own."""
    lo = ndi.gaussian_filter1d(np.asarray(profile, float), cut_um / ds_um, mode="nearest")
    return lo, profile - lo
