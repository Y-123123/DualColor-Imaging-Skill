"""Explicit microscopy axes, units, metadata provenance, and CLI overrides."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

CANONICAL = 'TCZYX'
LENGTH_UNITS = {'µm': 1., 'μm': 1., 'um': 1., 'micron': 1., 'micrometer': 1.,
                'nm': .001, 'mm': 1000., 'cm': 10000., 'm': 1e6}
TIME_UNITS = {'s': 1., 'sec': 1., 'ms': .001, 'µs': 1e-6, 'μs': 1e-6,
              'us': 1e-6, 'min': 60., 'h': 3600.}


def convert_unit(value, unit, kind):
    table = LENGTH_UNITS if kind == 'length' else TIME_UNITS
    if unit not in table:
        raise ValueError(f'Unsupported {kind} unit {unit!r}; supply an explicit override.')
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f'Non-finite {kind} metadata.')
    return number * table[unit]


def normalise_axes(data, axes):
    """Reorder labeled axes; drop only explicitly singleton unsupported axes."""
    a = np.asarray(data)
    axes = axes.upper()
    if len(axes) != a.ndim or len(set(axes)) != len(axes):
        raise ValueError(f'Axes {axes!r} do not uniquely label shape {a.shape}.')
    if 'Y' not in axes or 'X' not in axes:
        raise ValueError('Input axes must include Y and X.')
    for index in reversed(range(len(axes))):
        if axes[index] not in CANONICAL:
            if a.shape[index] != 1:
                raise ValueError(f'Unsupported non-singleton axis {axes[index]}: '
                                 'select/export one scene or use --axes for a known layout.')
            a = np.take(a, 0, axis=index)
            axes = axes[:index] + axes[index + 1:]
    present = [c for c in CANONICAL if c in axes]
    a = np.transpose(a, [axes.index(c) for c in present])
    shape = tuple(a.shape[present.index(c)] if c in present else 1 for c in CANONICAL)
    return a.reshape(shape)


@dataclass
class Acquisition:
    path: Path
    shape: tuple
    px_um: float
    dz_um: float
    times_s: np.ndarray
    bit_depth: int = 0
    objective: str = ''
    na: float = float('nan')
    immersion: str = ''
    channel_names: tuple = ()
    source_axes: str = ''
    metadata_sources: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def dt_s(self):
        return float(np.median(np.diff(self.times_s))) if len(self.times_s) > 1 else float('nan')

    @property
    def duration_min(self):
        return float((self.times_s[-1] - self.times_s[0]) / 60)

    def resolvable_velocity_um_s(self, max_px_per_frame=2.):
        # A user-selected linking heuristic, not an optical guarantee.
        return max_px_per_frame * self.px_um / self.dt_s

    def displacement_um(self, velocity_um_s):
        return velocity_um_s * self.dt_s

    def nyquist_px_um(self, emission_nm=510.):
        return emission_nm * 1e-3 / (4 * self.na) if np.isfinite(self.na) and self.na > 0 else float('nan')


def _children(element, name):
    return [e for e in element if e.tag.split('}')[-1] == name]


def _ome_metadata(xml, series, T):
    root = ET.fromstring(xml)
    images = _children(root, 'Image')
    if series >= len(images):
        raise ValueError('OME image index does not match the selected TIFF series.')
    pixels = _children(images[series], 'Pixels')[0]
    result = {}
    for axis in 'XYZ':
        key = 'PhysicalSize' + axis
        if key in pixels.attrib:
            result[axis] = convert_unit(pixels.get(key), pixels.get(key + 'Unit', 'µm'), 'length')
    if pixels.get('TimeIncrement'):
        step = convert_unit(pixels.get('TimeIncrement'), pixels.get('TimeIncrementUnit', 's'), 'time')
        result['times'] = np.arange(T, dtype=float) * step
    # Use the same reference channel and plane throughout; never average staggered
    # channel/Z acquisition times into one frame timestamp.
    planes = _children(pixels, 'Plane')
    by_time = {}
    for plane in planes:
        if plane.get('TheC', '0') == '0' and plane.get('TheZ', '0') == '0' and plane.get('DeltaT'):
            t = int(plane.get('TheT', '0'))
            value = convert_unit(plane.get('DeltaT'), plane.get('DeltaTUnit', 's'), 'time')
            if t in by_time and not np.isclose(by_time[t], value):
                raise ValueError('Conflicting OME timestamps for the same reference frame.')
            by_time[t] = value
    if len(by_time) == T and set(by_time) == set(range(T)):
        result['times'] = np.array([by_time[t] for t in range(T)])
    result['names'] = tuple(c.get('Name', '') for c in _children(pixels, 'Channel'))
    result['bits'] = int(pixels.get('SignificantBits', '0'))
    return result


def read_image(path, *, axes=None, series=0, pixel_size_um=None,
               z_step_um=None, frame_interval_s=None):
    """Read one scene/series and fail on unknown quantitative metadata."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Input file not found: {path.name}')
    if series < 0:
        raise ValueError('--series must be nonnegative.')
    meta, sources, notes = {}, {}, []
    if path.suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as tf:
            if series >= len(tf.series):
                raise ValueError('Selected TIFF series does not exist.')
            selected = tf.series[series]
            raw, source_axes = selected.asarray(), selected.axes
            canonical = normalise_axes(raw, axes or source_axes)
            T = canonical.shape[0]
            if tf.ome_metadata:
                meta = _ome_metadata(tf.ome_metadata, series, T)
                sources.update({k: 'OME-XML' for k in meta})
            ij = tf.imagej_metadata or {}
            if not tf.ome_metadata:
                if 'spacing' in ij and 'unit' in ij:
                    meta['Z'] = convert_unit(ij['spacing'], ij['unit'], 'length')
                    sources['Z'] = 'ImageJ spacing'
                if ij.get('finterval'):
                    meta['times'] = np.arange(T) * convert_unit(ij['finterval'], ij.get('tunit', 's'), 'time')
                    sources['times'] = 'ImageJ finterval'
            # ResolutionUnit NONE has no physical meaning without an explicit
            # ImageJ unit. Avoid treating a pixel density as micrometres/pixel.
            page = selected.pages[0]
            resolution_unit = int(page.tags['ResolutionUnit'].value) if 'ResolutionUnit' in page.tags else 1
            factor = {2: 25400., 3: 10000.}.get(resolution_unit)
            if factor is None and ij.get('unit') in LENGTH_UNITS:
                factor = LENGTH_UNITS[ij['unit']]
            if factor is not None:
                for axis in 'XY':
                    tag = page.tags.get(axis + 'Resolution')
                    if axis not in meta and tag is not None:
                        numerator, denominator = tag.value
                        if numerator > 0 and denominator > 0:
                            meta[axis] = factor * denominator / numerator
                            sources[axis] = 'TIFF resolution and declared unit'
    elif path.suffix.lower() == '.czi':
        if series != 0:
            raise ValueError('CZI scene selection is not implemented; export one scene to OME-TIFF.')
        import czifile
        with czifile.CziFile(path) as czi:
            raw, source_axes = czi.asarray(), czi.axes
            canonical = normalise_axes(raw, axes or source_axes)
            root = ET.fromstring(czi.metadata())
            for element in root.iter():
                if element.tag.split('}')[-1] == 'Distance' and element.get('Id') in ('X', 'Y', 'Z'):
                    values = _children(element, 'Value')
                    if values and values[0].text:
                        meta[element.get('Id')] = float(values[0].text) * 1e6
                        sources[element.get('Id')] = 'CZI scaling (metres)'
            ts = czi.timestamps
            if ts is not None:
                ts = np.asarray(ts, dtype=float).ravel()
                if len(ts) == canonical.shape[0]:
                    meta['times'] = ts
                    sources['times'] = 'CZI timestamp attachment'
            for tag, key in [('LensNA', 'na'), ('ComponentBitCount', 'bits')]:
                found = next((e.text for e in root.iter() if e.tag.split('}')[-1] == tag and e.text), None)
                if found:
                    meta[key] = float(found)
    else:
        raise ValueError('Supported inputs: .czi, .tif, .tiff. Convert ND2/LIF to OME-TIFF first.')

    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError('Input must contain real numeric intensities.')
    if not np.isfinite(canonical).all() or np.min(canonical) < 0:
        raise ValueError('Non-finite or negative intensities: provide a documented valid fluorescence input.')
    T, C, Z, Y, X = canonical.shape
    if C < 2 or min(Y, X) < 8:
        raise ValueError('Need at least two channels and an XY field of at least 8 by 8 pixels.')
    for value, label in [(pixel_size_um, '--pixel-size-um'), (z_step_um, '--z-step-um'),
                         (frame_interval_s, '--frame-interval-s')]:
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f'{label} must be finite and positive.')
    if pixel_size_um is not None:
        meta['X'] = meta['Y'] = pixel_size_um
        sources['X'] = sources['Y'] = 'explicit CLI override'
    if z_step_um is not None:
        meta['Z'] = z_step_um
        sources['Z'] = 'explicit CLI override'
    if frame_interval_s is not None:
        meta['times'] = np.arange(T, dtype=float) * frame_interval_s
        sources['times'] = 'explicit CLI override'
    for axis in 'XY':
        if not np.isfinite(meta.get(axis, np.nan)) or meta[axis] <= 0:
            raise ValueError('Unknown XY calibration; supply --pixel-size-um from the acquisition record.')
    if not np.isclose(meta['X'], meta['Y'], rtol=1e-4, atol=0):
        raise ValueError('Anisotropic XY pixels are unsupported; resample with documented calibration first.')
    if Z > 1 and (not np.isfinite(meta.get('Z', np.nan)) or meta['Z'] <= 0):
        raise ValueError('Unknown Z calibration; supply --z-step-um from the acquisition record.')
    if Z == 1:
        notes.append('Single Z plane: 2D analysis; dz=1 is an internal placeholder, not a measured Z spacing.')
        meta['Z'] = 1.
        sources['Z'] = '2D geometry placeholder'
    times = np.asarray(meta.get('times', [0.] if T == 1 else []), dtype=float)
    if len(times) != T or not np.isfinite(times).all() or (T > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError('Missing/invalid frame timestamps; supply --frame-interval-s only for verified regular sampling.')
    times = times - times[0]
    if T > 2 and not np.allclose(np.diff(times), np.median(np.diff(times)), rtol=.01, atol=1e-6):
        notes.append('Irregular frame timestamps: use actual times for intervals; kymographs must display nonuniform time.')
    if axes:
        sources['axes'] = 'explicit CLI override'
        notes.append('Axis labels were supplied explicitly; correctness is the operator responsibility.')
    else:
        sources['axes'] = 'file axis labels'
    bits = int(meta.get('bits', 0))
    if not bits and np.issubdtype(raw.dtype, np.integer):
        bits = raw.dtype.itemsize * 8
        sources['bits'] = 'integer storage dtype (effective detector range may differ)'
    acq = Acquisition(path, canonical.shape, meta['X'], meta['Z'], times,
                      bit_depth=bits, na=float(meta.get('na', np.nan)),
                      channel_names=meta.get('names', ()), source_axes=axes or source_axes,
                      metadata_sources=sources, notes=notes)
    return acq, canonical.astype(np.float32)


def add_input_arguments(parser):
    parser.add_argument('--axes', help='Known axis labels for the stored array; use only if file labels are ambiguous.')
    parser.add_argument('--series', type=int, default=0, help='TIFF series index; CZI only supports 0.')
    parser.add_argument('--pixel-size-um', type=float, help='Verified square XY pixel size, overriding file metadata.')
    parser.add_argument('--z-step-um', type=float, help='Verified Z plane spacing, overriding file metadata.')
    parser.add_argument('--frame-interval-s', type=float, help='Verified regular frame interval, overriding file metadata.')


def input_options(args):
    return {key: getattr(args, key, None) for key in
            ('axes', 'series', 'pixel_size_um', 'z_step_um', 'frame_interval_s')}


def collect_inputs(path, pattern=None):
    path = Path(path)
    files = sorted(path.glob(pattern)) if path.is_dir() and pattern else (
        sorted(f for f in path.iterdir() if f.suffix.lower() in ('.czi', '.tif', '.tiff')) if path.is_dir() else [path])
    if not files or not all(f.is_file() for f in files):
        raise ValueError('No supported input files found.')
    if len({f.stem for f in files}) != len(files):
        raise ValueError('Input stems must be unique to prevent result-key collisions.')
    if any('|' in f.stem for f in files):
        raise ValueError('Input filenames cannot contain the reserved | separator.')
    return files


def validate_channels(acq, channels):
    if len(set(channels)) != 2 or any(c < 0 or c >= acq.shape[1] for c in channels):
        raise ValueError('Select two different, valid channel indices.')
