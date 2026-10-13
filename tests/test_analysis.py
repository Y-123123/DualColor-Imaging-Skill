"""Numerical invariants and an end-to-end synthetic-data smoke test."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'skills' / 'dual-color-live-imaging' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
import dcli_lib as lib
from image_io import normalise_axes, read_image, convert_unit, _ome_metadata
from run_onset_order import describe, onset_table, rank_matched_points, halfmax_onset


class AnalysisTests(unittest.TestCase):
    def test_axis_reorder_preserves_coordinates(self):
        a = np.arange(2*3*4*8*9).reshape(2,3,4,8,9)
        b = normalise_axes(a.transpose(1,0,3,4,2), 'CTYXZ')
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(normalise_axes(a[None], 'STCZYX'), a)

    def test_ambiguous_axes_fail(self):
        for shape, axes in [((2,2,8,9),'QCYX'), ((2,8,9),'YYX'), ((2,8,9),'TY')]:
            with self.assertRaises(ValueError):
                normalise_axes(np.zeros(shape), axes)

    def test_physical_units(self):
        self.assertEqual(convert_unit(400, 'nm', 'length'), .4)
        self.assertEqual(convert_unit(2, 'min', 'time'), 120)
        self.assertEqual(convert_unit(4, '\u00b5m', 'length'), 4)
        with self.assertRaises(ValueError):
            convert_unit(1, 'pixels', 'length')

    def test_ome_reference_times_not_channel_average(self):
        xml = '''<OME><Image><Pixels PhysicalSizeX="400" PhysicalSizeXUnit="nm"
        PhysicalSizeY="0.4" TimeIncrement="1"><Channel Name="A"/>
        <Plane TheT="0" TheC="0" TheZ="0" DeltaT="5"/>
        <Plane TheT="1" TheC="0" TheZ="0" DeltaT="7"/>
        <Plane TheT="2" TheC="0" TheZ="0" DeltaT="12"/>
        <Plane TheT="1" TheC="1" TheZ="0" DeltaT="99"/>
        </Pixels></Image></OME>'''
        m = _ome_metadata(xml, 0, 3)
        self.assertEqual(m['X'], .4)
        np.testing.assert_array_equal(m['times'], [5,7,12])

    def test_ome_tiff_irregular_timing_and_2d(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.ome.tif'
            tifffile.imwrite(path, np.ones((3,2,16,17), np.uint16), ome=True,
                metadata={'axes':'TCYX','PhysicalSizeX':400,'PhysicalSizeXUnit':'nm',
                          'PhysicalSizeY':.4,'PhysicalSizeYUnit':'um',
                          'Plane':{'DeltaT':[10,10.5,12,12.5,17,17.5]}})
            acq, data = read_image(path)
            self.assertEqual(data.shape, (3,2,1,16,17))
            self.assertAlmostEqual(acq.px_um, .4)
            np.testing.assert_allclose(acq.times_s, [0,2,7])
            self.assertTrue(any('Irregular' in s for s in acq.notes))
            self.assertEqual(acq.metadata_sources['Z'], '2D geometry placeholder')

    def test_unscaled_tiff_requires_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.tif'
            tifffile.imwrite(path, np.ones((3,2,2,16,17), np.uint16),
                             metadata={'axes':'TCZYX'}, photometric='minisblack')
            with self.assertRaisesRegex(ValueError, 'XY calibration'):
                read_image(path)
            with self.assertRaisesRegex(ValueError, 'Z calibration'):
                read_image(path, pixel_size_um=.4)
            with self.assertRaisesRegex(ValueError, 'timestamps'):
                read_image(path, pixel_size_um=.4, z_step_um=.8)
            acq,_ = read_image(path, pixel_size_um=.4, z_step_um=.8, frame_interval_s=2)
            np.testing.assert_array_equal(acq.times_s, [0,2,4])
            self.assertEqual(acq.metadata_sources['X'], 'explicit CLI override')

    def test_anisotropic_xy_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.ome.tif'
            tifffile.imwrite(path, np.ones((2,2,16,17), np.uint16), ome=True,
                metadata={'axes':'TCYX','PhysicalSizeX':.4,'PhysicalSizeY':.8,'TimeIncrement':1})
            with self.assertRaisesRegex(ValueError, 'Anisotropic'):
                read_image(path)

    def test_czi_adapter_with_mocked_reader(self):
        class FakeCzi:
            axes = 'TCZYX'
            timestamps = np.array([20,22,25])
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def asarray(self): return np.ones((3,2,2,16,17), np.uint16)
            def metadata(self):
                return '<Metadata><Distance Id="X"><Value>4e-7</Value></Distance>' + \
                    '<Distance Id="Y"><Value>4e-7</Value></Distance>' + \
                    '<Distance Id="Z"><Value>8e-7</Value></Distance></Metadata>'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.czi'; path.touch()
            with patch('czifile.CziFile', return_value=FakeCzi()):
                acq, _ = read_image(path)
            self.assertAlmostEqual(acq.px_um, .4)
            self.assertAlmostEqual(acq.dz_um, .8)
            np.testing.assert_array_equal(acq.times_s, [0,2,5])

    def test_channel_shift_and_drift_direction(self):
        rng = np.random.default_rng(123)
        a = ndi.gaussian_filter(rng.random((48,48)), 1)
        data = np.stack([np.stack([a,a]), np.stack([ndi.shift(a,(2,-3)),ndi.shift(a,(2,-3))])])[:,:,None]
        shifts = lib.estimate_drift(data)
        np.testing.assert_allclose(shifts[1], [-2,3], atol=.15)
        shifted = data[:1].copy(); shifted[0,1,0] = ndi.shift(a,(2,-3))
        aligned = lib.apply_channel_offset(shifted, 1, [-2,3])
        np.testing.assert_allclose(aligned[0,1,0,5:-5,5:-5], a[5:-5,5:-5], atol=1e-6)
        np.testing.assert_array_equal(aligned[:,0], shifted[:,0])

    def test_registration_empty_overlap_fails(self):
        with self.assertRaises(ValueError):
            lib.valid_mask((16,16), np.array([[0,0],[20,0]]))

    def test_absent_channel_not_flattened_or_given_false_offset(self):
        data=np.zeros((2,2,1,16,16),np.float32)
        data[:,0,0,4:12,4:12]=100
        corrected,factors,raw=lib.bleach_factors(data,strict=False)
        self.assertTrue(np.isnan(factors[:,1]).all())
        np.testing.assert_array_equal(corrected,data)
        self.assertTrue(np.isnan(lib.channel_offset_um(data,0,1,.4)[0]))
        with self.assertRaisesRegex(ValueError,'undefined'):
            lib.bleach_factors(data)

    def test_matching_maximizes_pairs_instead_of_greedy(self):
        a = np.array([[0,0,0],[0,0,1.1]])
        b = np.array([[0,0,.1],[0,0,-.8]])
        pairs = lib.match_clusters(a,b,1,1,1.05,1)
        self.assertEqual({(i,j) for i,j,_ in pairs}, {(0,1),(1,0)})
        self.assertEqual(lib.match_clusters([[0,0,0]],[[3,0,0]],1,1,1,2), [])

    def test_spatial_null_seed_and_empty_counts(self):
        a = np.array([[0,2,2],[0,4,4]]); b = a.copy()
        mask = np.ones((1,8,8), bool)
        x = lib.null_match_fraction(a,b,mask,1,1,n_iter=30,rng=np.random.default_rng(4))
        y = lib.null_match_fraction(a,b,mask,1,1,n_iter=30,rng=np.random.default_rng(4))
        np.testing.assert_array_equal(x,y)
        self.assertTrue(np.all((x>=0)&(x<=1)))
        self.assertTrue(np.isnan(lib.null_match_fraction([],b,mask,1,1)).all())

    def test_count_matching_uses_selected_intensity_indices(self):
        p = np.array([[0,0,0],[0,1,1],[0,2,2]])
        np.testing.assert_array_equal(rank_matched_points(p,[100,2,50],2), p[[0,2]])

    def test_onset_actual_times_and_censoring(self):
        table = onset_table([{'A':{1:0},'B':{2:0},'pos':np.zeros(3)},
                             {'A':{0:0},'B':{2:0},'pos':np.zeros(3)}])
        summary = describe(table,'detected',np.array([0,2,9]))
        self.assertEqual(summary['median_delta_s'],7)
        self.assertEqual(summary['n_sites'],1)
        self.assertEqual(summary['n_left_censored'],1)
        self.assertTrue(np.isnan(halfmax_onset([3,3,3])))
        self.assertEqual(halfmax_onset([2,2,10,10]),2)
        self.assertEqual(describe(onset_table([]),'empty',[0,1])['n_sites'],0)

    def test_profile_displacement_sign(self):
        x = np.arange(100)
        a = np.exp(-((x-40)/5)**2) + .3*np.exp(-((x-65)/4)**2)
        b = np.exp(-((x-43)/5)**2) + .3*np.exp(-((x-68)/4)**2)
        shift,r = lib.xcorr_shift(a,b,.2,max_lag_um=4)
        self.assertAlmostEqual(shift,.6,places=2)
        self.assertGreater(r,.99)

    def test_missing_roi_does_not_fall_back_to_full_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rois.json'
            path.write_text(json.dumps({'per_file':{'other':{'rois':{}}}}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'exact input stem'):
                lib.load_rois(path,(16,16),'sample')


class EndToEndTests(unittest.TestCase):
    def run_cli(self, script, *args, expected=0):
        run = subprocess.run([sys.executable,'-X','utf8',str(SCRIPTS/script),*map(str,args)],
                             capture_output=True, text=True, encoding='utf-8', timeout=120)
        self.assertEqual(run.returncode,expected,run.stdout+'\n'+run.stderr)
        return run

    def test_synthetic_pipeline_onset_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); image=root/'synthetic.ome.tif'; out=root/'analysis'
            self.run_cli('make_synthetic_demo.py','--output',image)
            self.run_cli('audit_acquisition.py','--input',image,'--out',root/'audit')
            cmd = ['--input',image,'--out',out,'--structure','any',
                   '--min-structure-um','10','--min-skeleton-um','20',
                   '--min-branch-um','5','--extend-um','0','--n-null','20']
            self.run_cli('run_pipeline.py',*cmd)
            self.run_cli('run_onset_order.py','--results',out/'results')
            self.run_cli('make_report_figures.py','--results',out/'results','--out',root/'figures')
            cf=pd.read_csv(out/'results'/'colocalisation.csv')
            self.assertEqual(cf.frame.nunique(),8)
            self.assertTrue(cf.n_pairs.max()>0)
            self.assertTrue((cf.n_pairs<=cf[['n_A','n_B']].min(axis=1)).all())
            np.testing.assert_allclose(cf.time_min.to_numpy(), np.arange(8)*2/60)
            with np.load(out/'results'/'clusters.npz',allow_pickle=False) as archive:
                prefix=next(k[:-5] for k in archive.files if k.endswith('|meta'))
                n=len(archive[prefix+'|site_positions'])
                self.assertGreater(n,0)
                self.assertEqual(archive[prefix+'|raw_traces_A'].shape,(8,n))
                self.assertTrue(np.isfinite(archive[prefix+'|raw_traces_B']).all())
            self.assertGreater(len(list((root/'figures').glob('*.pdf'))),0)
            failed=subprocess.run([sys.executable,'-X','utf8',str(SCRIPTS/'run_pipeline.py'),*map(str,cmd)],
                                  capture_output=True,text=True,encoding='utf-8',timeout=120)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('not empty',failed.stderr)


if __name__=='__main__':
    unittest.main()
