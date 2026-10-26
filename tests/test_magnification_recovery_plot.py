from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS

from lenscluster.magnification_recovery_plot import binned_recovery, image_truth_magnification, render_recovery, write_magnification_recovery


def test_native_truth_redshift_scaling(tmp_path):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [10, 5]
    wcs.wcs.crpix = [5, 5]
    wcs.wcs.cdelt = [-.001, .001]
    header = wcs.to_header(); header['REDSHIFT'] = 9.
    paths = []
    for name, value in [('k', .3), ('g1', .12), ('g2', .05)]:
        path = tmp_path / f'{name}.fits'
        fits.writeto(path, np.full((10, 10), value), header)
        paths.append(str(path))
    config = SimpleNamespace(kappa_true_fits=paths[0], gammax_true_fits=paths[1], gammay_true_fits=paths[2], caustic_source_redshift=9.)
    images = pd.DataFrame({'x_obs_arcsec':[0.,0.], 'y_obs_arcsec':[0.,0.], 'z_source':[1.,9.]})
    cosmo = FlatLambdaCDM(H0=70, Om0=.3)
    mu, _ = image_truth_magnification(images, (3,10.,5.), .3, cosmo, config)
    # Independent flat-universe efficiency from radial comoving distances.
    chi_l = cosmo.comoving_distance(.3).value
    efficiency = 1-chi_l/cosmo.comoving_distance([1.,9.]).value
    f = efficiency/efficiency[1]
    np.testing.assert_allclose(mu, 1/((1-.3*f)**2-(.12*f)**2-(.05*f)**2), rtol=2e-14)
    assert mu[0] != mu[1]


def test_bins_and_style_isolation(tmp_path):
    a = np.full(8, 2.)
    b = np.arange(1.,9.)
    bins = binned_recovery(a,b)
    np.testing.assert_allclose(bins.iloc[0][['model_p16','model_median','model_p84']].to_numpy(float),np.percentile(b,[16,50,84]))
    assert binned_recovery(a[:7],b[:7]).empty
    before = plt.rcParams['font.size']
    points = pd.DataFrame({'finite':[True], 'true_abs_mu':[2.], 'model_abs_mu':[3.]})
    output = tmp_path/'recovery.pdf'
    render_recovery(a,b,bins,points,output)
    assert output.read_bytes().startswith(b'%PDF')
    assert plt.rcParams['font.size']==before


def test_missing_truth_skips(tmp_path):
    config=SimpleNamespace(kappa_true_fits=None,gammax_true_fits=None,gammay_true_fits=None)
    assert write_magnification_recovery(None,pd.DataFrame(),None,None,None,config,tmp_path,tmp_path,'toy') is None
    assert not list(tmp_path.iterdir())


def test_writer_preserves_family_redshifts_and_records_limits(tmp_path, monkeypatch):
    import lenscluster.magnification_recovery_plot as module
    x,y=np.meshgrid(np.linspace(-1,1,5),np.linspace(-1,1,5))
    truth=dict(x=x,y=y,truth={'mu':np.full(x.shape,2.)},model={'mu':{'median':np.full(x.shape,3.)}},z_source=9.)
    images=pd.DataFrame(dict(family_id=['a','a','b','b'],image_label=['a1','a2','b1','b2'],
                             x_obs_arcsec=[-1,1,1,-1],y_obs_arcsec=[-1,-1,1,1],
                             z_source=[1.,1.,2.,2.],magnification_model=[3.,-4.,5.,6.]))
    seen=[]
    def image_truth(frame,*args):
        seen.extend(frame.z_source)
        return np.array([2.,-3.,4.,200.]),[9.,9.,9.]
    monkeypatch.setattr(module,'image_truth_magnification',image_truth)
    config=SimpleNamespace(kappa_true_fits='k',gammax_true_fits='g1',gammay_true_fits='g2')
    path=write_magnification_recovery(truth,images,None,None,None,config,tmp_path,tmp_path,'toy')
    assert path.name=='toy_magnification_recovery.pdf'
    assert seen==[1.,1.,2.,2.]
    points=pd.read_csv(tmp_path/'magnification_recovery_images.csv')
    assert points.within_display.sum()==3
    np.testing.assert_array_equal(points.model_abs_mu,[3,4,5,6])
