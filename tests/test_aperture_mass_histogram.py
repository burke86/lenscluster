from types import SimpleNamespace as NS
import json

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from lenscluster import aperture_mass_histogram as am


@pytest.mark.parametrize('prefix',['hera','ares'])
def test_uniform_mass_apertures_and_output(tmp_path,monkeypatch,prefix):
    wcs=WCS(naxis=2)
    wcs.wcs.crpix=[101,101];wcs.wcs.crval=[0,0]
    wcs.wcs.cdelt=[-.1/3600,.1/3600];wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    truth=tmp_path/'truth.fits'
    fits.writeto(truth,np.full((201,201),.5),wcs.to_header())
    members=pd.DataFrame(dict(member_id=['p:1','p:2','p:3'],catalog_id=['1','2','3'],
        catalog_mag=[22.,22.,22.],x_centre=[0.,5.,100.],y_centre=[0.,0.,0.]))
    state=NS(reference=(3,0.,0.),kpc_per_arcsec=6.,members=NS(member_ids=tuple(members.member_id)))
    class Evaluator:
        def packed_state_for_z(self,p,z):return None
        def total_fields_at(self,xy,packed,chunk):
            return .5*xy,np.broadcast_to(.5*np.eye(2),(len(xy),2,2))
    monkeypatch.setattr(am,'mass_per_kappa_pixel_msun',lambda *a:.01)
    paths=NS(tables=tmp_path/'tables',figures=tmp_path/'figures')
    paths.tables.mkdir();paths.figures.mkdir()
    members.to_csv(paths.tables/'member_diagnostics.csv',index=False)
    images=pd.DataFrame(dict(x_obs_arcsec=[-2,2,2,-2],y_obs_arcsec=[-2,-2,2,2]))
    config=NS(paths=NS(figure_prefix=prefix),truth=NS(kappa_true_fits=truth,caustic_source_redshift=9.))
    path=am.write_aperture_mass_histogram(config,state,Evaluator(),NS(physical=np.ones((2,1))),paths,images,log=lambda s:None)
    table=pd.read_csv(paths.tables/'member_aperture_masses.csv')
    np.testing.assert_allclose(table.ratio[:2],1,atol=1e-12)
    np.testing.assert_allclose(table.model_mass_msun[:2],.5*np.pi*table.aperture_arcsec[:2]**2,rtol=1e-12)
    assert table.inside_image_hull.tolist()==[True,False,False]
    assert table.usable.tolist()==[True,True,False]
    assert path.name==f'{prefix}_all_vs_image_footprint_histogram.pdf'
    assert path.read_bytes().startswith(b'%PDF')
    stats=json.loads((paths.tables/'aperture_mass_histogram.json').read_text())
    assert stats['samples']['all']['n']==2
    assert stats['samples']['image_hull']['n']==1
    assert stats['skipped_member_ids']==['p:3']
    smaller=am.aperture_masses(state,Evaluator(),np.ones(1),members,
        images.to_numpy(),config.truth,normalization_kpc=3.)
    np.testing.assert_allclose(smaller.model_mass_msun[:2],table.model_mass_msun[:2]*(3/am.NORMALIZATION_KPC)**2,rtol=1e-12)
    def field(x,y,family):
        return .1*x,.1*y,np.full_like(x,.1),np.zeros_like(x),np.zeros_like(x),np.full_like(x,.1)
    corrected=am.aperture_masses(state,Evaluator(),np.ones(1),members,images.to_numpy(),config.truth,
                                grf_correction=field,grf_weight=2.)
    np.testing.assert_allclose(corrected.ratio[:2],1.4,atol=1e-12)


def test_degenerate_footprint_and_other_clusters():
    assert not am.inside_image_footprint([[0,0]],[[0,0],[1,0],[2,0]]).any()
    assert am.write_aperture_mass_histogram(NS(paths=NS(figure_prefix='as1063')),None,None,None,None,None) is None


def test_quartiles_and_rendering_do_not_drop_display_tails(tmp_path):
    values=np.r_[np.linspace(.8,1.2,198),.1,4.]
    table=pd.DataFrame(dict(ratio=values,complete_aperture=True,inside_image_hull=np.arange(200)<10))
    stats=am.distribution_statistics(table)['all']
    np.testing.assert_allclose([stats['p25'],stats['median'],stats['p75']],np.quantile(values,[.25,.5,.75]))
    counts=am.render_histogram(table,tmp_path/'hist.pdf')
    assert counts.all_members.sum()==200
    assert counts.inside_image_hull.sum()==10
