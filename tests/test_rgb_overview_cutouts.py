"""Composite ordering and position contour levels."""
from types import SimpleNamespace as NS
import matplotlib.pyplot as plt
import numpy as np
from lenscluster import rgb_overview_cutouts as rc


def test_panel_order_and_only_two_sigma(monkeypatch):
    monkeypatch.setattr(rc,'rgb_reference_band',lambda b:b[0])
    monkeypatch.setattr(rc,'arcsec_to_skycoord',lambda *a:None)
    monkeypatch.setattr(rc,'arcsec_to_radec',lambda x,y,*a:(x,y))
    monkeypatch.setattr(rc,'cutout_window',lambda *a:NS(x0=0,y0=0))
    monkeypatch.setattr(rc,'rgb_cutout',lambda *a:np.zeros((100,100,3)))
    monkeypatch.setattr(rc,'critical_curve_quantiles',lambda *a:[])
    axis=np.linspace(-1,1,40);x,y=np.meshgrid(axis,axis)
    grid=dict(x=x,y=y,density=np.exp(-x*x-y*y),levels=[.3,.7])
    monkeypatch.setattr(rc,'_credible_density_grid',lambda *a:grid)
    data=dict(images=[dict(label=k,center=[0,0],fov=3,z=2,grid=grid,critical=dict(median_determinant=np.ones((4,4)),lower=np.ones((4,4)),upper=np.ones((4,4)))) for k in rc.IMAGE_LABELS],
        systems=[dict(system_id=k,center=[0,0],fov=3,z_source=2,critical=[],seeds=[],origin=[],normal=[],
                      labels=['a'],image_family_index=[0],positions=np.zeros((20,1,2)),observed=[[0,0]]) for k in rc.SYSTEM_IDS])
    ref=NS(pixel_scale_arcsec=.03,wcs=NS(world_to_pixel_values=lambda x,y:(x*40+50,y*40+50)))
    fig=plt.figure()
    info=rc.render_overview_cutouts(fig,data,({'f':ref},('f',),None,'m0416'),(3,0,0))
    assert len(fig.axes)==8
    assert info['top']==list(rc.IMAGE_LABELS) and info['right']==list(rc.SYSTEM_IDS)
    for ax in fig.axes:
        assert len(ax.collections)==1
        np.testing.assert_allclose(ax.collections[0].levels,[.3])
    assert [ax.texts[0].get_text().split('\n')[0] for ax in fig.axes]==list(rc.IMAGE_LABELS)+['System '+k for k in rc.SYSTEM_IDS]
    plt.close(fig)
