from types import SimpleNamespace
import numpy as np
import pandas as pd
import jax.numpy as jnp
from lenscluster import image_position_contours as ipc
from lenscluster.model import FamilyData
from lenscluster.exact_images import ExactImageSolver


def test_extra_assignment_tolerance_and_empty():
    result=ipc.match_extra_roots(np.array([[10.,0.],[20.,0.]]),np.array([[20.1,0.],[10.1,0.],[100.,0.]]),1.)
    np.testing.assert_allclose(result[:2],[[20,0],[10,0]])
    assert np.isnan(result[2]).all()
    assert ipc.match_extra_roots([],np.zeros((2,2)),1.).shape==(2,2)


def test_catalog_matching_precedes_extra_tracking(synthetic_config,monkeypatch):
    family=FamilyData('a',2.,.1,['a1','a2'],np.array([0.,1.]),np.zeros(2),np.ones(2))
    state=SimpleNamespace(images=SimpleNamespace(image_labels=('a1','a2'),family_ids=('a',),x_obs=family.x_obs,y_obs=family.y_obs,family_index_per_image=np.zeros(2,int)),
                          family_data=[family],dpie_sigma0_factor_by_z={2.:.3})
    evaluator=SimpleNamespace(n_components=1,physical=lambda t:t,source_positions=lambda t:jnp.zeros((1,2)),packed_state=lambda p,f:(f,None))
    calls=[]
    def refine(self,packed,sx,sy,x,y,shoot):
        calls.append(np.asarray(packed));x=x.copy();x[2]=0.
        return dict(x=x,y=y,identification_pass=np.ones(len(x),bool))
    monkeypatch.setattr(ExactImageSolver,'_refine',refine)
    monkeypatch.setattr(ExactImageSolver,'solve_all_global',lambda *a: (_ for _ in ()).throw(AssertionError('Global search not allowed')))
    extra=pd.DataFrame(dict(family_id=['a','a'],x_model_arcsec=[.01,10.],y_model_arcsec=[0.,0.]))
    data=ipc.compute_position_draws(synthetic_config,state,evaluator,SimpleNamespace(samples=np.zeros((3,1))),extra,log=lambda _:None)
    assert len(calls)==3
    np.testing.assert_allclose(calls,np.full((3,4),.3))
    np.testing.assert_allclose(data['recovered'],np.tile([[0.,0.],[1.,0.]],(3,1,1)))
    assert np.isnan(data['extra'][:,0]).all()
    np.testing.assert_allclose(data['extra'][:,1],np.tile([10.,0.],(3,1)))


def test_gaussian_contour_probabilities_and_no_covariance_fallback():
    rng=np.random.default_rng(83);points=rng.normal(size=(4000,2))*[.1,.2]
    contours,reason=ipc.position_contours(points)
    assert reason is None
    assert [c['probability'] for c in contours]==[.682689492,.954499736]
    # Gaussian KDE broadens slightly; radii should bracket the expected 2D values.
    radii=[np.median(np.linalg.norm(np.concatenate(c['segments'])/[.1,.2],axis=1)) for c in contours]
    np.testing.assert_allclose(radii,np.sqrt(-2*np.log(1-np.array([.682689492,.954499736]))),rtol=.12)
    assert ipc.position_contours(np.zeros((100,2)))[0]==[]
    assert ipc.position_contours(points[:5])[0]==[]


def test_overview_contours_have_tiny_catalog_crosses(monkeypatch):
    import matplotlib.pyplot as plt
    from lenscluster import mass_figure as mf
    monkeypatch.setattr(mf.cutouts,'rgb_cutout',lambda *a:np.ones((16,16,3)))
    monkeypatch.setattr(mf,'aligned_rgb',lambda rgb,*a:rgb)
    geometry=dict(resolution=16,xlim=(-10.,10.),ylim=(-10.,10.),center_xy=(0.,0.),center=None,size=20.)
    catalog=pd.DataFrame(dict(x_obs_arcsec=[0.,1.],y_obs_arcsec=[0.,1.],x_model_arcsec=[.1,np.nan],y_model_arcsec=[.1,np.nan]))
    contour=dict(kind='recovered',probability=.682689492,segments=[np.array([[0.,0.],[.1,.2],[.3,0.]])])
    fig,ax=plt.subplots()
    mf.overview_axes(fig,ax,None,None,None,None,catalog,pd.DataFrame(),geometry,position_contours={'curves':[contour]})
    assert len(ax.collections)==2
    for collection in ax.collections:
        assert len(collection.get_paths()[0].vertices)==4
        np.testing.assert_array_equal(collection.get_sizes(),[4])
    assert any(line.get_color()=='tab:blue' and line.get_linestyle()=='-' for line in ax.lines)
    labels=[t.get_text() for t in ax.get_legend().get_texts()]
    assert 'Covariance ellipse' not in labels
    assert 'Catalog: missed (best fit)' in labels
    plt.close(fig)


def test_overview_draw_selection_and_cache_subset(tmp_path,monkeypatch):
    for count,best in [(17,8),(256,100),(4000,3424)]:
        indices=ipc.overview_draw_indices(count,best)
        assert len(indices)==min(count,256) and best in indices
        assert len(set(indices))==len(indices)
    config=SimpleNamespace(to_json_dict=lambda:{})
    state=SimpleNamespace(members=SimpleNamespace(classification={}))
    posterior=SimpleNamespace(samples=np.arange(4000.)[:,None],log_prob=np.arange(4000.))
    seen=[]
    def compute(c,s,e,p,extra,**kwargs):
        seen.append(len(p.samples))
        return dict(recovered=np.zeros((len(p.samples),1,2)),extra=np.zeros((len(p.samples),0,2)),image_labels=np.array(['a']))
    sizes=[]
    monkeypatch.setattr(ipc,'compute_position_draws',compute)
    monkeypatch.setattr(ipc,'prepare_contours',lambda d,**k:sizes.append(len(d['recovered'])) or dict(curves=[],summaries=[]))
    for _ in range(2):ipc.load_or_compute_contours(config,state,None,posterior,pd.DataFrame(),tmp_path,log=lambda _:None)
    assert seen==[256] and sizes==[256,256]
    # Older full-draw caches are reused and retained, while only 256 enter contours.
    np.savez_compressed(tmp_path/'posterior_overview_positions.npz',recovered=np.zeros((4000,1,2)),extra=np.zeros((4000,0,2)))
    ipc.load_or_compute_contours(config,state,None,posterior,pd.DataFrame(),tmp_path,log=lambda _:None)
    assert seen==[256] and sizes[-1]==256
    with np.load(tmp_path/'posterior_overview_positions.npz') as saved:assert len(saved['recovered'])==4000
