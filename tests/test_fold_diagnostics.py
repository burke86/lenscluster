from types import SimpleNamespace
from dataclasses import replace
import json

import numpy as np
import matplotlib.pyplot as plt
import pytest
from astropy.wcs import WCS

from lenscluster import fold_diagnostics as fd


def test_pair_identity_parity_family_and_separation():
    xy=np.array([[0., 0.], [0., 1.], [0., 1.1], [0., 1.2], [10., 0.], [10., 2.]])
    families=np.array([0,0,1,1,2,2]);labels=np.array(['b','a','c','d','e','f']);det=np.array([1,-1,1,1,1,-1])
    pairs=fd.candidate_pairs(xy,families,labels,det,1.5)
    assert [[labels[i] for i in pair] for pair in pairs]==[['a','b']]
    order=np.array([4,2,1,0,5,3])
    reordered=fd.candidate_pairs(xy[order],families[order],labels[order],det[order],1.5)
    assert [[labels[order][i] for i in pair] for pair in reordered]==[['a','b']]
    assert not fd.candidate_pairs(xy,families,labels,np.ones(6),1.5)


@pytest.mark.parametrize('distance',[-.02,0.,.03])
def test_relative_source_exact_fold_including_missing_pair(distance):
    # beta=(2x, y^2/2 + shift); critical line y=0, caustic beta_y=shift.
    for shift in [0., 2.]:
        def mapped(points):
            x,y=np.asarray(points).T
            b=np.column_stack([2*x,y*y/2+shift])
            a=np.zeros((len(x),2,2));a[:,0,0]=2.;a[:,1,1]=y
            return b,a
        value,error=fd.relative_source(mapped,np.array([.4,shift+distance]),[np.array([0.,.01])],
                                       np.zeros(2),np.array([1.,0.]),np.array([1.,0.]))
        np.testing.assert_allclose(value,[.4,distance],atol=1e-12)
        assert error<1e-12


def test_degenerate_caustic_rejected():
    def mapped(points):
        return np.zeros_like(points),np.zeros((len(points),2,2))
    with pytest.raises(ValueError,match='Degenerate'):
        fd.relative_source(mapped,np.zeros(2),[np.zeros(2)],np.zeros(2),np.array([1.,0.]),np.array([1.,0.]))


def test_empty_artifact(tmp_path):
    fd.save_fold_data(dict(pairs=[],skipped=[],draw_indices=[]),tmp_path)
    assert json.loads((tmp_path/'fold_pairs_source_caustics.json').read_text())['pairs']==[]


def test_fold_render_style_and_all_draw_accounting(synthetic_config,tmp_path,monkeypatch):
    from lenscluster import cutouts
    rng=np.random.default_rng(11);n=40
    rel=rng.normal(size=(n,2))*[.04,.01];rel[:,1]+=.005
    curve=np.stack([np.linspace(-2,2,97),np.zeros(97)],axis=1)
    pair=dict(labels=['a','b'],family_id='toy',z_source=2.,observed=np.array([[0.,-.5],[0.,.5]]),
              separation=1.,fov=4.,relative=rel,positions=rng.normal(size=(n,2,2))*.1,
              critical=np.tile(curve,(n,1,1)),source_root_residual=np.zeros(n),failures=[],
              normal=np.array([0.,1.]),origin=np.zeros(2),seeds=curve)
    data=dict(pairs=[pair],best_draw=0,draw_indices=list(range(n)))
    fd.save_fold_data(data,tmp_path)
    assert json.loads((tmp_path/'fold_pairs_source_caustics.json').read_text())['pairs'][0]['source_draws_total']==n
    wcs=WCS(naxis=2);wcs.wcs.crpix=[1000,1000];wcs.wcs.crval=[0,0];wcs.wcs.cdelt=[-.05/3600,.05/3600];wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    band=SimpleNamespace(wcs=wcs,shape=(2000,2000),pixel_scale_arcsec=.05)
    monkeypatch.setattr(cutouts,'rgb_cutout',lambda *args:np.full((80,80,3),.2))
    before=dict(plt.rcParams)
    output=fd.render_fold_diagnostics(synthetic_config,SimpleNamespace(reference=(3,0.,0.)),data,
                                      SimpleNamespace(figures=tmp_path),({'F814W':band},('F814W',),None,'toy'))
    assert output.read_bytes().startswith(b'%PDF')
    assert dict(plt.rcParams)==before


def test_compute_all_draws_including_no_pair(synthetic_config, monkeypatch):
    import jax.numpy as jnp
    class FoldEvaluator:
        n_components=1
        def physical(self,t): return t
        def source_positions(self,t): return jnp.array([[0.,t[0]]])
        def packed_state_for_z(self,p,z): return p
        def component_rows(self,points,packed,indices,derivative=False):
            x,y=points.T
            alpha=jnp.stack([jnp.zeros_like(x),y-y*y/2],axis=-1)[None]
            u=jnp.zeros((1,len(points),2,2)).at[0,:,1,1].set(1-y)
            if derivative:
                du=jnp.zeros((1,len(points),2,2,2)).at[0,:,1,1,1].set(-1.)
                return alpha,u,du
            return alpha,u
        def total_fields_at(self,points,packed):
            a,u=self.component_rows(jnp.asarray(points),packed,None)
            return np.asarray(a[0]),np.asarray(u[0])
        def fields(self,p,force_exact=True):
            return {'jacobian':np.array([[[1.,0.],[0.,-.5]],[[1.,0.],[0.,.5]]])}
    state=SimpleNamespace(images=SimpleNamespace(x_obs=np.array([0.,0.]),y_obs=np.array([-.5,.5]),
                          image_labels=['a','b'],family_index_per_image=np.array([0,0]),family_ids=['f']),
                          family_data=[SimpleNamespace(family_id='f',z_source=2.)])
    samples=np.array([[-.02],[.03]])
    positions=np.array([[[np.nan,np.nan],[np.nan,np.nan]],[[0.,-np.sqrt(.06)],[0.,np.sqrt(.06)]]])
    monkeypatch.setattr(fd,'posterior_image_positions',lambda *args,**kwargs:positions)
    result=fd.compute_fold_diagnostics(synthetic_config,state,FoldEvaluator(),SimpleNamespace(samples=samples,log_prob=[0.,1.]),log=lambda _:None)
    assert len(result['pairs'])==1
    pair=result['pairs'][0]
    np.testing.assert_allclose(pair['relative'][:,1],samples[:,0],atol=1e-10)
    assert not pair['failures']
    assert np.isnan(pair['positions'][0]).all()
    np.testing.assert_allclose(np.sort(pair['positions'][1,:,1]),[-np.sqrt(.06),np.sqrt(.06)],atol=1e-8)


@pytest.mark.parametrize('bad',[0.,-1.,np.inf,np.nan])
def test_separation_validation(synthetic_config,bad):
    config=replace(synthetic_config,image_diagnostics=replace(synthetic_config.image_diagnostics,fold_pair_max_separation_arcsec=bad))
    with pytest.raises(ValueError,match='fold_pair'):
        config.validate()


def test_standard_finder_once_per_draw_and_label_mapping(synthetic_config, monkeypatch):
    from lenscluster.state import build_state
    from lenscluster.evaluator import Evaluator
    from lenscluster.exact_images import ExactImageSolver
    state=build_state(synthetic_config); evaluator=Evaluator(state)
    state.family_data=list(reversed(state.family_data))
    calls=[]
    def local(self,images,families,packed,sources):
        calls.append([str(f.family_id) for f in families])
        result=[]
        for family in families:
            x=np.asarray(family.x_obs);y=np.asarray(family.y_obs)
            mask=np.ones(len(x),dtype=bool);mask[0]=False
            result.append(dict(matched_model_x_arcsec=x+.01,matched_model_y_arcsec=y+.02,recovered_image_mask=mask))
        return result
    monkeypatch.setattr(ExactImageSolver,'predict_all_local',local)
    monkeypatch.setattr(ExactImageSolver,'predict_all_global',lambda *args:pytest.fail('No global search allowed'))
    posterior=SimpleNamespace(samples=np.array([state.initial_theta,state.initial_theta]))
    positions=fd.posterior_image_positions(synthetic_config,state,evaluator,posterior,log=lambda _:None)
    assert len(calls)==2 and calls[0]==list(map(str,state.images.family_ids))
    for family in state.family_data:
        for k,label in enumerate(family.image_labels):
            i=state.images.image_labels.index(label)
            if k==0:assert np.isnan(positions[:,i]).all()
            else:np.testing.assert_allclose(positions[:,i],np.tile([family.x_obs[k]+.01,family.y_obs[k]+.02],(2,1)))


@pytest.mark.parametrize('count,best',[(1,0),(17,8),(128,64),(1000,373),(10000,7777)])
def test_fold_draw_cap_is_reproducible_and_includes_best(count,best):
    indices=fd.fold_draw_indices(count,best)
    assert len(indices)==min(count,128)
    assert len(set(indices))==len(indices)
    assert best in indices and indices.min()>=0 and indices.max()<count
    np.testing.assert_array_equal(indices,fd.fold_draw_indices(count,best))


def test_system_knots_include_nonfold_knot_without_cross_system_or_recovery_filter(synthetic_config, tmp_path, monkeypatch):
    """One verified knot must bring its other system knots into the shared frame, even with a missing image root."""
    import jax.numpy as jnp
    catalog = tmp_path / 'images.yaml'
    catalog.write_text('''format: lenscluster-images/1
systems:
  S1:
    z: 2
    knots: [A, B]
    images:
      a:
        knots:
          A: [0, 0]
          B: [0.0001, 0]
      b:
        knots:
          A: [0, 0.0001]
          B: [0.0001, 0.0001]
  S2:
    z: 2
    images:
      a: {ra: 0.001, dec: 0}
      b: {ra: 0.001, dec: 0.0001}
''')
    config = replace(synthetic_config, model=replace(synthetic_config.model,
        image_constraints=replace(synthetic_config.model.image_constraints, catalog_path=catalog)))
    labels = ['S1A.a', 'S1A.b', 'S1B.a', 'S1B.b', 'S2.a', 'S2.b']
    state = SimpleNamespace(images=SimpleNamespace(image_labels=labels, family_ids=['S1A', 'S1B', 'S2'],
        family_index_per_image=np.repeat(np.arange(3), 2), x_obs=np.array([0.,0.,.1,.1,1.,1.]), y_obs=np.tile([-.5,.5],3)))
    class Evaluator:
        def physical(self, theta): return theta
        def source_positions(self, theta): return jnp.array([[theta[0], .01], [theta[0]+.2, -.01], [9.,9.]])
        def packed_state_for_z(self, physical, z): return physical
    def kernels(evaluator):
        def fields(packed, points): return np.zeros_like(points), np.broadcast_to(np.eye(2), (len(points),2,2))
        def relative(packed, sources, seeds, *args):
            return np.tile(seeds,(len(sources),1,1)), sources, np.zeros(len(sources)), np.zeros(len(sources),dtype=int)
        return fields, None, relative
    monkeypatch.setattr(fd, '_kernels', kernels)
    calls=[]
    def missing_family(config, state, evaluator, posterior, *, family_ids, log):
        calls.append(family_ids)
        roots=np.zeros((len(posterior.samples),len(labels),2));roots[:,3]=np.nan
        return roots
    monkeypatch.setattr(fd, 'posterior_image_positions', missing_family)
    curve=np.column_stack((np.linspace(-2,2,97),np.zeros(97)))
    pair=dict(labels=labels[:2], family_id='S1A', z_source=2., separation=1., fov=4.,
              observed=np.array([[0.,-.5],[0.,.5]]), positions=np.zeros((2,2,2)), relative=np.zeros((2,2)),
              normal=np.array([0.,1.]), origin=np.zeros(2), seeds=curve, critical=np.tile(curve,(2,1,1)), failures=[])
    data=dict(pairs=[pair], best_draw=1, draw_indices=[1,3])
    result=fd.add_system_knots(config,state,Evaluator(),SimpleNamespace(samples=np.arange(4.)[:,None],log_prob=np.arange(4.)),data,log=lambda _:None)
    assert calls == [['S1B']]
    assert len(result['systems'])==1
    system=result['systems'][0]
    assert system['system_id']=='S1' and system['knot_ids']==['A','B']
    assert system['labels']==labels[:4]
    np.testing.assert_allclose(system['relative'][:,:,0],[[1.,1.2],[3.,3.2]])
    assert np.isnan(system['positions'][:,3]).all()
    assert np.isfinite(system['relative']).all()
    fd.save_fold_data(result,tmp_path)
    loaded=fd.load_fold_data(tmp_path)
    np.testing.assert_allclose(loaded['systems'][0]['relative'],system['relative'])
    assert loaded['systems'][0]['source_draws_valid_by_knot']==[2,2]
    assert loaded['systems'][0]['image_draws_valid']==[2,2,2,0]


def test_smoothed_contours_preserve_probability_mass_and_resolve_thin_cloud():
    rng=np.random.default_rng(23)
    points=rng.normal(size=(128,2)) @ np.array([[.2, .18],[0.,.002]])
    grid=fd._credible_density_grid(points,grid_size=256)
    assert grid is not None
    assert abs(grid['integrated_mass']-1)<.002
    assert 1 <= grid['bandwidth_factor'] <= 2
    assert grid['points_used']==len(points)
    for level,probability in zip(grid['levels'],[.954499736,.682689492]):
        enclosed=grid['density'][grid['density']>=level].sum()*grid['cell_area']
        assert abs(enclosed-probability)<.002
    assert fd._credible_density_grid(np.zeros((20,2))) is None
    assert fd._credible_density_grid(points[:8]) is None


def test_smoothed_contours_retain_a_resolved_second_mode():
    import contourpy
    rng=np.random.default_rng(13)
    points=np.vstack((rng.normal([-4.,0.],[.25,.3],(64,2)),rng.normal([4.,0.],[.25,.3],(64,2))))
    grid=fd._credible_density_grid(points,grid_size=256)
    # Curvilinear grid contours, not an imposed covariance ellipse or a largest-island-only filter.
    segments=contourpy.contour_generator(x=grid['x'],y=grid['y'],z=grid['density']).lines(grid['levels'][1])
    assert len(segments)==2
