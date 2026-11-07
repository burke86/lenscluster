"""Reduced inference chart and efficiency-aware adapter; no production edits."""
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.transforms import biject_to, ExpTransform
from lenscluster.config import (LensClusterSolverConfig, LensModelConfig, ReferenceFrameConfig,
    CosmologyConfig, DPIEHaloConfig, ShearHaloConfig, MemberPopulationConfig, PriorConfig,
    ImageConstraintsConfig, RunPathsConfig, ScalingModelConfig, LikelihoodConfig, LogNormalPriorConfig)
from lenscluster.state import build_state, with_member_classes
from lenscluster.evaluator import Evaluator
from lenscluster.model import SHEAR_PROFILE
from lenscluster.sampling import distribution_for_site, SampleSite
from lenscluster import smoothed_likelihood as sl
from lenscluster.jax_cosmology import dpie_sigma0_factor_from_lensing_efficiency
from .distances import flat_wcdm_lensing_efficiency,kpc_per_arcsec
from .common import OUT, SETTINGS, write_json
from .catalogs import IMAGES, REFERENCE

def configuration():
    z=.396;kpc=float(kpc_per_arcsec(z))
    U=lambda lo,hi: PriorConfig('uniform',lower=lo,upper=hi)
    def halo(id,x,y,e,angle,core,v,xb,yb,cb,vb):
        return DPIEHaloConfig(id,x,y,e,angle,core*kpc,2000*kpc,v,z,priors={
            'x_centre':U(*xb),'y_centre':U(*yb),'ellipticite':U(.2,.9),
            'angle_pos':U(-180,180),'core_radius_kpc':U(.5,cb*kpc),'v_disp':U(*vb)})
    halos=(halo('1',.13392,-.04501,.8,-35.7557,13.5871,650,(-15,15),(-15,15),20,(350,1000)),
           halo('2',20.29565,-35.83494,.6417,-55.3798,14.416,805.1496,(15,30),(-45,-30),25,(250,1000)),
           ShearHaloConfig.free('shear',z,gamma_upper=.3))
    pop=MemberPopulationConfig('members',OUT/'members_selected.cat',17.02,.15,248.,10*kpc,z,
                               sigma_prior=U(50,500),cutkpc_prior=U(kpc,50*kpc))
    return LensClusterSolverConfig(model=LensModelConfig(
        reference=ReferenceFrameConfig(*REFERENCE),cosmology=CosmologyConfig(),large_halos=halos,
        member_populations=(pop,),image_constraints=ImageConstraintsConfig(IMAGES,sigma_inst_arcsec=.3)),
        paths=RunPathsConfig(OUT,'reduced_model'),
        scaling=ScalingModelConfig(fit_corekpc=False,alpha_sigma_prior_mean=.3,beta_radius_prior_mean=.6),
        likelihood=LikelihoodConfig(likelihood_mode='smoothed',sigma_inst_arcsec=.3,sigma_extra_prior=None,
                                    sigma_sel_prior=None,sigma_def_prior=LogNormalPriorConfig(.02,.7,1.),
                                    source_position_prior_mean_arcsec=(0.,0.),source_position_prior_sigma_arcsec=100.,
                                    freeze_determinant_gradient=False))

class EfficiencyEvaluator(Evaluator):
    """Shear amplitudes are quoted at z_ref; scale them for every arbitrary ray."""
    def __init__(self,state,reference_efficiency):
        self.reference_efficiency=reference_efficiency
        super().__init__(state,'smoothed')

    def packed_state(self,physical,factor_per_image=None):
        packed,valid=super().packed_state(physical,factor_per_image)
        factor=(jnp.take(self.factor_by_z,self.state.images.z_index_per_image)
                if factor_per_image is None else jnp.asarray(factor_per_image))
        efficiency=factor*self.efficiency_per_factor/self.reference_efficiency
        if efficiency.ndim:
            packed=packed._replace(gamma1=packed.gamma1[:,None]*efficiency,
                                   gamma2=packed.gamma2[:,None]*efficiency)
        else:
            packed=packed._replace(gamma1=packed.gamma1*efficiency,gamma2=packed.gamma2*efficiency)
        return packed,valid

    def component_rows(self,xy,packed,components,*,derivative=False):
        # Production's shear rows accept scalar component amplitudes. Handle
        # per-ray efficiencies here without changing production broadcasting.
        if packed.gamma1.ndim==1:
            return super().component_rows(xy,packed,components,derivative=derivative)
        zeros=jnp.zeros(packed.gamma1.shape[0]);plain=packed._replace(gamma1=zeros,gamma2=zeros)
        result=super().component_rows(xy,plain,components,derivative=derivative)
        a,u=result[:2];mask=np.asarray(self.state.packed_lens_spec.profile_type)[components]==SHEAR_PROFILE
        for pos in np.flatnonzero(mask):
            ci=int(components[pos]);g1,g2=packed.gamma1[ci],packed.gamma2[ci]
            a=a.at[pos].set(jnp.stack([g1*xy[:,0]+g2*xy[:,1],g2*xy[:,0]-g1*xy[:,1]],axis=-1))
            u=u.at[pos].set(jnp.stack([jnp.stack([g1,g2],-1),jnp.stack([g2,-g1],-1)],-2))
        return (a,u,result[2]) if derivative else (a,u)

    def packed_state_for_z(self,physical,z_source):
        f=flat_wcdm_lensing_efficiency(self.state.z_lens,z_source,70.,.3,-1.)
        return self.packed_state(physical,dpie_sigma0_factor_from_lensing_efficiency(f))[0]

class Model:
    def __init__(self,free_ratios=False,*,image_catalog=None):
        self.free_ratios=free_ratios;self.config=configuration()
        if image_catalog is not None:
            self.config=replace(self.config,model=replace(self.config.model,
                image_constraints=replace(self.config.model.image_constraints,catalog_path=image_catalog)))
        self.state=build_state(self.config)
        self.state=with_member_classes(self.state,np.arange(len(self.state.members.member_ids)),
                                      {'mode':'all_exact_feasibility'},centered=np.zeros(len(self.state.members.member_ids),bool))
        self.ids=np.asarray(self.state.images.family_ids,dtype=str)
        family={str(f.family_id):f for f in self.state.family_data}
        self.families=[family[x] for x in self.ids]
        self.z=np.array([f.z_source for f in self.families]);self.fi=jnp.asarray(self.state.images.family_index_per_image)
        self.ref=int(np.flatnonzero(self.ids==SETTINGS['reference_family'])[0])
        if not np.isclose(self.z[self.ref],3.221,atol=1e-6,rtol=0):
            raise ValueError('Reference family 0141 must have z=3.221; refusing a changed anchor')
        self.fiducial_f=flat_wcdm_lensing_efficiency(.396,jnp.asarray(self.z),70.,.3,-1.)
        self.fref=float(self.fiducial_f[self.ref]);self.ratio_indices=np.array([i for i in range(len(self.ids)) if i!=self.ref])
        factors={float(z):float(dpie_sigma0_factor_from_lensing_efficiency(flat_wcdm_lensing_efficiency(.396,z,70.,.3,-1.)))
                 for z in self.state.z_source_values}
        self.state=replace(self.state,kpc_per_arcsec=float(kpc_per_arcsec(.396)),dpie_sigma0_factor_by_z=factors)
        self.ev=EfficiencyEvaluator(self.state,self.fref)
        specs=self.state.parameter_specs
        self.active=np.array([i for i,p in enumerate(specs) if p.component_family in ('large','source_position','error_scale')
                              or (p.component_family=='scaling' and p.field in ('sigma','cutkpc'))])
        self.specs=[specs[i] for i in self.active];self.names=[p.name for p in self.specs]
        self.source=np.array([i for i,p in enumerate(self.specs) if p.component_family=='source_position'])
        self.global_indices=np.array([i for i,p in enumerate(self.specs) if p.component_family!='source_position'])
        # Resolve catalog/state indices on the host, before JIT traces anything.
        active_lookup={int(full):i for i,full in enumerate(self.active)}
        source_full=np.column_stack([np.asarray(self.state.images.source_x_param_index),
                                     np.asarray(self.state.images.source_y_param_index)])
        self.source_report_indices=np.asarray([[active_lookup[int(ix)],active_lookup[int(iy)]]
                                               for ix,iy in source_full],dtype=np.int32)
        self.base=jnp.asarray(self.state.initial_theta)
        for i,p in enumerate(specs):
            if p.component_family=='member_scatter_offset':self.base=self.base.at[i].set(0.)
            if p.field=='alpha_sigma':self.base=self.base.at[i].set(.3)
            if p.field=='beta_radius':self.base=self.base.at[i].set(.6)
        self.priors=[];self.transforms=[]
        for p in self.specs:
            prior=distribution_for_site(SampleSite(p.sample_name,(0,),False),[p])
            transform=biject_to(prior.support)
            # Preserve uniformity in physical units, not the log sampling chart.
            if p.prior_kind=='uniform' and p.transform_kind=='log_positive':
                prior=dist.TransformedDistribution(dist.Uniform(float(p.physical_lower),float(p.physical_upper)),ExpTransform().inv)
                # The generic inverse-exp transform advertises real support;
                # explicitly retain the finite transformed Uniform bounds.
                transform=biject_to(dist.constraints.interval(np.log(float(p.physical_lower)),np.log(float(p.physical_upper))))
            self.priors.append(prior);self.transforms.append(transform)
        self.nbase=len(self.active)
        initial=np.asarray(self.base)[self.active]
        self.u0=jnp.array([tr.inv(jnp.asarray(t)) for tr,t in zip(self.transforms,initial)])
        if free_ratios:
            f=np.asarray(self.fiducial_f)[self.ratio_indices]
            self.u0=jnp.concatenate([self.u0,jnp.asarray(np.log(f/(1-f)))])
            self.names += [f'f.{self.ids[i]}' for i in self.ratio_indices]
        self.dimension=len(self.u0)
        self.xy=np.column_stack([self.state.images.x_obs,self.state.images.y_obs])
        self.components=np.arange(self.ev.n_components,dtype=np.int32)
        self.potential=jax.jit(self._potential)
        self.vg=jax.jit(jax.value_and_grad(self._potential))
        self.report=jax.jit(self._report)

    def theta(self,u):
        return jnp.stack([tr(u[i]) for i,tr in enumerate(self.transforms)])

    def full(self,u):return self.base.at[self.active].set(self.theta(u))

    def ratios(self,u):
        return self.fiducial_f.at[self.ratio_indices].set(jax.nn.sigmoid(u[self.nbase:])) if self.free_ratios else self.fiducial_f

    def fields(self,physical,f,xy=None,derivative=True,sheet=0.):
        points=self.ev.xy if xy is None else jnp.asarray(xy)
        packed,valid=self.ev.packed_state(physical,dpie_sigma0_factor_from_lensing_efficiency(f))
        rows=self.ev.component_rows(points,packed,self.components,derivative=derivative)
        alpha=jnp.sum(rows[0],axis=0)+sheet*jnp.asarray(f)[...,None]*points
        u=jnp.sum(rows[1],axis=0)+sheet*jnp.asarray(f)[...,None,None]*jnp.eye(2)
        a=jnp.eye(2)-u
        if derivative:
            da=-jnp.sum(rows[2],axis=0)
            g=(a[:,1,1,None]*da[:,0,0]+a[:,0,0,None]*da[:,1,1]
               -a[:,1,0,None]*da[:,0,1]-a[:,0,1,None]*da[:,1,0])
        else:g=jnp.zeros_like(points)
        return dict(backprojected=points-alpha,jacobian=a,determinant_gradient=g,valid=valid.is_valid)

    def statistics(self,u):
        full=self.full(u);physical=self.ev.physical(full);f=self.ratios(u)
        fields=self.fields(physical,f[self.fi]);a=fields['jacobian'];sd=physical[self.ev.sigma_def_index]
        cov=sl.source_covariance(a,.3,sd)['covariance']
        white=sl.source_whitening(fields['backprojected'],cov,self.fi,len(self.ids))
        im=self.state.images
        coords=jnp.stack([full[im.source_x_param_index],full[im.source_y_param_index]],-1)
        transport=sl.transport_sources(coords,white,jnp.zeros(2),100.)
        stats=sl.statistics(fields['backprojected'],transport['source'][self.fi],a,fields['determinant_gradient'],.3,sd,parity=None)
        stats.update(physical=physical,sources=transport['source'],correction=transport['log_correction'],
                     model_valid=fields['valid'],transport_valid=jnp.all(transport['valid']),whitening=white)
        return stats

    def prior_u(self,u):
        theta=self.theta(u)
        value=sum(pr.log_prob(theta[i])+tr.log_abs_det_jacobian(u[i],theta[i])
                  for i,(pr,tr) in enumerate(zip(self.priors,self.transforms)))
        if self.free_ratios:
            v=u[self.nbase:];value+=jnp.sum(jax.nn.log_sigmoid(v)+jax.nn.log_sigmoid(-v))
        return value

    def _potential(self,u):
        finite=jnp.all(jnp.isfinite(u));candidate=jnp.where(finite,u,self.u0)
        physical=self.ev.physical(self.full(candidate));packed,valid=self.ev.packed_state(physical)
        # Test the reference-redshift shear, not whichever image happens to be first.
        raw,_=Evaluator.packed_state(self.ev,physical,jnp.asarray(dpie_sigma0_factor_from_lensing_efficiency(self.fref)))
        shear=jnp.sum(raw.gamma1**2+raw.gamma2**2)
        good=finite & valid.is_valid & (shear<.3**2)
        safe=jnp.where(good,candidate,self.u0);stats=self.statistics(safe)
        value=-(jnp.sum(stats['loglike'])+stats['correction']+self.prior_u(safe))
        good=good & jnp.all(stats['valid']) & stats['transport_valid'] & jnp.isfinite(value)
        return jnp.where(good,value,jnp.inf)

    def _report(self,u):
        stats=self.statistics(u);values=stats['physical'][self.active]
        values=values.at[self.source_report_indices].set(stats['sources'])
        if self.free_ratios:values=jnp.concatenate([values,self.ratios(u)[self.ratio_indices]])
        return values

    def physical_from_report(self,report):
        """For maps/Fisher: hold angular geometry fixed when varying cosmology."""
        physical=self.ev.physical(self.base).at[self.active].set(report[:self.nbase])
        source=jnp.asarray(report)[self.source_report_indices]
        return physical,source

    def physical_likelihood(self,report,family_indices=None,cosmology=None,sheet=0.,gaussian=False):
        physical,source=self.physical_from_report(report)
        f=self.fiducial_f if cosmology is None else flat_wcdm_lensing_efficiency(.396,jnp.asarray(self.z),70.,cosmology[0],cosmology[1])
        fields=self.fields(physical,f[self.fi],sheet=sheet,derivative=not gaussian)
        a=fields['jacobian'];sd=physical[self.ev.sigma_def_index]
        stats=sl.statistics(fields['backprojected'],source[self.fi],a,fields['determinant_gradient'],.3,sd)
        mask=jnp.ones(len(self.xy),bool) if family_indices is None else jnp.isin(self.fi,jnp.asarray(family_indices))
        return jnp.sum(jnp.where(mask,stats['gaussian_loglike' if gaussian else 'loglike'],0.))

    def write_configuration(self):
        write_json(OUT/'model_config.json',self.config.to_json_dict())
        write_json(OUT/'parameter_definitions.json',dict(names=self.names,global_indices=self.global_indices,
            source_indices=self.source,units='arcsec for positions, kpc for radii, km/s for velocity, dimensionless shapes/f',
            priors=[dict(name=p.name,kind=p.prior_kind,chart=p.transform_kind,lower=p.lower,upper=p.upper,
                         physical_lower=p.physical_lower,physical_upper=p.physical_upper,mean=p.mean,std=p.std) for p in self.specs],
            shear='uniform Cartesian prior inside radius 0.3 at reference-family redshift',
            reference_family=str(self.ids[self.ref]),reference_redshift=float(self.z[self.ref]),reference_f=self.fref,
            fixed_slopes=[.3,.6],fixed_member_deviations=0.,all_members_exact=True,
            normalization_prior='uniform in physical sigma and cut radius',
            simplified_model='two halos, no gas/foreground/independent galaxy; not the published four-halo model'))
