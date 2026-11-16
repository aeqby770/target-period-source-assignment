"""Unified source-conditional period inference on observed time coordinates.

The same feature map, joint objective and readout apply to every record length.
Time and periods are represented relative to the configured period scale.
"""
from pathlib import Path
import json,time,hashlib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import softmax,logsumexp,gammaln
from scipy.optimize import minimize,least_squares,minimize_scalar
from scipy.fft import rfft,irfft,next_fast_len
from period_support import project_periods_with_mean
from source_assignment_core import generic_coherent_fwl_power,fit_monotone_mean,interpolate_mean

SCHEMA="source_assignment_unified_0.3.0"
SOURCE_OPERATOR="pooled_H2_gain_original_FG_measurement"
CORE_POLICY="pooled_source90_joint_cohort"
MODEL_KIND="pooled_measurement_joint_cohort"
DEFAULT_VARIANT=5
METHODS=np.asarray(["unified","no_pool_refit","no_pool_fixed","cellwise_gain","no_peer_features",
    "independent_period_prior","uniform_mean_prior","zero_within_spread","no_waveform_views",
    "source_deviations","no_support_projection","gaussian_emission","duration_interactions"])
BASE_NAMES=np.asarray(["cell_fundamental","cell_second_increment","cell_acf_period","cell_acf_half",
    "peer_fundamental","peer_second_increment","peer_acf_period","peer_acf_half",
    "coherence","relative_cell_power","phase_alignment","orthogonal_power","quadrature"])


def _observed_steps(time,mask):
    """Predecessors refer to observed samples, including across internal gaps."""
    index=np.arange(len(time))[None]
    previous=np.maximum.accumulate(np.where(mask,index,-1),axis=1)
    previous=np.pad(previous[:,:-1],((0,0),(1,0)),constant_values=-1)
    paired=mask&(previous>=0)
    gap=np.where(paired,np.asarray(time)[None]-np.asarray(time)[np.maximum(previous,0)],0.)
    return previous,paired,gap


def ou_whiten(values,time,mask,tau):
    """Stationary OU innovations on each record's actual observed time steps.

    values has shape (N,T) or (N,T,C); tau is in the same physical unit as time.
    The first observation is unchanged, missing positions are zero, and tau=0
    is the exact independent-noise boundary.
    """
    values=np.asarray(values,dtype=float)
    scalar=values.ndim==2
    signal=values[:,:,None] if scalar else values
    mask=np.asarray(mask,dtype=bool)
    signal=np.where(mask[:,:,None],signal,0.)
    tau=np.broadcast_to(np.asarray(tau,dtype=float),(len(mask),))
    assert signal.shape[:2]==mask.shape and np.all(np.isfinite(tau)) and np.all(tau>=0.)
    previous,paired,gap=_observed_steps(np.asarray(time,dtype=float),mask)
    ratio=np.divide(gap,tau[:,None],out=np.full_like(gap,np.inf),where=paired&(tau[:,None]>0.))
    correlation=np.exp(-ratio)
    innovation=np.sqrt(-np.expm1(-2.*ratio))
    old=np.take_along_axis(signal,np.maximum(previous,0)[:,:,None],axis=1)
    result=(signal-correlation[:,:,None]*old)/innovation[:,:,None]
    result*=mask[:,:,None]
    return result[:,:,0] if scalar else result


def _gaussian_eigen_evidence(eigen,basis,rhs,noise_variance,periodic_variance,components):
    """Shared amplitude calculation; eigen geometry can be reused across cells."""
    noise_variance=np.asarray(noise_variance,dtype=float)
    periodic_variance=np.asarray(periodic_variance,dtype=float)
    ridge=2.*components*noise_variance/periodic_variance
    denominator=eigen+ridge[...,None]
    rotated=np.einsum("...ij,...i->...j",basis,rhs,optimize=True)
    coefficient=np.einsum("...ij,...j->...i",basis,rotated/denominator,optimize=True)
    eigen_variance=noise_variance[...,None]/denominator
    log_evidence=.5*(np.sum(rotated**2/denominator,axis=-1)/noise_variance-
        np.sum(np.log1p(eigen/ridge[...,None]),axis=-1))
    return coefficient,eigen_variance,log_evidence


def gaussian_period_evidence(gram,rhs,noise_variance,periodic_variance,components):
    """Integrate H2 amplitudes relative to the same Gaussian noise null.

    components is the number of clocks K (1/2); each clock has four columns.
    Each coefficient prior variance is periodic_variance/(2*K).
    """
    gram=np.asarray(gram,dtype=float)
    rhs=np.asarray(rhs,dtype=float)
    assert gram.shape[-1]==gram.shape[-2]==4*components and rhs.shape[-1]==4*components
    assert np.all(np.asarray(noise_variance)>0.) and np.all(np.asarray(periodic_variance)>0.)
    eigen,basis=np.linalg.eigh((gram+gram.swapaxes(-1,-2))/2.)
    eigen=np.maximum(eigen,0.)
    coefficient,eigen_variance,log_evidence=_gaussian_eigen_evidence(eigen,basis,rhs,noise_variance,periodic_variance,components)
    covariance=np.einsum("...ik,...k,...jk->...ij",basis,eigen_variance,basis,optimize=True)
    return {"coefficient":coefficient,"covariance":covariance,"log_evidence":log_evidence}


def scale_free_pair_evidence(time,mask,signals,period,tau,record_chunk=16):
    """OU-FWL foreground evidence with flat BG amplitudes and Jeffreys noise.

    signals is (N,T,V). The foreground g prior uses the FG design Gram before
    BG projection, and g=nu is the residual dimension after trend and BG.
    BG harmonic coefficients use unit cos/sin coordinates. The returned BG
    marginal omits only OU/trend constants common to all Q in that record.
    """
    time=np.asarray(time,dtype=float);mask=np.asarray(mask,dtype=bool)
    signals=np.asarray(signals,dtype=float);period=np.asarray(period,dtype=float)
    tau=np.broadcast_to(np.asarray(tau,dtype=float),(len(mask),))
    assert signals.ndim==3 and signals.shape[:2]==mask.shape
    count=mask.sum(1);assert np.all(count>=8)
    patterns,owner=np.unique(mask,axis=0,return_inverse=True)
    acquisition,layout=np.unique(np.column_stack([owner,tau]),axis=0,return_inverse=True)
    nrecord,nperiod,nview=len(mask),len(period),signals.shape[-1]
    conditional_bf=np.empty((nrecord,nperiod,nperiod,nview))
    partial_r2=np.empty_like(conditional_bf)
    background_marginal=np.empty((nrecord,nperiod,nview))
    foreground_rank=np.empty((nrecord,nperiod,nperiod),dtype=np.int8)
    background_rank=np.empty((nrecord,nperiod),dtype=np.int8)
    effective_n=np.empty((nrecord,nperiod),dtype=int)
    period_scale=np.sqrt(period.min()*period.max())
    t=(time-time[0])/period_scale
    angle=2.*np.pi*t[None]/(period/period_scale)[:,None]
    harmonic=np.stack([np.cos(angle),np.sin(angle)],axis=-1)
    eps=np.finfo(float).eps;tiny=np.finfo(float).tiny

    def geometry(index):
        selected=np.flatnonzero(layout==index)
        observed=patterns[int(acquisition[index,0])][None]
        n=int(observed.sum());current_tau=float(acquisition[index,1])
        centered=(t-(observed*t[None]).sum()/n)*observed[0]
        trend=np.stack([observed[0]/np.sqrt(n),centered/np.sqrt(np.sum(centered**2))],axis=-1)
        wz=ou_whiten(trend[None],time,observed,np.asarray([current_tau]))[0]
        iz=np.linalg.inv(wz.T@wz)
        flat=harmonic.transpose(1,0,2).reshape(len(time),-1)
        wh=ou_whiten(flat[None],time,observed,np.asarray([current_tau]))[0]
        wh-=wz@(iz@(wz.T@wh))
        design=wh.reshape(len(time),nperiod,2).transpose(1,0,2)
        cross=(wh.T@wh).reshape(nperiod,2,nperiod,2).transpose(0,2,1,3)
        gram=cross[np.arange(nperiod),np.arange(nperiod)]
        eigen_bg,basis_bg=np.linalg.eigh((gram+gram.swapaxes(-1,-2))/2.)
        bg_tolerance=eps*max(n,2)*np.maximum(eigen_bg[:,-1],tiny)
        bg_identified=eigen_bg>bg_tolerance[:,None]
        inverse_bg=np.einsum("qik,qk,qjk->qij",basis_bg,
            np.divide(1.,eigen_bg,out=np.zeros_like(eigen_bg),where=bg_identified),basis_bg,optimize=True)
        rank_bg=bg_identified.sum(-1)
        logdet_bg=np.sum(np.log(np.maximum(eigen_bg,tiny))*bg_identified,axis=-1)
        projection=cross@inverse_bg[None]
        conditional=gram[:,None]-projection@cross.swapaxes(-1,-2)
        conditional=(conditional+conditional.swapaxes(-1,-2))/2.
        # Identical clock spans add no foreground coefficient, including when
        # those clock columns have a lower observational rank.
        conditional[np.arange(nperiod),np.arange(nperiod)]=0.
        eigen_fg,basis_fg=np.linalg.eigh(conditional)
        reference_scale=np.maximum(eigen_bg[:,-1,None],eigen_bg[None,:,-1])
        fg_tolerance=eps*max(n,4)*np.maximum(reference_scale,tiny)
        fg_identified=eigen_fg>fg_tolerance[:,:,None]
        rank_fg=fg_identified.sum(-1)
        inverse_fg=np.einsum("pqik,pqk,pqjk->pqij",basis_fg,
            np.divide(1.,eigen_fg,out=np.zeros_like(eigen_fg),where=fg_identified),basis_fg,optimize=True)
        nu=n-2-rank_bg
        # A candidate clock's amplitude prior belongs to its own physical
        # design, not the vanishing FG residual span as P approaches Q.
        inverse_root=np.einsum("pik,pk,pjk->pij",basis_bg,
            np.divide(1.,np.sqrt(np.maximum(eigen_bg,0.)),out=np.zeros_like(eigen_bg),where=bg_identified),basis_bg,optimize=True)
        relative_gram=inverse_root[:,None]@conditional@inverse_root[:,None]
        relative_gram=np.where((rank_fg>0)[:,:,None,None],relative_gram,0.)
        prior_eigen,prior_basis=np.linalg.eigh((relative_gram+relative_gram.swapaxes(-1,-2))/2.)
        prior_eigen=np.clip(prior_eigen,0.,1.)
        denominator=prior_eigen+1./nu[None,:,None]
        regularized=np.einsum("pqik,pqk,pqjk->pqij",prior_basis,1./denominator,prior_basis,optimize=True)
        regularized=inverse_root[:,None]@regularized@inverse_root[:,None]
        logdet_foreground=np.sum(np.log1p(nu[None,:,None]*prior_eigen),axis=-1)
        contraction=np.max(prior_eigen/denominator,axis=-1)
        foreground_rank[selected]=rank_fg
        background_rank[selected]=rank_bg
        effective_n[selected]=nu
        normalisation=gammaln(nu/2.)-.5*nu*np.log(np.pi)-.5*logdet_bg

        def block(records):
            wy=ou_whiten(signals[records],time,mask[records],tau[records])
            wy-=np.einsum("ti,niv->ntv",wz,
                np.einsum("ij,ntj,ntv->niv",iz,np.broadcast_to(wz,(len(records),*wz.shape)),wy,optimize=True),optimize=True)
            rhs=np.einsum("qti,ntv->nqiv",design,wy,optimize=True)
            beta_bg=np.einsum("qij,nqjv->nqiv",inverse_bg,rhs,optimize=True)
            residual_bg=wy[:,None]-np.einsum("qti,nqiv->nqtv",design,beta_bg,optimize=True)
            s0=np.sum(residual_bg**2,axis=2)
            energy=np.sum(wy**2,axis=1)
            # This is a floating-point projection floor, not added noise.
            floor=np.maximum((eps*n)**2*energy,tiny)
            s0_used=np.maximum(s0,floor[:,None])
            conditional_rhs=np.einsum("pti,nqtv->npqiv",design,residual_bg,optimize=True)
            gain=np.einsum("npqiv,pqij,npqjv->npqv",conditional_rhs,inverse_fg,conditional_rhs,optimize=True)
            r2=np.clip(gain/s0_used[:,None],0.,1.)
            r2*=rank_fg[None,:,:,None]>0
            prior_gain=np.einsum("npqiv,pqij,npqjv->npqv",conditional_rhs,regularized,conditional_rhs,optimize=True)
            explained=np.clip(prior_gain/s0_used[:,None],0.,contraction[None,:,:,None])
            bf=-.5*logdet_foreground[None,:,:,None]-.5*nu[None,None,:,None]*np.log1p(-explained)
            conditional_bf[records]=bf
            partial_r2[records]=r2
            background_marginal[records]=normalisation[None,:,None]-.5*nu[None,:,None]*np.log(s0_used)
            return None

        list(map(block,np.array_split(selected,max(1,int(np.ceil(len(selected)/record_chunk))))))
        return None

    list(map(geometry,np.arange(len(acquisition))))
    return {"conditional_log_bf":conditional_bf,"background_log_marginal":background_marginal,
        "foreground_rank":foreground_rank,"background_rank":background_rank,
        "effective_n":effective_n,"partial_r2":partial_r2}


def physical_autocorrelation(values,mask,time_axis,period):
    """Lag correlations on actual observed coordinates, with an exact FFT fast path."""
    patterns,owner=np.unique(mask,axis=0,return_inverse=True)
    output=np.zeros((len(values),len(period)))

    def acquisition(index):
        selected=owner==index
        actual=time_axis[patterns[index]]
        signal=values[selected][:,patterns[index]]
        delta=np.diff(actual)
        if np.allclose(delta,np.median(delta),rtol=1e-9,atol=1e-12):
            fft=rfft(signal,n=next_fast_len(2*len(actual)-1),axis=-1)
            products=irfft(fft*fft.conj(),n=next_fast_len(2*len(actual)-1),axis=-1)[:,:len(actual)]
            lag=np.arange(len(actual))
            norm=np.maximum((len(actual)-lag)[None]*np.mean(signal**2,axis=1)[:,None],1e-20)
            correlation=products/norm
            continuous=period/np.median(delta)
            left=np.floor(continuous).astype(int)
            valid=(left+1)<len(actual)
            first=np.minimum(left,len(actual)-1)
            second=np.minimum(left+1,len(actual)-1)
            result=correlation[:,first]*(1-(continuous-left))[None]+correlation[:,second]*(continuous-left)[None]
            output[selected]=result*valid[None]
        else:
            query=actual[None]+period[:,None]
            right=np.searchsorted(actual,query)
            valid=right<len(actual)
            right=np.clip(right,1,len(actual)-1)
            left=right-1
            fraction=(query-actual[left])/(actual[right]-actual[left])
            reliability=np.median(delta)/(actual[right]-actual[left])

            def block(indices):
                y=signal[indices]
                shifted=y[:,left]*(1-fraction)[None]+y[:,right]*fraction[None]
                weight=valid*reliability
                numerator=np.sum(y[:,None]*shifted*weight[None],axis=-1)
                denominator=np.maximum(weight.sum(axis=1)[None]*np.mean(y**2,axis=1)[:,None],1e-20)
                return numerator/denominator
            blocks=np.array_split(np.arange(len(signal)),max(1,int(np.ceil(len(signal)/16))))
            output[selected]=np.concatenate(list(map(block,blocks)),axis=0)
        return None

    list(map(acquisition,np.arange(len(patterns))))
    return output


def observations(frame):
    """Keep all supplied observations and expose their actual missingness."""
    wide=frame.pivot(index="trace_id",columns="x",values="signal_y").sort_index(axis=1)
    meta=frame[["trace_id","run","condition"]].drop_duplicates().set_index("trace_id").loc[wide.index]
    time_axis=wide.columns.to_numpy(dtype=float)
    mask=wide.notna().to_numpy()
    count=mask.sum(axis=1)
    assert np.all(count>=8) and np.all(np.diff(time_axis)>0)
    code,runs=pd.factorize(meta.run,sort=True)
    membership=(np.arange(len(runs))[:,None]==code[None]).astype(float)
    assert np.all(membership.sum(axis=1)>1)
    return {"time":time_axis,"signal":wide.fillna(0).to_numpy(dtype=float),"mask":mask,
        "count":count,"metadata":meta,"run_index":code,"runs":np.asarray(runs,dtype=str),
        "membership":membership,"average":membership/membership.sum(axis=1)[:,None]}


def detrend(values,time_axis,mask):
    count=mask.sum(axis=1)
    center=(time_axis[None]-(mask*time_axis[None]).sum(axis=1)[:,None]/count[:,None])*mask
    trend=np.stack([mask/np.sqrt(count[:,None]),center/np.sqrt((center**2).sum(axis=1))[:,None]],axis=-1)
    residual=values-np.einsum("nti,ni->nt",trend,np.einsum("nti,nt->ni",trend,values,optimize=True),optimize=True)
    return residual*mask


def shared_background(data,period_scale,strength=.05,per_record=False):
    """A dimensionless irregular-time curvature model for the common drift."""
    t=(data["time"]-data["time"][0])/period_scale
    residual=detrend(data["signal"],t,data["mask"])
    if per_record:
        patterns,owner=np.unique(data["mask"],axis=0,return_inverse=True)
        smooth=np.zeros_like(residual)
        def smooth_records(layout):
            records=np.flatnonzero(owner==layout)
            observed=np.flatnonzero(patterns[layout])
            actual=t[observed]
            dt=np.diff(actual)
            d2=sparse.diags([2/(dt[:-1]*(dt[:-1]+dt[1:])),
                -2/(dt[:-1]*dt[1:]),2/(dt[1:]*(dt[:-1]+dt[1:]))],
                [0,1,2],shape=(len(actual)-2,len(actual)),format="csc")
            precision=sparse.eye(len(actual),format="csc")+strength*(d2.T@d2)
            values=residual[records[:,None],observed[None]].T
            fitted=np.asarray(spsolve(precision,values)).reshape(len(observed),len(records))
            smooth[records[:,None],observed[None]]=fitted.T
            return None
        list(map(smooth_records,np.arange(len(patterns))))
        return residual,residual-smooth*data["mask"]
    available=data["membership"]@data["mask"].astype(float)
    mean=(data["membership"]@residual)/np.maximum(available,1.)
    def smooth_cohort(item):
        values,number=item
        present=number>0
        actual=t[present]
        dt=np.diff(actual)
        d2=sparse.diags([2/(dt[:-1]*(dt[:-1]+dt[1:])),
            -2/(dt[:-1]*dt[1:]),2/(dt[1:]*(dt[:-1]+dt[1:]))],
            [0,1,2],shape=(len(actual)-2,len(actual)),format="csc")
        weight=number[present]/number.max()
        precision=sparse.diags(weight,format="csc")+strength*(d2.T@d2)
        output=np.zeros_like(values)
        output[present]=spsolve(precision,values[present]*weight)
        return output
    # Independent sparse systems use each cohort's observed time coordinates.
    smooth=np.stack(list(map(smooth_cohort,zip(mean,available))))
    return residual,residual-smooth[data["run_index"]]*data["mask"]


def source_waveform_statistics(frame,period,period_scale=None,harmonics=None,maximum_harmonics=16,return_waveforms=False,statistics_kind="source"):
    """Observed-mask waveform statistics without clock fitting or interpolation.

    Source statistics stream period chunks and share Gram matrices by mask.
    The optional normalized_signal is the per-record curvature residual used
    for source supervision; prediction returns statistics and metadata only.
    """
    data=observations(frame)
    period=np.asarray(period,dtype=float)
    scale=float(np.sqrt(period.min()*period.max())) if period_scale is None else float(period_scale)
    _,signal=shared_background(data,float(period.max()),strength=(2*np.pi)**-4,per_record=True)
    rms=np.sqrt(np.sum(signal**2,axis=1)/data["count"])
    signal=np.divide(signal,rms[:,None],out=np.zeros_like(signal),where=rms[:,None]>0.)
    if harmonics is None:
        observed=np.flatnonzero(data["mask"].ravel())
        actual=data["time"][observed%len(data["time"])]
        consecutive=np.diff(observed//len(data["time"]))==0
        spacing=np.diff(actual)
        median_dt=np.median(spacing[consecutive&(spacing>0.)])
        harmonics=max(1,min(int(maximum_harmonics),int(np.floor(period.min()/(2*median_dt)))-1))
    harmonics=int(harmonics)
    peer=_peer_waveform(signal,data)
    statistics=generic_coherent_fwl_power(data["time"],data["mask"],signal,peer,period,
        harmonics=harmonics,return_statistics={"source":"source","full":True}[statistics_kind])
    metadata=data["metadata"]
    result={"period":period,"period_scale":np.asarray(scale),"trace_id":metadata.index.to_numpy(dtype=str),
        "run":metadata.run.to_numpy(dtype=str),"condition":metadata.condition.to_numpy(dtype=str),
        "mask":data["mask"],"time":data["time"],"count":data["count"],"run_names":data["runs"],
        "run_index":data["run_index"],"membership":data["membership"],"average":data["average"],
        "harmonics":np.asarray(harmonics),"statistics_kind":np.asarray(statistics_kind),
        "mixing_quality":measured_mixing_quality(data,period,scale)}
    result.update(statistics)
    if return_waveforms:
        result["normalized_signal"]=signal
    return result


def source_reference_joint_measurement(bundle,parent,index,device="cuda"):
    if "source_joint_features" in bundle:
        assert str(bundle["source_joint_emission_kind"])==str(parent["source_emission_kind"])
        assert str(bundle.get("source_joint_hierarchy_kind","same_period_foreground"))==str(parent.get("source_hierarchy_kind","same_period_foreground"))
        case=int(parent.get("case_index",np.arange(len(parent["emission_state_weight"])))[index])
        position=int(pd.Index(bundle["source_joint_case_index"]).get_indexer([case])[0])
        assert position>=0
        np.testing.assert_allclose(bundle["source_joint_background_log_prior"][position],parent["emission_background_log_prior"][index],rtol=0.,atol=1e-12)
        value=bundle["source_joint_features"][position]
        assert value.shape==(*bundle["source_base"].shape[:2],2)
        return value
    q=bundle["mixing_quality"]
    background=np.broadcast_to(np.exp(parent["emission_background_log_prior"][index]),(len(q),len(bundle["period"])))
    from source_assignment_core import source_operator_covariance
    from source_correlated_geometry import calibrated_joint_source_evidence
    weight=parent["emission_state_weight"][index]
    active=np.flatnonzero(weight>0.);weight=weight[active];weight=weight/weight.sum()
    shared=parent["emission_shared_covariance"][index,active];private=parent["emission_private_covariance"][index,active]
    n=len(q);unit=np.ones((n,2));peer_weight=bundle["source_peer_weight_square"]
    finite=parent.get("emission_finite_covariance",np.zeros_like(parent["emission_shared_covariance"]))[index,active]
    cf=source_operator_covariance(shared[:,0],private[:,0],peer_weight,unit,finite_loading=finite[:,0])
    if str(parent.get("source_hierarchy_kind",""))=="own_source_peer_foreground_nuisance":
        common=parent["emission_foreground_shared_source"][index,active]
        response=np.column_stack([np.ones((len(common),2)),np.repeat(common[:,None],2,axis=1)])
        cf*=response[:,None,:,None]*response[:,None,None,:]
    cb=source_operator_covariance(shared[:,1],private[:,1],peer_weight,unit,finite_loading=finite[:,1])
    ce=source_operator_covariance(parent["emission_noise_shared_covariance"][index,active],
        parent["emission_noise_private_covariance"][index,active],peer_weight,unit)
    scale=bundle["source_pair_scale"]
    r=bundle["source_rhs"]/scale[:,None,None,:]
    energy=bundle["source_observation_energy"]/scale[:,:,None]/scale[:,None,:]
    measured=calibrated_joint_source_evidence(bundle["joint_gram"],r,bundle["layout"],cf,cb,ce,
        energy,bundle["count"]-2.,background,weight,device=device,
        shared_clock_probability=float(parent.get("shared_clock_probability",.5)),covariance_group=peer_weight,
        state_chunk=8)
    return measured["features"]


def _latent_representation_features(bundle,parent,index,device="cuda"):
    """Two complete period features, with an explicit legacy/new measurement dispatch."""
    base=bundle["source_base"];q=bundle["mixing_quality"]
    if bool(parent.get("role_disabled",False)):
        coherent=base.copy();coherent[:,:,11]=bundle["source_coherent_power"]
        return quality_design(_polynomial(coherent),q)
    polynomial=_polynomial(base)
    background=softmax(np.einsum("npd,d->np",quality_design(polynomial,q),
        parent["background_coefficient"][index],optimize=True),axis=-1)
    kind=str(parent.get("source_emission_kind","legacy_relative_roles"))
    if kind=="calibrated_shared_private_joint_gaussian":
        role=source_reference_joint_measurement(bundle,parent,index,device)
    else:
        assert kind=="legacy_relative_roles",kind
        from source_correlated_geometry import correlated_role_assignment_evidence
        covariance=parent["role_covariance"][index]
        level,owner=np.unique(q,return_inverse=True)
        role=np.empty((*base.shape[:2],2));middle=len(bundle["period"])//2
        def measured_role(number):
            selected=np.flatnonzero(owner==number)
            needed,layout=np.unique(bundle["layout"][selected],return_inverse=True)
            h=np.exp((1.-level[number])*np.log(covariance[0])+level[number]*np.log(covariance[1]))
            value,_=correlated_role_assignment_evidence(bundle["joint_gram"][needed],bundle["source_rhs"][selected],layout,
                h[None],background[None,selected],middle,middle,device=device,record_chunk=64)
            role[selected]=value[0]
            return None
        list(map(measured_role,np.arange(len(level))))
    coherent=base.copy();coherent[:,:,11]=bundle["source_coherent_power"]
    core=_polynomial(coherent);terms=_role_terms(role[None],base)[0]
    return quality_design(np.concatenate([core,terms],axis=-1),q)


def _latent_emission_parent(model,name):
    parent={"background_coefficient":model[name+"_background_coefficient"],
        "role_covariance":model[name+"_role_covariance"]}
    if bool(model.get("role_disabled",False)):
        parent.update({"role_disabled":np.asarray(True),"component_policy":model["component_policy"]})
        return parent
    if "case_index" in model:
        parent["case_index"]=model["case_index"]
    kind=str(model.get("source_emission_kind","legacy_relative_roles"))
    if kind=="calibrated_shared_private_joint_gaussian":
        parent.update({"source_emission_kind":model["source_emission_kind"],
            "shared_clock_probability":model["shared_clock_probability"],
            "emission_background_log_prior":model["emission_background_log_prior"]})
        parent.update({key:model[name+"_"+key] for key in ["emission_shared_covariance","emission_private_covariance",
            "emission_noise_shared_covariance","emission_noise_private_covariance","emission_state_weight"]})
        if name+"_emission_finite_covariance" in model:
            parent["emission_finite_covariance"]=model[name+"_emission_finite_covariance"]
    else:
        assert kind=="legacy_relative_roles",kind
    if "source_hierarchy_kind" in model:
        parent["source_hierarchy_kind"]=model["source_hierarchy_kind"]
        parent["emission_foreground_shared_source"]=model[name+"_emission_foreground_shared_source"]
    return parent


def _latent_parent_coefficient(parent,index):
    if bool(parent.get("role_disabled",False)):
        assert np.all(parent["role_coefficient"][index]==0.)
        assert str(parent.get("support_reference_center",""))=="observed_support_reference_proper_mean" or str(parent.get("core_reference_center",""))=="original_reference_core208"
        return parent["core_coefficient"][index]
    if str(parent.get("source_emission_kind","legacy_relative_roles"))=="calibrated_shared_private_joint_gaussian":
        assert str(parent.get("role_feature_kind",""))=="calibrated_shared_private_joint_gaussian","New source measurements require newly fitted reference role coefficients"
    if "source_support_kind" in parent:
        assert str(parent.get("support_reference_center",""))=="observed_support_reference_proper_mean"
    if "source_hierarchy_kind" in parent:
        assert str(parent.get("role_hierarchy_kind",""))==str(parent["source_hierarchy_kind"])
    core=parent["core_coefficient"][index];role=parent["role_coefficient"][index]
    nc=len(core)//2;nr=len(role)//2
    return np.r_[core[:nc],role[:nr],core[nc:],role[nr:]]


def predict_latent_representation_density(raw,balanced,model,cases=None,device="cuda",include_log_probability=False):
    """One conditional mixture density supplies individual and cohort readouts."""
    assert str(model["readout_kind"])=="conditional_latent_representation_density"
    support_kind=str(model.get("source_support_kind",""))
    raw,balanced=_support_bundle_pair(raw,balanced,support_kind)
    np.testing.assert_array_equal(raw["trace_id"],balanced["trace_id"])
    np.testing.assert_array_equal(raw["run_index"],balanced["run_index"])
    np.testing.assert_allclose(raw["mixing_quality"],balanced["mixing_quality"],rtol=0.,atol=1e-10)
    balanced=dict(balanced,mixing_quality=raw["mixing_quality"])
    axis=raw["period"]/float(raw["period_scale"])
    np.testing.assert_allclose(axis,balanced["period"]/float(balanced["period_scale"]),rtol=0.,atol=1e-12)
    np.testing.assert_allclose(axis,model["period"]/float(model["period_scale"]),rtol=0.,atol=1e-12)

    def case(index):
        def encoder(pair):
            name,bundle=pair
            parent=_latent_emission_parent(model,name)
            design=_latent_representation_features(bundle,parent,index,device)
            if support_kind:
                design=_support_component_design(design,raw)
                energy=_support_prior_score(model["foreground_log_prior"][index],raw)+np.einsum("njd,d->nj",design,model[name+"_coefficient"][index],optimize=True)
                energy=np.where(raw["source_support_valid"],energy,-np.inf)
                weight=energy-logsumexp(energy,axis=-1,keepdims=True)
                log_kernel=np.log(np.maximum(raw["source_support_kernel"],np.finfo(float).tiny))
                log=logsumexp(weight[:,:,None]+log_kernel,axis=1)
                return log-logsumexp(log,axis=-1,keepdims=True)
            energy=model["foreground_log_prior"][index][None]+np.einsum("npd,d->np",design,model[name+"_coefficient"][index],optimize=True)
            return energy-logsumexp(energy,axis=-1,keepdims=True)

        log_raw,log_balanced=map(encoder,[("raw",raw),("balanced",balanced)])
        q=(raw["mixing_quality"]-model["gate_center"][index])/model["gate_scale"][index]
        gate=model["gate_coefficient"][index,0]+model["gate_coefficient"][index,1]*q
        log_gate=-np.logaddexp(0.,-gate);log_other=-np.logaddexp(0.,gate)
        log_density=np.logaddexp(log_other[:,None]+log_raw,log_gate[:,None]+log_balanced)
        log_density-=logsumexp(log_density,axis=-1,keepdims=True)
        mass=np.exp(log_density)
        point=mass@raw["period"]
        mean=point@raw["average"].T
        result={"estimate":point,"source_period":point,"group_mean":mean,
            "posterior":mass,
            "balanced_probability":np.exp(log_gate),"deviation":point-mean[raw["run_index"]]}
        if include_log_probability:
            result["log_posterior"]=log_density
            result["raw_log_posterior"]=log_raw
            result["balanced_log_posterior"]=log_balanced
        return result

    cases=np.arange(len(model["gate_coefficient"])) if cases is None else np.atleast_1d(cases)
    values=list(map(case,cases))
    return {key:np.stack(list(map(lambda value:value[key],values))) for key in values[0]}


def calibrate_source_density(log_probability,period,run_index,calibrated_mean,calibration_kind="minimum_kl",within_scale=None):
    """Source distributions sharing one reference-trained cohort mean.

    The default solves a common minimum-KL exponential tilt. The affine W2\n    option pushes every grid mass through a common fixed-scale translation\n    and bounds, then uses mean-preserving grid hats. It changes the
    complete source distributions, whose expectations supply both individual
    periods and the cohort mean.  calibrated_mean is a No-only prediction from
    a calibrator learned on training reference cohorts, never a query label.
    """
    from scipy.optimize import brentq
    log=np.asarray(log_probability,dtype=float)
    period=np.asarray(period,dtype=float)
    run_index=np.asarray(run_index,dtype=int)
    target=np.asarray(calibrated_mean,dtype=float)
    assert log.ndim==3 and log.shape[1]==len(run_index) and log.shape[2]==len(period)
    assert np.all(np.isfinite(log)) and np.all(np.diff(period)>0.)
    groups=int(run_index.max())+1
    assert target.shape==(len(log),groups)
    assert np.all(np.isfinite(target)) and np.all((target>=period[0])&(target<=period[-1]))
    span=period[-1]-period[0]
    axis=(period-period[0])/span
    wanted=(target-period[0])/span
    assert calibration_kind in {"minimum_kl","wasserstein_affine"}
    affine_scale=np.ones_like(target) if within_scale is None else np.broadcast_to(np.asarray(within_scale,dtype=float),target.shape)
    assert np.all(np.isfinite(affine_scale)) and np.all(affine_scale>=0.)
    probability=np.empty_like(log);tilt=np.full_like(target,np.nan)
    shift=np.full_like(target,np.nan);clipped_mass=np.zeros(log.shape[:2])

    def cohort(position):
        case,group=position
        selected=np.flatnonzero(run_index==group)
        current=log[case,selected]
        value=float(wanted[case,group])
        if calibration_kind=="wasserstein_affine":
            base=softmax(current,axis=-1)
            pooled=base.mean(0);base_mean=float(pooled@axis)
            location=affine_scale[case,group]*(axis-base_mean)
            low=-float(location.max());high=1.-float(location.min())
            if value==0. or value==1.:
                center=low if value==0. else high
            else:
                def transport_mean(center):
                    return float(pooled@np.clip(center+location,0.,1.)-value)
                center=brentq(transport_mean,low,high,xtol=4.*np.finfo(float).eps,rtol=4.*np.finfo(float).eps)
            unbounded=center+location
            mapped=np.clip(unbounded,0.,1.)
            index=np.clip(np.searchsorted(axis,mapped,side="right")-1,0,len(period)-2)
            fraction=np.clip((mapped-axis[index])/(axis[index+1]-axis[index]),0.,1.)
            operator=np.zeros((len(period),len(period)))
            operator[np.arange(len(period)),index]=1.-fraction
            operator[np.arange(len(period)),index+1]+=fraction
            mass=base@operator
            eta=np.nan;shift[case,group]=period[0]+span*center
            clipped_mass[case,selected]=base@((unbounded<0.)|(unbounded>1.)).astype(float)
        elif value==0. or value==1.:
            mass=np.zeros_like(current);mass[:,int(value)*(len(period)-1)]=1.
            eta=-np.inf if value==0. else np.inf
        else:
            width=np.max(np.ptp(current,axis=-1))
            distance=min(value,1.-value)
            bound=(width+np.log(2.*len(period))-np.log(distance))/np.min(np.diff(axis))
            def moment(eta):
                mass=softmax(current+eta*axis[None],axis=-1)
                return float(np.mean(mass@axis)-value)
            eta=brentq(moment,-bound,bound,xtol=1e-12,rtol=4.*np.finfo(float).eps)
            mass=softmax(current+eta*axis[None],axis=-1)
        probability[case,selected]=mass;tilt[case,group]=eta
        return None

    list(map(cohort,np.ndindex(target.shape)))
    point=probability@period
    membership=(np.arange(groups)[:,None]==run_index[None]).astype(float)
    average=membership/membership.sum(1)[:,None]
    mean=point@average.T
    result={"posterior":probability,"source_period":point,"estimate":point,"group_mean":mean,
        "deviation":point-mean[:,run_index],"cohort_density_tilt":tilt,
        "calibration_kind":np.asarray("cohort_minimum_KL_exponential_density_tilt")}
    if calibration_kind=="wasserstein_affine":
        result.update({"cohort_density_shift":shift,"cohort_density_scale":np.array(affine_scale,copy=True),
            "cell_clipped_probability":clipped_mass,
            "calibration_kind":np.asarray("cohort_bounded_W2_affine_pushforward"),
            "calibration_discretization":np.asarray("linear_grid_hats_preserve_probability_and_first_moment"),
            "calibration_interval_semantics":np.asarray("transported_predictive_mass; not_generative_Bayesian_credible_interval")})
    return result


def fit_latent_representation_density(raw,balanced,raw_parent,balanced_parent,reference,qualified,training_run,
        independent_unit,view_context,maxiter=900,device="cuda",case_callback=None,initial_model=None,case_indices=None,reference_source_id=None):
    """Jointly fit two complete source-density encoders and an observed-q gate.

    All native, zero-response and positive-response views share this objective.
    Only original qualified sources count as N; biological units count as G.
    The finite-cohort mean is the mean of the same individual marginals used
    by the proper score. Shared No context is conditioned on once, without a
    latent population-location independence layer or a later mean fitting stage.
    """
    role_disabled=bool(raw_parent.get("role_disabled",False))
    assert role_disabled==bool(balanced_parent.get("role_disabled",False))
    support_kind=str(raw_parent.get("source_support_kind",""))
    assert support_kind==str(balanced_parent.get("source_support_kind",""))
    raw,balanced=_support_bundle_pair(raw,balanced,support_kind)
    period=raw["period"];scale=float(raw["period_scale"]);axis=period/scale
    np.testing.assert_array_equal(raw["trace_id"],balanced["trace_id"])
    np.testing.assert_array_equal(raw["run_index"],balanced["run_index"])
    np.testing.assert_allclose(raw["mixing_quality"],balanced["mixing_quality"],rtol=0.,atol=1e-10)
    balanced=dict(balanced,mixing_quality=raw["mixing_quality"])
    view_context=dict(view_context,independent_unit=np.asarray(independent_unit))
    weights=source_reference_weights(raw,view_context,training_run)
    identifiers=view_context.get("reference_source_id",view_context["origin_trace_id"]) if reference_source_id is None else reference_source_id
    code,names=pd.factorize(identifiers,sort=True)
    source_member=(np.arange(len(names))[:,None]==code[None]).astype(float)
    original_train=weights["train"]&view_context["is_original"][None]
    number=((original_train.astype(float)@source_member.T)>0.).sum(1)
    np.testing.assert_array_equal(number,raw_parent["reference_source_count"])
    np.testing.assert_array_equal(number,balanced_parent["reference_source_count"])
    case_indices=np.arange(len(training_run)) if case_indices is None else np.asarray(case_indices)
    resume_index=np.full(len(training_run),-1,dtype=int)
    if initial_model is not None and str(initial_model.get("regularization_center",""))=="reference_source_encoders" and str(initial_model.get("readout_kind",""))=="conditional_latent_representation_density" and str(initial_model.get("source_support_kind",""))==support_kind and bool(initial_model.get("role_disabled",False))==role_disabled and str(initial_model.get("source_hierarchy_kind",""))==str(raw_parent.get("source_hierarchy_kind","")):
        resume_index=pd.Index(initial_model["case_index"]).get_indexer(case_indices)
    target=np.where(qualified,reference/scale,0.)
    teacher=period_reference_mass(axis,target,qualified)
    reference_number=raw["membership"]@qualified.astype(float)
    reference_mean=np.divide(raw["membership"]@target,reference_number,out=np.zeros(len(reference_number)),where=reference_number>0.)
    sd=np.sqrt(np.maximum(weights["training_weight"]@(target**2)-(weights["training_weight"]@target)**2,(axis[1]-axis[0])**2/12.))
    native_weight=weights["training_weight"]*view_context["is_original"][None]
    native_weight/=native_weight.sum(1)[:,None]
    q=raw["mixing_quality"]
    center=native_weight@q
    qscale=np.sqrt(np.maximum(native_weight@(q**2)-center**2,np.finfo(float).eps))
    foreground=foreground_reference_log_prior(raw,raw_parent,np.arange(len(training_run)))
    np.testing.assert_allclose(foreground,foreground_reference_log_prior(balanced,balanced_parent,np.arange(len(training_run))),rtol=0.,atol=1e-10)
    dtype=torch.float64
    tensor=lambda value:torch.as_tensor(value,dtype=dtype,device=device)
    taxis=tensor(axis);ty=tensor(teacher);average=tensor(raw["average"])
    completed=[]

    def case(index):
        start=time.time()
        cuda_device=str(device).startswith("cuda")
        if cuda_device:
            torch.cuda.reset_peak_memory_stats(device)
        xraw=_latent_representation_features(raw,raw_parent,index,device)
        xbalanced=_latent_representation_features(balanced,balanced_parent,index,device)
        original=np.flatnonzero(view_context["is_original"])
        units=weights["effective_training_units"][index:index+1]

        def conditioned(pair):
            design,parent=pair
            native=design[None,original];nw=native_weight[index:index+1,original]
            xs,transform=_role_conditioning(native,nw,number[index:index+1])
            penalty=conditional_prior_transform(native,q[original],nw,xs,transform,number[index:index+1],units)
            score_design=_support_component_design(design,raw) if support_kind else design
            return tensor(score_design),tensor(_latent_parent_coefficient(parent,index)),tensor(xs[0]),tensor(transform[0]),tensor(penalty[0])

        xr,ir,sr,rr,pr=conditioned((xraw,raw_parent))
        xb,ib,sb,rb,pb=conditioned((xbalanced,balanced_parent))
        dimension=len(ir)
        del xraw,xbalanced
        basis=tensor(np.column_stack([np.ones(len(q)),(q-center[index])/qscale[index]]))
        tf=tensor(_support_prior_score(foreground[index],raw) if support_kind else foreground[index])
        if support_kind:
            tkernel=tensor(np.log(np.maximum(raw["source_support_kernel"],np.finfo(float).tiny)))
            tvalid=torch.as_tensor(raw["source_support_valid"],dtype=torch.bool,device=device)
        tw=tensor(weights["training_weight"][index]);tg=tensor(weights["group_weight"][index]);tr=tensor(reference_mean)
        left=view_context["pair_left"];right=view_context["pair_right"]
        eligible=view_context["pair_eligible"]&training_run[index,left]&training_run[index,right]
        pair_weight=(weights["group_weight"][index,left]+weights["group_weight"][index,right])*eligible/2.
        pair_weight/=max(pair_weight.sum(),np.finfo(float).tiny)
        tl=torch.as_tensor(left,dtype=torch.long,device=device);tre=torch.as_tensor(right,dtype=torch.long,device=device)
        te=tensor(view_context["pair_effect"]);twp=tensor(pair_weight)
        progress={"iterations":0}

        def objective(parameter,readout=False):
            theta=tensor(parameter).requires_grad_(not readout)
            dr=theta[:dimension];db=theta[dimension:2*dimension];gate=theta[-2:]
            cr=ir+(rr@dr)/sr;cb=ib+(rb@db)/sb
            if support_kind:
                log_raw=_support_torch_log_density(tf+xr@cr,tkernel,tvalid)
                log_balanced=_support_torch_log_density(tf+xb@cb,tkernel,tvalid)
            else:
                log_raw=torch.log_softmax(tf[None]+xr@cr,dim=-1)
                log_balanced=torch.log_softmax(tf[None]+xb@cb,dim=-1)
            eta=basis@gate
            log_mix=torch.logaddexp(torch.nn.functional.logsigmoid(-eta)[:,None]+log_raw,
                torch.nn.functional.logsigmoid(eta)[:,None]+log_balanced)
            log=torch.log_softmax(log_mix,dim=-1)
            mass=log.exp();individual=mass@taxis;mean=average@individual
            ce=-(log*ty*tw[:,None]).sum()
            ordered=((mass.cumsum(-1)[:,:-1]-ty.cumsum(-1)[:,:-1]).square()*tw[:,None]*tensor(np.diff(axis))[None]).sum()/sd[index]
            moment=.5*((mean-tr).square()*tg).sum()/sd[index]**2
            cf=.5*(((mean[tre]-mean[tl]-te)/sd[index]).square()*twp).sum()
            penalty=.5*((pr@dr).square().sum()+(pb@db).square().sum()+gate.square().sum()/units[0])
            loss=ce+ordered+moment+cf+penalty
            if readout:
                return {"raw_coefficient":cr.detach().cpu().numpy(),"balanced_coefficient":cb.detach().cpu().numpy(),
                    "gate_coefficient":gate.detach().cpu().numpy(),"posterior":mass.detach().cpu().numpy(),
                    "log_posterior":log.detach().cpu().numpy(),"estimate":individual.detach().cpu().numpy()*scale,
                    "group_mean":mean.detach().cpu().numpy()*scale,"balanced_probability":eta.sigmoid().detach().cpu().numpy(),
                    "loss_components":np.asarray([float(value.detach().cpu()) for value in [ce,ordered,moment,cf,penalty]])}
            gradient=torch.autograd.grad(loss,theta)[0]
            progress["loss"]=float(loss.detach().cpu())
            progress["components"]=torch.stack([value.detach() for value in [ce,ordered,moment,cf,penalty]])
            return progress["loss"],gradient.detach().cpu().numpy()

        def iteration(parameter):
            progress["iterations"]+=1
            if progress["iterations"]%25==0:
                print("REPRESENTATION_MIXTURE_ITER",index,previous_iterations+progress["iterations"],progress["loss"],
                    "CE_CRPS_mean_CF_prior",progress["components"].cpu().numpy().tolist(),"elapsed_s",time.time()-start,flush=True)

        initial=np.zeros(2*dimension+2)
        resumed=resume_index[index]
        previous_iterations=0;previous_elapsed=0.
        if resumed>=0:
            np.testing.assert_allclose(initial_model["foreground_log_prior"][resumed],foreground[index],rtol=0.,atol=1e-12)
            initial[:dimension]=np.linalg.solve(rr.cpu().numpy(),(initial_model["raw_coefficient"][resumed]-ir.cpu().numpy())*sr.cpu().numpy())
            initial[dimension:2*dimension]=np.linalg.solve(rb.cpu().numpy(),(initial_model["balanced_coefficient"][resumed]-ib.cpu().numpy())*sb.cpu().numpy())
            initial[-2:]=initial_model["gate_coefficient"][resumed]
            previous_iterations=int(initial_model["iterations"][resumed])
            previous_elapsed=float(initial_model["elapsed_s"][resumed])
        initial_loss=objective(initial)[0]
        print("REPRESENTATION_MIXTURE_START",index,number[index],units[0],initial_loss,"feature_s",time.time()-start,flush=True)
        fit=minimize(objective,initial,jac=True,callback=iteration,method="L-BFGS-B",options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-5,"maxcor":25,"maxls":40})

        def fitted_output(current):
            value=objective(current.x,True)
            value.update({"optimizer_parameter":current.x.copy(),"initial_loss":np.asarray(initial_loss),"final_loss":np.asarray(current.fun),"success":np.asarray(current.success),
                "iterations":np.asarray(previous_iterations+progress["iterations"]),"gradient":np.asarray(np.max(abs(current.jac))),"optimizer_message":np.asarray(str(current.message)),
                "elapsed_s":np.asarray(previous_elapsed+time.time()-start),
                "cuda_peak_allocated_bytes":np.asarray(torch.cuda.max_memory_allocated(device) if cuda_device else 0),
                "cuda_peak_reserved_bytes":np.asarray(torch.cuda.max_memory_reserved(device) if cuda_device else 0)})
            return value

        while not fit.success and fit.status==1:
            output=fitted_output(fit)
            if case_callback is not None:
                case_callback(index,assemble(completed+[output]))
            print("REPRESENTATION_MIXTURE_CONTINUE",index,previous_iterations+progress["iterations"],float(fit.fun),flush=True)
            fit=minimize(objective,fit.x,jac=True,callback=iteration,method="L-BFGS-B",options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-5,"maxcor":25,"maxls":40})
        output=fitted_output(fit)
        print("REPRESENTATION_MIXTURE_CASE",index,bool(fit.success),previous_iterations+progress["iterations"],float(fit.fun),"gate",output["gate_coefficient"].tolist(),"elapsed_s",time.time()-start,flush=True)
        completed.append(output)
        if case_callback is not None:
            case_callback(index,assemble(completed))
        del xr,xb
        return output

    def assemble(values):
        count=len(values)
        model={key:np.stack(list(map(lambda value:value[key],values))) for key in values[0]}
        model.update({"gate_center":center[:count],"gate_scale":qscale[:count],"foreground_log_prior":foreground[:count],
            "foreground_reference_mean":raw_parent["foreground_reference_mean"][:count],"foreground_reference_sd":raw_parent["foreground_reference_sd"][:count],
            "reference_source_count":number[:count],"effective_training_units":weights["effective_training_units"][:count],
            "raw_background_coefficient":raw_parent["background_coefficient"][:count],"raw_role_covariance":raw_parent["role_covariance"][:count],
            "balanced_background_coefficient":balanced_parent["background_coefficient"][:count],"balanced_role_covariance":balanced_parent["role_covariance"][:count],
            "readout_kind":np.asarray("conditional_latent_representation_density"),"source_representation":np.asarray("observed_and_amplitude_balanced_probability_mixture"),
            "schema":np.asarray("source_assignment_unified_0.3.1"),
            "period":period.copy(),"period_scale":np.asarray(scale),
            "gate_input":np.asarray("No_only_physical_peer_H1_energy"),"training_objective":np.asarray("joint_individual_proper_cohort_zero_and_positive_CF"),
            "regularization_center":np.asarray("reference_source_encoders"),
            "loss_component_names":np.asarray(["cross_entropy","CRPS","cohort_mean","counterfactual","prior"]),
            "likelihood_prior_policy":np.asarray("one_foreground_reference_prior_in_each_conditional_encoder")})
        emission_kind=str(raw_parent.get("source_emission_kind","legacy_relative_roles"))
        assert emission_kind==str(balanced_parent.get("source_emission_kind","legacy_relative_roles"))
        if emission_kind=="calibrated_shared_private_joint_gaussian":
            np.testing.assert_allclose(raw_parent["emission_background_log_prior"],balanced_parent["emission_background_log_prior"],rtol=0.,atol=1e-12)
            model.update({"source_emission_kind":np.asarray(emission_kind),
                "shared_clock_probability":raw_parent["shared_clock_probability"],
                "emission_background_log_prior":raw_parent["emission_background_log_prior"][:count],
                "emission_BG_prior_policy":np.asarray("training_reference_only_weak_law")})
            model.update({name+"_"+key:parent[key][:count] for name,parent in [("raw",raw_parent),("balanced",balanced_parent)]
                for key in ["emission_shared_covariance","emission_private_covariance","emission_noise_shared_covariance","emission_noise_private_covariance","emission_state_weight"]})
        if emission_kind=="calibrated_shared_private_joint_gaussian":
            model.update({name+"_emission_finite_covariance":parent.get("emission_finite_covariance",
                np.zeros_like(parent["emission_shared_covariance"]))[:count]
                for name,parent in [("raw",raw_parent),("balanced",balanced_parent)]})
        if support_kind:
            model.update({"source_support_kind":np.asarray(support_kind),
                "source_support_definition":raw["source_support_definition"],
                "support_reference_center":np.asarray("observed_support_reference_proper_mean"),
                "model_kind":np.asarray("observed_frequency_support_source_assignment"),
                "source_operator":np.asarray("No_H1_FWL_fixed_profile_components"),
                "source_core_policy":np.asarray("learn_component_source_weights_only")})
        if "source_hierarchy_kind" in raw_parent:
            assert str(raw_parent["source_hierarchy_kind"])==str(balanced_parent["source_hierarchy_kind"])
            model["source_hierarchy_kind"]=raw_parent["source_hierarchy_kind"]
            model["peer_foreground_nuisance_definition"]=raw_parent["peer_foreground_nuisance_definition"]
            model.update({name+"_emission_foreground_shared_source":parent["emission_foreground_shared_source"][:count]
                for name,parent in [("raw",raw_parent),("balanced",balanced_parent)]})
        if role_disabled:
            model.update({"role_disabled":np.asarray(True),
                "component_policy":raw_parent["component_policy"],
                "source_core_policy":np.asarray("core208_component_weights_role_zero"),
                "raw_role_coefficient":np.zeros((count,raw_parent["role_coefficient"].shape[1])),
                "balanced_role_coefficient":np.zeros((count,balanced_parent["role_coefficient"].shape[1]))})
            if "core_reference_center" in raw_parent:
                model["core_reference_center"]=raw_parent["core_reference_center"]
        model["source_period"]=model["estimate"].copy()
        model["deviation"]=model["estimate"]-model["group_mean"][:,raw["run_index"]]
        return model

    values=list(map(case,np.arange(len(training_run))))
    return assemble(values)
def harmonic_profile(values,data,period,scale,harmonics=3,ridge_fraction=1e-9):
    """Variable-projection harmonic fits for every actual observation mask."""
    t=(data["time"]-data["time"][0])/scale
    mask=data["mask"]
    unique,layout=np.unique(mask,axis=0,return_inverse=True)
    lengths=unique.sum(axis=1)
    center=(t[None]-(unique*t[None]).sum(axis=1)[:,None]/lengths[:,None])*unique
    trend=np.stack([unique/np.sqrt(lengths[:,None]),center/np.sqrt((center**2).sum(axis=1))[:,None]],axis=-1)
    angle=2*np.pi*t[None,:,None]/(period/scale)[:,None,None]*np.arange(1,harmonics+1)[None,None]
    raw=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(len(period),len(t),2*harmonics)
    cross=np.einsum("lti,ptk->lpik",trend,raw,optimize=True)
    design=raw[None]*unique[:,None,:,None]-np.einsum("lti,lpik->lptk",trend,cross,optimize=True)
    gram=np.einsum("lpti,lptj->lpij",design,design,optimize=True)
    ridge=np.trace(gram,axis1=-2,axis2=-1)[...,None,None]*float(ridge_fraction)/(2*harmonics)
    inverse=np.linalg.inv(gram+ridge*np.eye(2*harmonics))
    observed_values=detrend(values,data["time"],mask)
    rhs=np.einsum("ptk,nt->npk",raw,observed_values,optimize=True)
    beta=np.einsum("npij,npj->npi",inverse[layout],rhs,optimize=True)
    energy=np.maximum(np.sum(rhs*beta,axis=-1),0.)
    inverse_one=np.linalg.inv(gram[...,:2,:2]+ridge*np.eye(2))
    beta_one=np.einsum("npij,npj->npi",inverse_one[layout],rhs[...,:2],optimize=True)
    energy_one=np.maximum(np.sum(rhs[...,:2]*beta_one,axis=-1),0.)
    ss=np.sum(observed_values**2,axis=-1)[:,None]
    return {"time":t,"raw":raw,"design":design,"gram":gram,"inverse_one":inverse_one,
        "rhs":rhs,"beta":beta,"beta_one":beta_one,"energy":energy,"energy_one":energy_one,
        "ss":ss,"layout":layout,"unique":unique}


def _physical_peer(residual,data):
    """One cohort RMS before averaging; retain the peer's observed amplitude."""
    residual=detrend(residual,data["time"],data["mask"])
    energy=np.sum(residual**2,axis=1)/data["count"]
    cohort_rms=np.sqrt(data["average"]@energy)
    reference=np.divide(residual,cohort_rms[data["run_index"],None],out=np.zeros_like(residual),
        where=cohort_rms[data["run_index"],None]>0.)
    available=data["membership"]@data["mask"].astype(float)
    denominator=available[data["run_index"]]-data["mask"]
    peer=((data["membership"]@reference)[data["run_index"]]-reference)/np.maximum(denominator,1.)
    peer=detrend(peer,data["time"],data["mask"])
    return reference,peer


def apply_pooled_clock_balance(values,time,mask,run_index,period_pairs,tau,cellwise=False):
    """Apply one pair of pooled clock gains in each cohort's physical space.

    values is (N,T), period_pairs is (G,2), tau is (N,) in time units.
    Both data and reconstruction designs are detrended on their actual masks.
    OU is used only for coefficient estimation; the physical residual is kept.
    """
    values=np.asarray(values,dtype=float);time=np.asarray(time,dtype=float)
    mask=np.asarray(mask,dtype=bool);run_index=np.asarray(run_index,dtype=int)
    period_pairs=np.asarray(period_pairs,dtype=float)
    tau=np.broadcast_to(np.asarray(tau,dtype=float),(len(values),))
    count=mask.sum(1);membership=(np.arange(len(period_pairs))[:,None]==run_index[None]).astype(float)
    average=membership/membership.sum(1)[:,None]
    original=detrend(values,time,mask)
    input_rms=np.sqrt(average@(np.sum(original**2,axis=1)/count))
    y=np.divide(original,input_rms[run_index,None],out=np.zeros_like(original),where=input_rms[run_index,None]>0.)
    scale=np.sqrt(period_pairs.min()*period_pairs.max());t=(time-time[0])/scale
    center=(t[None]-(mask*t[None]).sum(1)[:,None]/count[:,None])*mask
    trend=np.stack([mask/np.sqrt(count[:,None]),center/np.sqrt(np.sum(center**2,axis=1))[:,None]],axis=-1)
    angle=2.*np.pi*t[None,:,None,None]*np.arange(1,3)[None,None,None]/(period_pairs[run_index]/scale)[:,None,:,None]
    physical=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(len(values),len(time),8)*mask[:,:,None]
    physical-=trend@(trend.transpose(0,2,1)@physical)
    wz=ou_whiten(trend,time,mask,tau);iz=np.linalg.inv(wz.transpose(0,2,1)@wz)
    wh=ou_whiten(physical,time,mask,tau)
    wh-=wz@(iz@(wz.transpose(0,2,1)@wh))
    wy=ou_whiten(y,time,mask,tau)
    wy-=np.einsum("nti,ni->nt",wz,(iz@(wz.transpose(0,2,1)@wy[:,:,None]))[:,:,0],optimize=True)
    gram=wh.transpose(0,2,1)@wh
    inverse=np.linalg.pinv(gram,hermitian=True)
    coefficient=(inverse@(wh.transpose(0,2,1)@wy[:,:,None]))[:,:,0]
    physical_residual=y-np.einsum("nti,ni->nt",physical,coefficient,optimize=True)
    white_residual=wy-np.einsum("nti,ni->nt",wh,coefficient,optimize=True)
    variance=np.sum(white_residual**2,axis=1)/(count-2.)
    covariance=inverse*variance[:,None,None]
    component_design=physical.reshape(len(values),len(time),2,4).transpose(0,2,1,3)
    component_gram=component_design.transpose(0,1,3,2)@component_design
    component_coefficient=coefficient.reshape(len(values),2,4)
    point=np.maximum(np.einsum("nki,nkij,nkj->nk",component_coefficient,component_gram,component_coefficient,optimize=True),0.)/count[:,None]
    component_covariance=np.stack([covariance[:,:4,:4],covariance[:,4:,4:]],axis=1)
    uncertainty=np.maximum(np.einsum("nkij,nkji->nk",component_gram,component_covariance,optimize=True),0.)/count[:,None]
    pool_point=average@point;pool_second=average@(point+uncertainty)
    common=np.sqrt(pool_point.mean(1))
    gains=np.divide(common[:,None],np.sqrt(pool_second),out=np.ones_like(pool_second),where=pool_second>0.)
    applied_gain=gains[run_index]
    if cellwise:
        common_cell=np.sqrt(point.mean(1))
        applied_gain=np.divide(common_cell[:,None],np.sqrt(point+uncertainty),out=np.ones_like(point),where=(point+uncertainty)>0.)
    balanced_coefficient=component_coefficient*applied_gain[:,:,None]
    balanced=physical_residual+np.einsum("nkti,nki->nt",component_design,balanced_coefficient,optimize=True)
    output_rms=np.sqrt(average@(np.sum(balanced**2,axis=1)/count))
    signal=np.divide(balanced,output_rms[run_index,None],out=np.zeros_like(balanced),where=output_rms[run_index,None]>0.)
    return {"signal":signal,"gains":gains,"gain_by_record":applied_gain,"pool_point_energy":pool_point,"pool_second_moment":pool_second,
        "coefficient":component_coefficient,"balanced_coefficient":balanced_coefficient,
        "residual":physical_residual,"input_cohort_rms":input_rms,"cohort_rms":output_rms,
        "clock_rank":np.linalg.matrix_rank(gram),"noise_variance":variance}


def pooled_clock_periods(values,data,period,scale,tau,maxiter=2000):
    """Shared unlabeled H2 clocks from pooled cell evidence, never mean waves."""
    y=_physical_peer(values,data)[0]
    time=data["time"];mask=data["mask"];count=data["count"]
    tau=np.broadcast_to(np.asarray(tau,dtype=float),(len(y),))
    t=(time-time[0])/scale;grid=period/scale;step=float(grid[1]-grid[0])
    p,q=np.triu_indices(len(period),1)
    patterns,owner=np.unique(mask,axis=0,return_inverse=True)
    acquisition,layout=np.unique(np.column_stack([owner,tau]),axis=0,return_inverse=True)
    pool_gain=np.zeros((len(data["runs"]),len(p)))
    pool_one=np.zeros((len(data["runs"]),len(period)))
    angle=2.*np.pi*t[None,:,None]*np.arange(1,3)[None,None]/grid[:,None,None]
    harmonic=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(len(period),len(t),4)

    def geometry(index):
        records=np.flatnonzero(layout==index);observed=patterns[int(acquisition[index,0])][None]
        n=int(observed.sum());current_tau=float(acquisition[index,1])
        center=(t-(observed*t[None]).sum()/n)*observed[0]
        z=np.stack([observed[0]/np.sqrt(n),center/np.sqrt(np.sum(center**2))],axis=-1)
        wz=ou_whiten(z[None],time,observed,np.asarray([current_tau]))[0];iz=np.linalg.inv(wz.T@wz)
        flat=harmonic.transpose(1,0,2).reshape(len(t),-1)
        wh=ou_whiten(flat[None],time,observed,np.asarray([current_tau]))[0]
        wh-=wz@(iz@(wz.T@wh))
        cross=(wh.T@wh).reshape(len(period),4,len(period),4).transpose(0,2,1,3)
        gram=cross[np.arange(len(period)),np.arange(len(period))]
        joint=np.concatenate([np.concatenate([gram[p],cross[p,q]],axis=-1),
            np.concatenate([cross[q,p],gram[q]],axis=-1)],axis=-2)
        inverse=np.linalg.pinv(joint,hermitian=True);one_inverse=np.linalg.pinv(gram,hermitian=True)
        design=wh.reshape(len(t),len(period),4).transpose(1,0,2)
        wy=ou_whiten(y[records],time,mask[records],tau[records])
        wy-=np.einsum("ti,ni->nt",wz,(iz@(wz.T@wy.T)).T,optimize=True)
        rhs=np.einsum("pti,nt->npi",design,wy,optimize=True)
        pair_rhs=np.concatenate([rhs[:,p],rhs[:,q]],axis=-1)
        gain=np.einsum("nbi,bij,nbj->nb",pair_rhs,inverse,pair_rhs,optimize=True)
        one=np.einsum("npi,pij,npj->np",rhs,one_inverse,rhs,optimize=True)
        pool_gain[:]+=data["membership"][:,records]@(gain/count[records,None])
        pool_one[:]+=data["membership"][:,records]@(one/count[records,None])
        return None

    list(map(geometry,np.arange(len(acquisition))))
    best=pool_gain.argmax(1);initial=np.stack([grid[p[best]],grid[q[best]]],axis=-1)

    def cohort(group):
        selected=np.flatnonzero(data["run_index"]==group);current_mask=mask[selected];n=count[selected]
        center=(t[None]-(current_mask*t[None]).sum(1)[:,None]/n[:,None])*current_mask
        z=np.stack([current_mask/np.sqrt(n[:,None]),center/np.sqrt(np.sum(center**2,axis=1))[:,None]],axis=-1)
        wz=ou_whiten(z,time,current_mask,tau[selected]);iz=np.linalg.inv(wz.transpose(0,2,1)@wz)
        wy=ou_whiten(y[selected],time,current_mask,tau[selected])
        wy-=np.einsum("nti,ni->nt",wz,(iz@(wz.transpose(0,2,1)@wy[:,:,None]))[:,:,0],optimize=True)

        def solve(periods,derivative=False):
            clocks=len(periods)
            angle=2.*np.pi*t[:,None,None]*np.arange(1,3)[None,None]/periods[None,:,None]
            raw=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(len(t),4*clocks)
            physical=raw[None]*current_mask[:,:,None]
            physical-=z@(z.transpose(0,2,1)@physical)
            h=ou_whiten(physical,time,current_mask,tau[selected]);h-=wz@(iz@(wz.transpose(0,2,1)@h))
            gram=h.transpose(0,2,1)@h;inverse=np.linalg.pinv(gram,hermitian=True)
            beta=(inverse@(h.transpose(0,2,1)@wy[:,:,None]))[:,:,0]
            residual=wy-np.einsum("nti,ni->nt",h,beta,optimize=True)
            if not derivative:
                return residual,gram
            change=np.stack([np.sin(angle),-np.cos(angle)],axis=-1)
            change*=(2.*np.pi*t[:,None,None]*np.arange(1,3)[None,None]/periods[None,:,None]**2)[:,:,:,None]
            change=change.reshape(len(t),4*clocks)[None]*current_mask[:,:,None]
            change-=z@(z.transpose(0,2,1)@change)
            change=ou_whiten(change,time,current_mask,tau[selected]);change-=wz@(iz@(wz.transpose(0,2,1)@change))
            change=change.reshape(len(selected),len(t),clocks,4)
            direct=np.einsum("ntsi,nsi->nts",change,beta.reshape(len(selected),clocks,4),optimize=True)
            rhs=np.einsum("ntsi,nt->nsi",change,residual,optimize=True)
            rhs=(rhs[:,:,:,None]*np.eye(clocks)[None,:,None,:]).reshape(len(selected),4*clocks,clocks)-h.transpose(0,2,1)@direct
            jacobian=-direct-h@(inverse@rhs)
            return residual,jacobian

        local=(pool_one[group]>=np.r_[-np.inf,pool_one[group,:-1]])&(pool_one[group]>=np.r_[pool_one[group,1:],-np.inf])
        def one(candidate):
            low=max(grid[0],grid[candidate]-step);high=min(grid[-1],grid[candidate]+step)
            objective=lambda x:np.sum(solve(np.asarray([x]))[0]**2/n[:,None])
            fit=minimize_scalar(objective,bounds=(low,high),method="bounded",options={"xatol":1e-10})
            points=np.asarray([low,fit.x,high]);cost=np.asarray([objective(low),fit.fun,objective(high)])
            chosen=int(cost.argmin())
            return float(cost.min()),float(points[chosen]),bool(fit.success or chosen!=1),fit.nfev
        single=list(map(one,np.flatnonzero(local)))
        one_fit=single[int(np.argmin(list(map(lambda x:x[0],single))))]
        best_one=one_fit[1]
        second_cost=np.asarray(list(map(lambda x:np.sum(solve(np.asarray([best_one,x]))[0]**2/n[:,None]),grid)))
        second_local=(second_cost<=np.r_[np.inf,second_cost[:-1]])&(second_cost<=np.r_[second_cost[1:],np.inf])
        starts=np.concatenate([initial[group:group+1],np.sort(np.column_stack([np.repeat(best_one,int(second_local.sum())),grid[second_local]]),axis=1)],axis=0)
        starts=np.unique(starts[starts[:,0]<starts[:,1]],axis=0)
        def two(start):
            midpoint=float(start.mean())
            fit=least_squares(lambda x:(solve(x)[0]/np.sqrt(n[:,None])).ravel(),start,
                jac=lambda x:(solve(x,True)[1]/np.sqrt(n[:,None,None])).reshape(-1,2),
                bounds=([grid[0],midpoint],[midpoint,grid[-1]]),x_scale="jac",tr_solver="exact",
                max_nfev=maxiter,ftol=1e-10,xtol=1e-10,gtol=1e-8)
            return float(np.sum(fit.fun**2)),fit.x,bool(fit.success),fit.nfev
        fitted=list(map(two,starts));best_two=fitted[int(np.argmin(list(map(lambda x:x[0],fitted))))]
        r_one,g_one=solve(np.asarray([best_one]));r_two,g_two=solve(best_two[1])
        effective=int(np.sum(n-2));energy=np.sum(wy**2)
        floor=max((np.finfo(float).eps*effective)**2*energy,np.finfo(float).tiny)
        rss_one=max(float(np.sum(r_one**2)),floor);rss_two=max(float(np.sum(r_two**2)),floor)
        bic_one=effective*np.log(rss_one/effective)+(int(np.linalg.matrix_rank(g_one).sum())+1)*np.log(effective)
        bic_two=effective*np.log(rss_two/effective)+(int(np.linalg.matrix_rank(g_two).sum())+2)*np.log(effective)
        use_two=bic_two<bic_one
        pair=best_two[1] if use_two else np.repeat(best_one,2)
        success,iterations=(best_two[2],best_two[3]) if use_two else (one_fit[2],one_fit[3])
        print("POOLED_CLOCKS",str(data["runs"][group]),len(selected),pair*scale,float(bic_one-bic_two),success,iterations,flush=True)
        return pair*scale,np.asarray([bic_one,bic_two]),success,iterations

    fits=list(map(cohort,np.arange(len(data["runs"]))))
    return {"period_pairs":np.stack(list(map(lambda x:x[0],fits))),"bic":np.stack(list(map(lambda x:x[1],fits))),
        "optimizer_success":np.asarray(list(map(lambda x:x[2],fits))),"optimizer_nfev":np.asarray(list(map(lambda x:x[3],fits)))}


def profile_features(residual,data,period,scale,harmonics=2,peer_reference=None,normalize_reference_peer=False,ridge_fraction=1e-9):
    residual=detrend(residual,data["time"],data["mask"])
    rms=np.sqrt((residual**2).sum(axis=1)/data["count"])
    values=residual/np.maximum(rms,np.finfo(float).eps)[:,None]
    profile=harmonic_profile(values,data,period,scale,harmonics,ridge_fraction=ridge_fraction)
    membership=data["membership"]
    code=data["run_index"]
    peers=membership.sum(axis=1)[code]-1
    beta=profile["beta"]
    beta_one=profile["beta_one"]
    available=membership@data["mask"].astype(float)
    if peer_reference is None:
        reference_values=values
        peer_values=((membership@values)[code]-values)/np.maximum(available[code]-data["mask"],1.)
        peer_values=detrend(peer_values,data["time"],data["mask"])
        peer_rms2=np.sum(peer_values**2*data["mask"],axis=1)/data["count"]
        peer_values/=np.sqrt(np.maximum(peer_rms2,1e-12))[:,None]
        coherence_scale=peer_rms2
    else:
        reference_values,peer_values=_physical_peer(peer_reference,data)
        coherence_scale=np.ones(len(values))
        if normalize_reference_peer:
            coherence_scale=np.sum(peer_values**2*data["mask"],axis=1)/data["count"]
            peer_values=np.divide(peer_values,np.sqrt(coherence_scale)[:,None],out=np.zeros_like(peer_values),where=coherence_scale[:,None]>0.)
    peer_profile=harmonic_profile(peer_values,data,period,scale,harmonics,ridge_fraction=ridge_fraction)
    peer_beta=peer_profile["beta"]
    peer_one=peer_profile["beta_one"]
    cell_norm=np.sqrt(np.sum(beta_one**2,axis=-1))
    peer_norm=np.sqrt(np.sum(peer_one**2,axis=-1))
    denominator=np.maximum(cell_norm*peer_norm,1e-10)
    own_power=profile["energy_one"]/data["count"][:,None]
    peer_h1=peer_profile["energy_one"]/data["count"][:,None]
    peer_power=peer_profile["energy"]/data["count"][:,None]
    cross_power=np.sum(profile["rhs"][...,:2]*peer_one,axis=-1)/data["count"][:,None]
    alignment=np.clip(cross_power/np.maximum(np.sqrt(own_power*peer_h1),1e-12),-1.,1.)
    signed=(beta_one[...,0]*peer_one[...,1]-beta_one[...,1]*peer_one[...,0])/denominator
    reference_power=own_power if peer_reference is None else harmonic_profile(reference_values,data,period,scale,1,ridge_fraction=ridge_fraction)["energy_one"]/data["count"][:,None]
    other_power=((membership@reference_power)[code]-reference_power)/peers[:,None]
    coherence=np.clip(peer_h1*coherence_scale[:,None]/np.maximum(other_power,1e-12),0.,1.)
    ratio=own_power/np.maximum(other_power,1e-12)
    orthogonal=own_power*(1-alignment**2)
    quadrature=signed*cell_norm*peer_norm*np.sqrt(np.maximum(np.linalg.det(profile["gram"][profile["layout"],:,:2,:2]),0.))/(data["count"][:,None]*np.maximum(np.sqrt(own_power*peer_h1),1e-12))
    quadrature=np.clip(quadrature,-1.,1.)
    ss=np.maximum(profile["ss"],1e-12)
    power=np.clip(profile["energy"]/ss,0.,1.-1e-8)
    power_one=np.clip(profile["energy_one"]/ss,0.,1.-1e-8)
    correlation=physical_autocorrelation(values,data["mask"],data["time"],np.concatenate([period,period/2]))
    peer_correlation=physical_autocorrelation(peer_values,data["mask"],data["time"],np.concatenate([period,period/2]))
    raw_features=np.stack([own_power,np.maximum(profile["energy"]/data["count"][:,None]-own_power,0.),
        correlation[:,:len(period)],correlation[:,len(period):],peer_h1,np.maximum(peer_power-peer_h1,0.),
        peer_correlation[:,:len(period)],peer_correlation[:,len(period):],coherence,ratio,
        alignment,orthogonal,quadrature],axis=-1)
    return raw_features,{"power":power,"power_one":power_one,"peer_power":peer_h1}


def _source_clock_fit(values,data,period,scale):
    """Independent regularized one/two-H2 fits for source/noise moments."""
    profile=harmonic_profile(values,data,period,scale,harmonics=2)
    pair_p,pair_q=np.triu_indices(len(period),1)
    total=np.sum(values**2,axis=1)
    best_pair=np.zeros((len(values),2),dtype=int)
    noise=np.zeros(len(values));periodic=np.zeros(len(values))
    one_power=profile["energy"]

    def geometry(index):
        rows=np.flatnonzero(profile["layout"]==index)
        h=profile["design"][index]
        gram=profile["gram"][index]
        cross=np.einsum("bti,btj->bij",h[pair_p],h[pair_q],optimize=True)
        joint=np.concatenate([np.concatenate([gram[pair_p],cross],axis=-1),
            np.concatenate([cross.swapaxes(-1,-2),gram[pair_q]],axis=-1)],axis=-2)
        inverse=np.linalg.pinv(joint,hermitian=True)

        def coarse(selected):
            rhs=np.concatenate([profile["rhs"][selected][:,pair_p],profile["rhs"][selected][:,pair_q]],axis=-1)
            gain=np.einsum("nbi,bij,nbj->nb",rhs,inverse,rhs,optimize=True)
            best=np.argmax(gain,axis=1)
            best_pair[selected]=np.stack([pair_p[best],pair_q[best]],axis=-1)
            energy=np.maximum(total[selected],np.finfo(float).eps)
            rss=np.maximum(total[selected]-gain[np.arange(len(selected)),best],np.finfo(float).eps*energy)
            noise[selected]=rss/(data["count"][selected]-2.)
            periodic[selected]=np.maximum(total[selected]-rss,np.finfo(float).eps*energy)/(data["count"][selected]-2.)
            return None

        list(map(coarse,np.array_split(rows,max(1,int(np.ceil(len(rows)/32))))))
        return None

    list(map(geometry,np.arange(len(profile["unique"]))))
    t=(data["time"]-data["time"][0])/scale
    mask=data["mask"];count=data["count"]
    centered=(t[None]-(mask*t[None]).sum(1)[:,None]/count[:,None])*mask
    trend=np.stack([mask/np.sqrt(count[:,None]),centered/np.sqrt(np.sum(centered**2,axis=1))[:,None]],axis=-1)
    grid=period/scale;step=float(grid[1]-grid[0])

    def record(index):
        def solve(periods,derivative=False):
            clocks=len(periods)
            angle=2*np.pi*t[:,None,None]*np.arange(1,3)[None,None]/periods[None,:,None]
            h=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(len(t),4*clocks)*mask[index,:,None]
            h-=trend[index]@(trend[index].T@h)
            ridge=2.*clocks*noise[index]/periodic[index]
            inverse=np.linalg.inv(h.T@h+ridge*np.eye(4*clocks))
            beta=inverse@(h.T@values[index])
            augmented=np.concatenate([h,np.sqrt(ridge)*np.eye(4*clocks)],axis=0)
            residual=np.r_[values[index],np.zeros(4*clocks)]-augmented@beta
            if not derivative:
                return residual,h,beta
            change=np.stack([np.sin(angle),-np.cos(angle)],axis=-1)
            change*=((2*np.pi*t[:,None,None]*np.arange(1,3)[None,None]/periods[None,:,None]**2)[:,:,:,None])
            change=change.reshape(len(t),4*clocks)*mask[index,:,None]
            change-=trend[index]@(trend[index].T@change)
            change=np.pad(change.reshape(len(t),clocks,4),((0,4*clocks),(0,0),(0,0)))
            direct=np.einsum("tsi,si->ts",change,beta.reshape(clocks,4),optimize=True)
            rhs=np.einsum("tsi,t->si",change,residual,optimize=True)
            rhs=(rhs[:,:,None]*np.eye(clocks)[:,None,:]).reshape(4*clocks,clocks)-augmented.T@direct
            jacobian=-direct-augmented@(inverse@rhs)
            return residual,h,beta,jacobian

        peak=(one_power[index]>=np.r_[-np.inf,one_power[index,:-1]])&(one_power[index]>=np.r_[one_power[index,1:],-np.inf])
        candidates=np.flatnonzero(peak)

        def single(candidate):
            low=max(grid[0],grid[candidate]-step);high=min(grid[-1],grid[candidate]+step)
            objective=lambda x:np.sum(solve(np.asarray([x]))[0]**2)
            fitted=minimize_scalar(objective,bounds=(low,high),method="bounded",options={"xatol":1e-10,"maxiter":200})
            refined=least_squares(lambda x:solve(x)[0],np.asarray([fitted.x]),jac=lambda x:solve(x,True)[3],
                bounds=(np.asarray([low]),np.asarray([high])),x_scale="jac",tr_solver="exact",
                ftol=1e-10,xtol=1e-10,gtol=1e-8,max_nfev=2000)
            assert refined.success,(index,refined.message,refined.nfev)
            def profile_derivative(x):
                value=solve(np.asarray([x]),True)
                return float(2.*value[0]@value[3][:,0])
            slope_low=profile_derivative(low);slope_high=profile_derivative(high)
            if slope_low<0.<slope_high:
                from scipy.optimize import brentq
                refined.x[0]=brentq(profile_derivative,low,high,xtol=1e-14,rtol=4*np.finfo(float).eps)
            values_at=np.asarray([low,refined.x[0],high])
            costs=np.asarray([objective(low),objective(refined.x[0]),objective(high)])
            period_one=values_at[np.argmin(costs)]
            residual=solve(np.asarray([period_one]))[0][:len(t)]
            return float(costs.min()),residual,period_one

        singles=list(map(single,candidates))
        one=singles[int(np.argmin(list(map(lambda value:value[0],singles))))]
        initial=grid[best_pair[index]]
        lower=np.maximum(grid[0],initial-step);upper=np.minimum(grid[-1],initial+step)
        midpoint=initial.mean();upper[0]=min(upper[0],midpoint);lower[1]=max(lower[1],midpoint)
        two=least_squares(lambda x:solve(x)[0],initial,jac=lambda x:solve(x,True)[3],bounds=(lower,upper),
            x_scale="jac",tr_solver="exact",ftol=1e-10,xtol=1e-10,gtol=1e-8,max_nfev=2000)
        assert two.success,(index,two.message,two.nfev)
        return one[1],solve(two.x)[0][:len(t)],one[2]*scale,two.x*scale,two.nfev

    fits=list(map(record,np.arange(len(values))))
    return {"one_residual":np.stack(list(map(lambda value:value[0],fits))),
        "two_residual":np.stack(list(map(lambda value:value[1],fits))),
        "one_period":np.asarray(list(map(lambda value:value[2],fits))),
        "two_periods":np.stack(list(map(lambda value:value[3],fits))),
        "two_nfev":np.asarray(list(map(lambda value:value[4],fits))),
        "noise_variance":noise,"periodic_variance":periodic,"profile":profile}


def _source_noise_tau(residual,data,scale):
    """Conditional-mean OU least squares, equal to positive AR1 OLS on a grid."""
    previous,paired,gap=_observed_steps(data["time"]/scale,data["mask"])
    old=np.take_along_axis(residual,np.maximum(previous,0),axis=1)
    tau=np.zeros(len(data["runs"]));rho=np.zeros(len(data["runs"]))

    def cohort(group):
        records=data["run_index"]==group
        valid=paired[records]
        current=residual[records][valid];lagged=old[records][valid];delta=gap[records][valid]
        estimated=np.sum(current*lagged)/max(np.sum(lagged**2),np.finfo(float).tiny)
        rho[group]=estimated
        bounded=np.clip(estimated,0.,1.-np.sqrt(np.finfo(float).eps))
        if np.allclose(delta,delta[0],rtol=1e-10,atol=1e-12):
            tau[group]=0. if bounded<=0. else -delta[0]/np.log(bounded)
        else:
            lower=-delta.min()/np.log(np.finfo(float).eps)
            upper=-delta.min()/np.log(1.-np.sqrt(np.finfo(float).eps))
            objective=lambda log_tau:np.sum((current-np.exp(-delta/np.exp(log_tau))*lagged)**2)
            fitted=minimize_scalar(objective,bounds=(np.log(lower),np.log(upper)),method="bounded",options={"xatol":1e-10})
            tau[group]=np.exp(fitted.x) if fitted.fun<np.sum(current**2) else 0.
        return None

    list(map(cohort,np.arange(len(data["runs"]))))
    return tau[data["run_index"]]*scale,{"tau":tau*scale,"lag_correlation":rho,"white_boundary":tau==0.}


def _h2_frequency_average(values,data,period,scale,tau,noise_variance,periodic_variance,profile,components=None):
    """OU coefficient evidence; posterior energies remain in physical coordinates."""
    p,q=np.triu_indices(len(period),1)
    output=np.zeros_like(values)
    component_output=None if components is None else np.zeros_like(components)
    evidence_one=np.zeros(len(values));evidence_two=np.zeros(len(values))
    physical_layout=profile["layout"]
    geometry,group=np.unique(np.column_stack([physical_layout,tau]),axis=0,return_inverse=True)
    time=data["time"];t=(time-time[0])/scale

    def acquisition(index):
        layout=int(geometry[index,0]);current_tau=float(geometry[index,1])
        rows=np.flatnonzero(group==index);mask=data["mask"][rows[:1]]
        count=int(mask.sum())
        center=(t-(mask*t[None]).sum()/count)*mask[0]
        trend=np.stack([mask[0]/np.sqrt(count),center/np.sqrt(np.sum(center**2))],axis=-1)
        physical=profile["design"][layout]
        flat_physical=physical.transpose(1,0,2).reshape(len(t),-1)
        physical_cross=(flat_physical.T@flat_physical).reshape(len(period),4,len(period),4).transpose(0,2,1,3)
        pg=physical_cross[np.arange(len(period)),np.arange(len(period))]
        physical_joint=np.concatenate([np.concatenate([pg[p],physical_cross[p,q]],axis=-1),
            np.concatenate([physical_cross[q,p],pg[q]],axis=-1)],axis=-2)
        physical_components=np.stack([pg[p],pg[q]],axis=1)
        wz=ou_whiten(trend[None],time,mask,np.asarray([current_tau]))[0]
        wh=ou_whiten(flat_physical[None],time,mask,np.asarray([current_tau]))[0]
        wh-=wz@(np.linalg.pinv(wz.T@wz)@(wz.T@wh))
        white_cross=(wh.T@wh).reshape(len(period),4,len(period),4).transpose(0,2,1,3)
        wg=white_cross[np.arange(len(period)),np.arange(len(period))]
        joint=np.concatenate([np.concatenate([wg[p],white_cross[p,q]],axis=-1),
            np.concatenate([white_cross[q,p],wg[q]],axis=-1)],axis=-2)
        eigen,basis=np.linalg.eigh((joint+joint.swapaxes(-1,-2))/2.)
        eigen=np.maximum(eigen,0.)
        one_eigen,one_basis=np.linalg.eigh((wg+wg.swapaxes(-1,-2))/2.)
        one_eigen=np.maximum(one_eigen,0.)
        pieces=basis.reshape(len(p),2,4,8)
        energy_basis=np.einsum("bsik,bsij,bsjk->bsk",pieces,physical_components,pieces,optimize=True)
        white_design=wh.reshape(len(t),len(period),4).transpose(1,0,2)

        def block(selected):
            yw=ou_whiten(values[selected],time,data["mask"][selected],tau[selected])
            rhs=np.einsum("pti,nt->npi",white_design,yw,optimize=True)
            pair_rhs=np.concatenate([rhs[:,p],rhs[:,q]],axis=-1)
            beta,eigen_variance,loglike=_gaussian_eigen_evidence(eigen[None],basis[None],pair_rhs,
                noise_variance[selected,None],periodic_variance[selected,None],2)
            mass=softmax(loglike,axis=1)
            parts=beta.reshape(len(selected),len(p),2,4)
            point=np.maximum(np.einsum("nbsi,bsij,nbsj->nbs",parts,physical_components,parts,optimize=True),0.)
            covariance_energy=np.maximum(np.einsum("nbk,bsk->nbs",eigen_variance,energy_basis,optimize=True),0.)
            posterior_rms=np.sqrt((point+covariance_energy)/data["count"][selected,None,None])
            common=np.sqrt(np.mean(point,axis=-1)/data["count"][selected,None])
            factor=np.divide(common[:,:,None],posterior_rms,out=np.zeros_like(posterior_rms),where=posterior_rms>0.)-1.
            delta=parts*factor[:,:,:,None]
            flat_delta=delta.reshape(len(selected),len(p),8)
            physical_rhs=np.concatenate([profile["rhs"][selected][:,p],profile["rhs"][selected][:,q]],axis=-1)
            energy=np.sum(values[selected]**2,axis=1)[:,None]+2*np.sum(flat_delta*physical_rhs,axis=-1)
            energy+=np.einsum("nbi,bij,nbj->nb",flat_delta,physical_joint,flat_delta,optimize=True)
            rms=np.sqrt(np.maximum(energy/data["count"][selected,None],np.finfo(float).tiny))
            weighted=mass/rms
            coefficient=np.zeros((len(selected),len(period),4))
            record=np.arange(len(selected))[:,None]
            np.add.at(coefficient,(record,p[None]),weighted[:,:,None]*delta[:,:,0])
            np.add.at(coefficient,(record,q[None]),weighted[:,:,None]*delta[:,:,1])
            output[selected]=values[selected]*weighted.sum(1)[:,None]+np.einsum("npi,pti->nt",coefficient,physical,optimize=True)
            if components is not None:
                cw=ou_whiten(components[selected],time,data["mask"][selected],tau[selected])
                crhs=np.einsum("pti,ntc->npic",white_design,cw,optimize=True)
                pair_crhs=np.concatenate([crhs[:,p],crhs[:,q]],axis=2).transpose(0,3,1,2)
                component_beta,_,_=_gaussian_eigen_evidence(eigen[None,None],basis[None,None],pair_crhs,
                    noise_variance[selected,None,None],periodic_variance[selected,None,None],2)
                component_parts=component_beta.transpose(0,2,3,1).reshape(len(selected),len(p),2,4,components.shape[-1])
                component_delta=component_parts*factor[:,:,:,None,None]
                component_coefficient=np.zeros((len(selected),len(period),4,components.shape[-1]))
                np.add.at(component_coefficient,(record,p[None]),weighted[:,:,None,None]*component_delta[:,:,0])
                np.add.at(component_coefficient,(record,q[None]),weighted[:,:,None,None]*component_delta[:,:,1])
                component_output[selected]=components[selected]*weighted.sum(1)[:,None,None]+np.einsum("npic,pti->ntc",component_coefficient,physical,optimize=True)
            _,_,one_loglike=_gaussian_eigen_evidence(one_eigen[None],one_basis[None],rhs,
                noise_variance[selected,None],periodic_variance[selected,None],1)
            evidence_one[selected]=logsumexp(one_loglike,axis=1)-np.log(len(period))
            evidence_two[selected]=logsumexp(loglike,axis=1)-np.log(len(p))
            return None

        list(map(block,np.array_split(rows,max(1,int(np.ceil(len(rows)/16))))))
        return None

    list(map(acquisition,np.arange(len(geometry))))
    details={"log_evidence_one":evidence_one,"log_evidence_two":evidence_two}
    if components is not None:
        details["component_output"]=component_output
    return output,details


def canonical_source_signal(residual,data,period,scale,share_order=True,marginal_order=False,components=None):
    """Correlated H2 source evidence, preserving physical posterior energies."""
    residual=detrend(residual,data["time"],data["mask"])
    rms=np.sqrt(np.sum(residual**2,axis=1)/data["count"])
    values=np.divide(residual,rms[:,None],out=np.zeros_like(residual),where=rms[:,None]>0.)
    component_values=None
    if components is not None:
        mask=data["mask"];count=data["count"];t=data["time"]
        centered=(t[None]-(mask*t[None]).sum(1)[:,None]/count[:,None])*mask
        trend=np.stack([mask/np.sqrt(count[:,None]),centered/np.sqrt((centered**2).sum(1))[:,None]],axis=-1)
        component_values=np.asarray(components,dtype=float).copy()
        component_values-=np.einsum("nti,nic->ntc",trend,np.einsum("nti,ntc->nic",trend,component_values,optimize=True),optimize=True)
        component_values=np.divide(component_values,rms[:,None,None],out=np.zeros_like(component_values),where=rms[:,None,None]>0.)
    fitted=_source_clock_fit(values,data,period,scale)
    tau,noise_geometry=_source_noise_tau(fitted["two_residual"],data,scale)
    conditional,evidence=_h2_frequency_average(values,data,period,scale,tau,
        fitted["noise_variance"],fitted["periodic_variance"],fitted["profile"],components=component_values)
    one_residual=ou_whiten(fitted["one_residual"],data["time"],data["mask"],tau)
    two_residual=ou_whiten(fitted["two_residual"],data["time"],data["mask"],tau)
    effective=data["count"]-2.
    one_rss=np.maximum(np.sum(one_residual**2,axis=1),np.finfo(float).tiny)
    two_rss=np.maximum(np.sum(two_residual**2,axis=1),np.finfo(float).tiny)
    bic_one=effective*np.log(one_rss/effective)+5.*np.log(effective)
    bic_two=effective*np.log(two_rss/effective)+10.*np.log(effective)
    from scipy.special import expit
    log_odds=.5*(bic_one-bic_two)
    if marginal_order:
        log_odds=evidence["log_evidence_two"]-evidence["log_evidence_one"]
    individual_probability=expit(log_odds)
    probability=expit(data["membership"]@log_odds)[data["run_index"]] if share_order else individual_probability
    representation=values+probability[:,None]*(conditional-values)
    representation*=data["mask"]
    final_rms=np.sqrt(np.sum(representation**2,axis=1)/data["count"])
    representation=np.divide(representation,final_rms[:,None],out=np.zeros_like(representation),where=final_rms[:,None]>0.)
    independent=values+individual_probability[:,None]*(conditional-values)
    independent_rms=np.sqrt(np.sum(independent**2,axis=1)/data["count"])
    independent=np.divide(independent,independent_rms[:,None],out=np.zeros_like(independent),where=independent_rms[:,None]>0.)
    if components is not None:
        component_conditional=evidence.pop("component_output")
        component_representation=component_values+probability[:,None,None]*(component_conditional-component_values)
        evidence["applied_components"]=np.divide(component_representation,final_rms[:,None,None],
            out=np.zeros_like(component_representation),where=final_rms[:,None,None]>0.)
    return representation,{"operator_input_rms":rms,"operator_output_rms":final_rms,
        "two_clock_probability":probability,"individual_probability":individual_probability,
        "independent_signal":independent,"noise_tau":tau,"noise_variance":fitted["noise_variance"],
        "periodic_variance":fitted["periodic_variance"],"two_clock_periods":fitted["two_periods"],
        "one_clock_period":fitted["one_period"],"two_clock_nfev":fitted["two_nfev"],
        "noise_geometry":noise_geometry,"bic_one":bic_one,"bic_two":bic_two}|evidence


def _peer_waveform(values,data):
    available=data["membership"]@data["mask"].astype(float)
    peer=((data["membership"]@values)[data["run_index"]]-values)/np.maximum(available[data["run_index"]]-data["mask"],1.)
    peer=detrend(peer,data["time"],data["mask"])
    return peer/np.sqrt(np.maximum(np.sum(peer**2,axis=1)/data["count"],1e-20))[:,None]


def _role_geometry(values,data,period,scale,peer_reference=None,cohort_scale=False,normalize_reference_peer=False):
    unit=detrend(values,data["time"],data["mask"])
    unit/=np.sqrt(np.maximum(np.sum(unit**2,axis=1)/data["count"],1e-20))[:,None]
    peer=_peer_waveform(unit,data) if peer_reference is None else _physical_peer(peer_reference,data)[1]
    if normalize_reference_peer:
        peer_rms=np.sqrt(np.sum(peer**2*data["mask"],axis=1)/data["count"])
        peer=np.divide(peer,peer_rms[:,None],out=np.zeros_like(peer),where=peer_rms[:,None]>0.)
    if cohort_scale:
        unit,peer=_physical_peer(values,data)
    profile=harmonic_profile(unit,data,period,scale,harmonics=1)
    other=harmonic_profile(peer,data,period,scale,harmonics=1)
    design=profile["design"]
    gram=profile["gram"]
    cross=np.einsum("lpti,lqtj->lpqij",design,design,optimize=True)
    shape=cross.shape
    joint=np.concatenate([np.concatenate([np.broadcast_to(gram[:,:,None],shape),cross],axis=-1),
        np.concatenate([cross.swapaxes(-1,-2),np.broadcast_to(gram[:,None],shape)],axis=-1)],axis=-2)
    return {"joint":joint,"rhs":np.stack([profile["rhs"],other["rhs"]],axis=-1),
        "layout":profile["layout"],"design":design,"unit":unit,"peer":peer}


def free_score_feature_bundle(frame,period_min=12.,period_max=48.,candidates=73,include_parent=False,include_controls=True):
    """Physical source views and the unchanged individual-anchor view stay explicit."""
    started=time.perf_counter()
    data=observations(frame)
    scale=np.sqrt(period_min*period_max)
    period=np.linspace(period_min,period_max,candidates)
    original,adjusted=shared_background(data,scale)
    canonical,diagnostic=canonical_source_signal(adjusted,data,period,scale)
    raw_geometry=_role_geometry(original,data,period,scale,peer_reference=original)
    base,raw_diagnostic=profile_features(original,data,period,scale,peer_reference=original)
    identity_base,_=profile_features(original,data,period,scale)
    bundle={"period":period,"period_scale":np.asarray(scale),"trace_id":data["metadata"].index.to_numpy(dtype=str),
        "run":data["metadata"].run.to_numpy(dtype=str),"condition":data["metadata"].condition.to_numpy(dtype=str),
        "run_index":data["run_index"],"run_names":data["runs"],"membership":data["membership"],"average":data["average"],
        "count":data["count"],"methods":METHODS,"time":data["time"],"mask":data["mask"],"base_features":base[:,:,:13],
        "identity_base_features":identity_base[:,:,:13],"source_operator":np.asarray(SOURCE_OPERATOR),
        "spectral_power":raw_diagnostic["power_one"],"peer_spectral_power":raw_diagnostic["peer_power"],
        "joint_gram":raw_geometry["joint"],"layout":raw_geometry["layout"],"harmonic_design":raw_geometry["design"],
        "original_unit":raw_geometry["unit"],"original_peer":raw_geometry["peer"],"original_rhs":raw_geometry["rhs"],
        "two_clock_probability":diagnostic["two_clock_probability"],"source_noise_tau":diagnostic["noise_tau"],
        "source_noise_variance":diagnostic["noise_variance"],"source_periodic_variance":diagnostic["periodic_variance"],
        "elapsed_s":np.asarray(time.perf_counter()-started)}
    waveforms={"source":canonical}
    reference={"source":adjusted}
    if include_controls:
        unadjusted,_=canonical_source_signal(original,data,period,scale)
        waveforms.update(unadjusted=unadjusted,unbalanced=adjusted,independent=diagnostic["independent_signal"])
        reference.update(unadjusted=original,unbalanced=adjusted,independent=adjusted)
    geometries=dict(zip(waveforms,map(lambda key:_role_geometry(waveforms[key],data,period,scale,peer_reference=reference[key]),waveforms)))
    bundle.update(dict(zip(map(lambda key:key+"_rhs",geometries),map(lambda value:value["rhs"],geometries.values()))))
    profiles=dict(zip(waveforms,map(lambda key:profile_features(waveforms[key],data,period,scale,peer_reference=reference[key])[0][:,:,:13],waveforms)))
    bundle.update(dict(zip(map(lambda key:key+"_base",profiles),profiles.values())))
    powers=dict(zip(geometries,map(lambda key:generic_coherent_fwl_power(data["time"],data["mask"],geometries[key]["unit"],geometries[key]["peer"],period,half_gram=True),geometries)))
    bundle.update(dict(zip(map(lambda key:key+"_coherent_half",powers),powers.values())))
    if include_controls:
        bundle["source_coherent_global"]=generic_coherent_fwl_power(data["time"],data["mask"],geometries["source"]["unit"],geometries["source"]["peer"],period,half_gram=False)
        bundle["original_coherent_half"]=generic_coherent_fwl_power(data["time"],data["mask"],raw_geometry["unit"],raw_geometry["peer"],period,half_gram=True)
    if include_parent:
        bundle["parent_coherent_power"]=generic_coherent_fwl_power(data["time"],data["mask"],raw_geometry["unit"],raw_geometry["peer"],period,half_gram=False)
    return bundle


def pooled_measurement_features(frame,period_min=12.,period_max=48.,candidates=73,include_anchor=True,include_controls=False,reference_period=None,background_reference=None,clock_cache=None,include_measurement=True,include_role_geometry=True):
    """Original FG measurement statistics on a single shared-gain source view."""
    data=observations(frame);period=np.linspace(period_min,period_max,candidates)
    scale=np.sqrt(period_min*period_max)
    if clock_cache is None:
        reference=_physical_peer(data["signal"],data)[0]
        clocks=_source_clock_fit(reference,data,period,scale)
        tau,noise=_source_noise_tau(clocks["two_residual"],data,scale)
        pair=pooled_clock_periods(data["signal"],data,period,scale,tau)
    else:
        order=pd.Index(clock_cache["trace_id"]).get_indexer(data["metadata"].index)
        assert np.all(order>=0) and str(clock_cache["source_operator"])==SOURCE_OPERATOR
        np.testing.assert_array_equal(clock_cache["period"],period)
        tau=clock_cache["raw_noise_tau"][order]
        first=np.unique(data["run_index"],return_index=True)[1]
        if "pool_period_pair_by_record" in clock_cache:
            pairs=clock_cache["pool_period_pair_by_record"][order][first]
            success=clock_cache["pool_fit_success_by_record"][order][first]
        else:
            group_order=pd.Index(clock_cache["run_names"]).get_indexer(data["runs"])
            assert np.all(group_order>=0)
            pairs=clock_cache["pool_period_pairs"][group_order]
            success=clock_cache["pool_period_fit_success"][group_order]
        pair={"period_pairs":pairs,"optimizer_success":success}
    balance=apply_pooled_clock_balance(data["signal"],data["time"],data["mask"],data["run_index"],pair["period_pairs"],tau)
    bundle={"period":period,"period_scale":np.asarray(scale),"trace_id":data["metadata"].index.to_numpy(dtype=str),
        "run":data["metadata"].run.to_numpy(dtype=str),"condition":data["metadata"].condition.to_numpy(dtype=str),
        "run_index":data["run_index"],"run_names":data["runs"],"membership":data["membership"],"average":data["average"],
        "count":data["count"],"time":data["time"],"mask":data["mask"],
        "raw_noise_tau":tau,"pool_period_pairs":pair["period_pairs"],"pool_gains":balance["gains"],
        "pool_period_pair_by_record":pair["period_pairs"][data["run_index"]],"pool_gain_by_record":balance["gains"][data["run_index"]],
        "pool_period_fit_success":pair["optimizer_success"],"pool_clock_rank":balance["clock_rank"],
        "pool_fit_success_by_record":pair["optimizer_success"][data["run_index"]],
        "duration_coordinate":_geometry_coordinate({"time":data["time"],"mask":data["mask"],"period_scale":np.asarray(scale)}),
        "source_operator":np.asarray(SOURCE_OPERATOR),"balanced_signal":balance["signal"]}
    if include_measurement:
        bundle["measurement_base_features"]=profile_features(balance["signal"],data,period,scale)[0]
    if include_anchor:
        original,_=shared_background(data,scale)
        bundle["identity_base_features"]=profile_features(original,data,period,scale)[0]
        bundle["original_peer"]=_physical_peer(original,data)[1]
    if include_controls:
        private=apply_pooled_clock_balance(data["signal"],data["time"],data["mask"],data["run_index"],pair["period_pairs"],tau,cellwise=True)
        bundle["cellwise_base_features"]=profile_features(private["signal"],data,period,scale)[0]
        if include_role_geometry:
            geometry=_role_geometry(balance["signal"],data,period,scale,cohort_scale=True)
            bundle.update({"joint_gram":geometry["joint"],"layout":geometry["layout"],"harmonic_design":geometry["design"],
                "pooled_rhs":geometry["rhs"],"pooled_unit":geometry["unit"],"pooled_peer":geometry["peer"]})
        if reference_period is not None:
            bundle.update(_emission_reference_statistics(balance["signal"],data,period,scale,reference_period,background_reference))
    return bundle


def feature_bundle(frame,period_min=12.,period_max=48.,candidates=73,include_parent=False,include_controls=True):
    return pooled_measurement_features(frame,period_min,period_max,candidates,include_anchor=True,include_controls=include_controls)


def _emission_reference_statistics(values,data,period,scale,foreground,background):
    unit,peer=_physical_peer(values,data)
    proxy=period[harmonic_profile(peer,data,period,scale,1)["energy_one"].argmax(1)]
    bg=proxy if background is None else np.where(np.isfinite(background),background,proxy)
    fg=np.where(np.isfinite(foreground)&(foreground>0.),foreground,scale)
    time=(data["time"]-data["time"][0])/scale;mask=data["mask"];n=data["count"]
    center=(time[None]-(mask*time[None]).sum(1)[:,None]/n[:,None])*mask
    trend=np.stack([mask/np.sqrt(n[:,None]),center/np.sqrt(np.sum(center**2,axis=1))[:,None]],axis=-1)
    angle=2.*np.pi*time[None,:,None]/(np.stack([fg,bg],axis=-1)/scale)[:,None]
    design=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(len(unit),len(time),4)*mask[:,:,None]
    design-=trend@(trend.transpose(0,2,1)@design)
    return {"emission_gram":design.transpose(0,2,1)@design,
        "emission_rhs":np.einsum("nti,nt->ni",design,unit,optimize=True),
        "emission_energy":np.sum(unit**2,axis=1),"emission_degrees":n-2,
        "emission_background_period":bg,
        "emission_background_is_proxy":np.ones(len(unit),dtype=bool) if background is None else ~np.isfinite(background)}


def source_reference_weights(bundle,context,training_run):
    """Unit, origin, declared view share, then distinct qualified FG source."""
    run=bundle["run_index"];member=bundle["membership"]
    q=context["qualified"];view_weight=bundle["average"]@context["view_weight"]
    origins,origin_names=pd.factorize(context["origin_run"],sort=True)
    origin_record=(np.arange(len(origin_names))[:,None]==origins[None]).astype(float)
    group_origin=(member@origin_record.T)>0
    assert np.all(group_origin.sum(1)==1)
    group_origin_code=group_origin.argmax(1)
    units,unit_names=pd.factorize(context["independent_unit"],sort=True)
    unit_record=(np.arange(len(unit_names))[:,None]==units[None]).astype(float)
    origin_unit=(origin_record@unit_record.T)>0
    assert np.all(origin_unit.sum(1)==1)
    origin_unit_code=origin_unit.argmax(1)
    nq=member@q.astype(float)
    group_weight=training_run*(nq>0.)[None]*view_weight[None]
    view_total=group_weight@group_origin
    share=group_weight/np.maximum(view_total[:,group_origin_code],np.finfo(float).tiny)
    eligible_origin=view_total>0.
    count_unit=eligible_origin.astype(float)@origin_unit
    effective_units=(count_unit>0.).sum(1)
    original_weight=eligible_origin/np.maximum(count_unit[:,origin_unit_code],1.)/effective_units[:,None]
    group_weight=share*original_weight[:,group_origin_code]
    identifiers=context.get("reference_source_id",context.get("origin_trace_id",bundle["trace_id"]))
    source_code,source_names=pd.factorize(identifiers,sort=True)
    source_record=(np.arange(len(source_names))[:,None]==source_code[None])
    source_in_group=(member@(source_record*q[None]).T)>0.
    source_number=source_in_group.sum(1)
    group_source_code=run*len(source_names)+source_code
    measurement_number=np.bincount(group_source_code,weights=q.astype(float),minlength=len(member)*len(source_names))[group_source_code]
    weight=group_weight[:,run]*q[None]/np.maximum(source_number[run],1.)[None]/np.maximum(measurement_number,1.)[None]
    train=training_run[:,run]&q[None]
    original_reference_count=(((train&context["is_original"][None]).astype(float)@source_record.T)>0.).sum(1)
    assert np.allclose(weight.sum(1),1.) and np.all(weight[~train]==0.)
    return {"training_weight":weight,"group_weight":group_weight,"train":train,
        "effective_sample_size":original_reference_count,"effective_training_units":effective_units}


def pooled_reference_measurement_features(path,context,period_min,period_max,candidates=73,include_controls=True):
    """Per-view extraction for the original 90-term source measurement model."""
    tags=np.unique(context["view_id"])
    record_keys=["trace_id","run","condition","count","measurement_base_features","pool_clock_rank","duration_coordinate",
        "raw_noise_tau","pool_period_pair_by_record","pool_gain_by_record"]
    if include_controls:
        record_keys+= ["identity_base_features","cellwise_base_features","emission_gram","emission_rhs","emission_energy","emission_degrees","emission_background_period","emission_background_is_proxy"]
    def view(tag):
        print("POOLED_REFERENCE_VIEW",str(tag),flush=True)
        frame=pd.read_parquet(path,columns=["trace_id","run","condition","x","signal_y"],filters=[("view_id","=",str(tag))])
        records=np.sort(frame.trace_id.unique());order=pd.Index(context["trace_id"]).get_indexer(records)
        current=pooled_measurement_features(frame,period_min,period_max,candidates,include_anchor=include_controls,include_controls=include_controls,
            reference_period=context["target"][order] if include_controls else None,
            background_reference=context["background_reference"][order] if include_controls else None,include_role_geometry=False)
        compact=dict(zip(record_keys,map(current.__getitem__,record_keys)))
        compact["pool_fit_success_by_record"]=current["pool_period_fit_success"][current["run_index"]]
        return compact
    parts=list(map(view,tags))
    keys=record_keys+["pool_fit_success_by_record"]
    bundle=dict(zip(keys,map(lambda key:np.concatenate(list(map(lambda part:part[key],parts)),axis=0),keys)))
    code,runs=pd.factorize(bundle["run"],sort=True)
    member=(np.arange(len(runs))[:,None]==code[None]).astype(float)
    bundle.update({"run_index":code,"run_names":np.asarray(runs,dtype=str),"membership":member,"average":member/member.sum(1)[:,None],
        "period":np.linspace(period_min,period_max,candidates),"period_scale":np.asarray(np.sqrt(period_min*period_max)),
        "source_operator":np.asarray("pooled_H2_gain_original_FG_measurement")})
    record_order=pd.Index(context["trace_id"]).get_indexer(bundle["trace_id"])
    run_order=pd.Index(context["run_name"]).get_indexer(bundle["run_names"])
    assert np.all(record_order>=0) and np.all(run_order>=0)
    selected=dict(zip(["target","qualified","origin_run","origin_trace_id","independent_unit","view_weight","is_original","background_reference"],
        map(lambda key:context[key][record_order],["target","qualified","origin_run","origin_trace_id","independent_unit","view_weight","is_original","background_reference"])))
    selected["nuisance_reference_kind"]=context["nuisance_reference_kind"]
    selected["training_run"]=np.concatenate([np.ones((1,len(runs)),dtype=bool),context["outer_train_run"][:,run_order]])
    return bundle,selected


def augment_reference_measurement_controls(path,context,bundle):
    """Add matched control coordinates while retaining cached main measurements."""
    tags=np.unique(context["view_id"]);period=bundle["period"]
    keys=["identity_base_features","cellwise_base_features","duration_coordinate","emission_gram","emission_rhs",
        "emission_energy","emission_degrees","emission_background_period","emission_background_is_proxy",
        "raw_noise_tau","pool_period_pair_by_record","pool_gain_by_record"]
    def view(tag):
        print("POOLED_REFERENCE_CONTROLS",str(tag),flush=True)
        frame=pd.read_parquet(path,columns=["trace_id","run","condition","x","signal_y"],filters=[("view_id","=",str(tag))])
        records=np.sort(frame.trace_id.unique());source_order=pd.Index(context["trace_id"]).get_indexer(records)
        cached_order=pd.Index(bundle["trace_id"]).get_indexer(records)
        assert np.all(source_order>=0) and np.all(cached_order>=0)
        cache=None
        if "raw_noise_tau" in bundle and "pool_period_pair_by_record" in bundle:
            cache={"trace_id":records,"period":period,"source_operator":bundle["source_operator"]}
            cache.update(dict(zip(["raw_noise_tau","pool_period_pair_by_record","pool_fit_success_by_record"],
                map(lambda key:bundle[key][cached_order],["raw_noise_tau","pool_period_pair_by_record","pool_fit_success_by_record"]))))
        current=pooled_measurement_features(frame,float(period.min()),float(period.max()),len(period),include_anchor=True,include_controls=True,
            reference_period=context["target"][source_order],background_reference=context["background_reference"][source_order],
            clock_cache=cache,include_measurement=False,include_role_geometry=False)
        return {"trace_id":current["trace_id"]}|dict(zip(keys,map(current.__getitem__,keys)))
    parts=list(map(view,tags));records=np.concatenate(list(map(lambda part:part["trace_id"],parts)))
    order=pd.Index(records).get_indexer(bundle["trace_id"])
    assert np.all(order>=0) and len(np.unique(order))==len(records)
    result=bundle.copy()
    result.update(dict(zip(keys,map(lambda key:np.concatenate(list(map(lambda part:part[key],parts)),axis=0)[order],keys))))
    return result


def _polynomial(base):
    left,right=np.triu_indices(base.shape[-1])
    return np.concatenate([base,base[:,:,left]*base[:,:,right]],axis=-1)


def _background_features(base):
    """Five physical-peer statistics and their fifteen quadratic products."""
    return _polynomial(base[:,:,4:9])


def measured_mixing_quality(data,period,scale):
    """Observed cohort periodic energy, with no labels or acquisition categories."""
    _,peer=_physical_peer(data["signal"],data)
    rms=np.sqrt(np.sum(peer**2*data["mask"],axis=1)/data["count"])
    peer=np.divide(peer,rms[:,None],out=np.zeros_like(peer),where=rms[:,None]>0.)
    profile=harmonic_profile(peer,data,period,scale,harmonics=1)
    energy=profile["energy_one"]/data["count"][:,None]
    quality=np.max(data["average"]@energy,axis=1)
    return np.clip(quality,0.,1.)[data["run_index"]]


def quality_design(features,quality=None):
    """One continuous score law over observed periodic and residual energy."""
    if quality is None:
        return features
    shape=(1,)*(features.ndim-3)+(len(quality),1,1)
    fraction=np.asarray(quality).reshape(shape)
    return np.concatenate([features*(1.-fraction),features*fraction],axis=-1)


def source_core_design(bundle,base,quality=None,coherent=False,paired=False):
    primary=_polynomial(base)
    if not paired:
        return quality_design(primary,quality)
    companion=bundle["companion_base_features"]
    if coherent:
        companion=companion.copy()
        companion[:,:,11]=bundle["companion_coherent_power"]
    return np.concatenate([primary,_polynomial(companion)],axis=-1)


def conditional_prior_transform(features,quality,weight,scale,transform,n,units):
    """Records identify the shared score; independent units identify its change."""
    dimension=features.shape[-1]//2
    values=np.broadcast_to(features,(len(weight),*features.shape[-3:]))
    original=values[...,:dimension]+values[...,dimension:]
    mean_period=original.mean(2);square_period=(original**2).mean(2)
    center=weight@quality
    difference=quality[None]-center[:,None]
    mean=np.einsum("fn,fnd->fd",weight,mean_period,optimize=True)
    variance=np.einsum("fn,fnd->fd",weight,square_period,optimize=True)-mean**2
    interaction=np.einsum("fn,fnd->fd",weight*difference,mean_period,optimize=True)
    interaction_variance=np.einsum("fn,fnd->fd",weight*difference**2,square_period,optimize=True)-interaction**2
    shared_sd=np.sqrt(np.maximum(variance,0.));interaction_sd=np.sqrt(np.maximum(interaction_variance,0.))
    low=np.min(np.where(weight>0.,quality[None],np.inf),axis=1)
    high=np.max(np.where(weight>0.,quality[None],-np.inf),axis=1)
    shared_sd=np.where(shared_sd>0.,shared_sd,1.)/np.sqrt(n[:,None])
    interaction_sd=np.where((interaction_sd>0.)&(high>low)[:,None],interaction_sd,1.)/np.sqrt(units[:,None])
    identity=np.eye(dimension)[None]
    upper=np.concatenate([identity*(shared_sd*(1.-center[:,None]))[:,:,None],identity*(shared_sd*center[:,None])[:,:,None]],axis=2)
    lower=np.concatenate([-identity*interaction_sd[:,:,None],identity*interaction_sd[:,:,None]],axis=2)
    penalty=np.concatenate([upper,lower],axis=1)
    return penalty@(transform/scale[:,:,None])


def source_role_evidence(bundle,covariance,background,rhs=None,reduction="expected_bf",correlated=False,quality=None):
    from source_assignment_core import role_assignment_evidence,conditional_role_assignment_evidence
    values=bundle["source_rhs"] if rhs is None else rhs
    if covariance.shape[-2]>values.shape[-1]:
        values=np.concatenate([values,bundle["companion_rhs"][:,:,:,None]],axis=-1)
    middle=len(bundle["period"])//2
    if correlated:
        from source_correlated_geometry import correlated_role_assignment_evidence
        if covariance.ndim==4:
            q=np.asarray(quality,dtype=float)
            level,owner=np.unique(q,return_inverse=True)
            role=np.empty((len(covariance),len(q),len(bundle["period"]),2))
            def conditioned(number):
                selected=np.flatnonzero(owner==number)
                current=np.exp((1.-level[number])*np.log(covariance[:,0])+level[number]*np.log(covariance[:,1]))
                measured,_=correlated_role_assignment_evidence(bundle["joint_gram"],values[selected],bundle["layout"][selected],
                    current,background[:,selected],middle,middle)
                role[:,selected]=measured
                return None
            list(map(conditioned,np.arange(len(level))))
            return role,{}
        return correlated_role_assignment_evidence(bundle["joint_gram"],values,bundle["layout"],covariance,background,middle,middle)
    if covariance.ndim==4:
        return conditional_role_assignment_evidence(bundle["joint_gram"],values,bundle["layout"],covariance,
            bundle["mixing_quality"],background,middle,middle,reduction=reduction)
    return role_assignment_evidence(bundle["joint_gram"],values,bundle["layout"],covariance,background,middle,middle,reduction=reduction)


def model_role_quality(bundle,model):
    roles=model["role_covariance"].shape[-2]
    dimension=roles*(bundle["source_base"].shape[-1]+1)+roles*(roles+1)//2
    return bundle["mixing_quality"] if model["role_coefficient"].shape[-1]==2*dimension else None


def foreground_reference_log_prior(bundle,model,cases):
    """Maximum-entropy foreground base measure from training references only."""
    axis=bundle["period"]/float(bundle["period_scale"])
    if str(model.get("foreground_prior_kind","reference"))=="uniform" or "foreground_reference_mean" not in model:
        return np.zeros((len(cases),len(axis)))
    mean=model["foreground_reference_mean"][cases]
    sd=model["foreground_reference_sd"][cases]
    score=-.5*((axis[None]-mean[:,None])/sd[:,None])**2
    return score-logsumexp(score,axis=-1,keepdims=True)


def model_role_evidence(bundle,model,background,cases):
    """Physical source-role evidence and its matched period-density control."""
    if str(model.get("role_kind","gaussian"))!="density":
        return source_role_evidence(bundle,model["role_covariance"][cases],background,reduction=str(model.get("role_reduction","expected_bf")))
    base=bundle["source_base"];coherent=base.copy();coherent[:,:,11]=bundle["source_coherent_power"]
    quality=model_role_quality(bundle,model)
    paired=str(model.get("core_measurements","single"))=="paired"
    score=foreground_reference_log_prior(bundle,model,cases)[:,None]+np.einsum("npd,fd->fnp",
        source_core_design(bundle,coherent,quality,coherent=True,paired=paired),model["core_coefficient"][cases],optimize=True)
    if model["role_covariance"].shape[-2]==3:
        dimension=model["core_coefficient"].shape[-1]//2
        primary=foreground_reference_log_prior(bundle,model,cases)[:,None]+np.einsum("npd,fd->fnp",_polynomial(coherent),model["core_coefficient"][cases,:dimension],optimize=True)
        companion=bundle["companion_base_features"].copy();companion[:,:,11]=bundle["companion_coherent_power"]
        secondary=foreground_reference_log_prior(bundle,model,cases)[:,None]+np.einsum("npd,fd->fnp",_polynomial(companion),model["core_coefficient"][cases,dimension:],optimize=True)
        role=np.stack([primary-logsumexp(primary,axis=-1,keepdims=True),np.log(np.maximum(background,np.finfo(float).tiny)),secondary-logsumexp(secondary,axis=-1,keepdims=True)],axis=-1)
    else:
        role=np.stack([score-logsumexp(score,axis=-1,keepdims=True),np.log(np.maximum(background,np.finfo(float).tiny))],axis=-1)
    return role,np.full((len(cases),3),np.nan)


def _fit_score(features,target,qualified,train,period,columns=None,parent=None,ordered=False,ranking=False,rivals=None,maxiter=900,training_weight=None,effective_sample_size=None,initial_coefficient=None,cohort_moments=None,ranking_without_prior=False,prior_scale_weight=None,quality=None,interaction_units=None,prior_multiplier=1.,paired_moments=None):
    """Original proper score and rival losses, with analytic gradients and whitening."""
    indices=np.flatnonzero(np.any(features!=0,axis=(0,1))) if columns is None else np.asarray(columns)
    x=features[:,:,indices]
    n=train.sum(1) if effective_sample_size is None else np.asarray(effective_sample_size,dtype=float)
    weight=train/train.sum(1)[:,None] if training_weight is None else np.asarray(training_weight,dtype=float)
    scale_weight=weight if prior_scale_weight is None else prior_scale_weight
    mean=np.einsum("fn,npd->fd",scale_weight,x,optimize=True)/len(period)
    scale=np.sqrt(np.maximum(np.einsum("fn,npd->fd",scale_weight,x*x,optimize=True)/len(period)-mean**2,0.))
    scale=np.where(scale>0.,scale,1.)
    centered=x-x.mean(1,keepdims=True)
    covariance=np.einsum("fn,nde->fde",weight,np.einsum("npd,npe->nde",centered,centered,optimize=True)/len(period),optimize=True)/(scale[:,:,None]*scale[:,None,:])
    covariance+=np.eye(len(indices))[None]/n[:,None,None]
    eig,vec=np.linalg.eigh(covariance)
    transform=np.einsum("fik,fk,fjk->fij",vec,1/np.sqrt(np.maximum(eig,1e-12)),vec,optimize=True)
    prior_transform=np.sqrt(prior_multiplier)*transform/np.sqrt(n[:,None,None])
    if quality is not None:
        prior_transform=conditional_prior_transform(x,quality,scale_weight,scale,transform,n,interaction_units)
    teacher=np.maximum(1-abs(period[None]-target[:,None])/(period[1]-period[0]),0)*qualified[:,None]
    teacher/=np.maximum(teacher.sum(1),1e-12)[:,None]
    sd=np.sqrt(np.maximum(weight@(target**2)-(weight@target)**2,(period[1]-period[0])**2/12))
    fixed=np.zeros((len(train),len(target),len(period))) if parent is None else parent
    cdf=teacher.cumsum(-1)[:,:-1]
    negative=np.zeros((len(target),2),dtype=bool) if rivals is None else teacher[np.arange(len(target))[:,None],rivals]==0
    negative_n=np.maximum(negative.sum(1),1)

    def objective(parameter):
        coordinates=parameter.reshape(scale.shape)
        theta=np.einsum("fij,fj->fi",transform,coordinates,optimize=True)
        prior_residual=np.einsum("fij,fj->fi",prior_transform,coordinates,optimize=True)
        score=fixed+np.einsum("npd,fd->fnp",x,theta/scale,optimize=True)
        log=score-logsumexp(score,axis=-1,keepdims=True)
        mass=np.exp(log)
        value=-np.sum(log*teacher[None]*weight[:,:,None])+.5*np.sum(prior_residual**2)
        derivative=mass-teacher[None]
        if ordered:
            error=mass.cumsum(-1)[:,:,:-1]-cdf[None]
            value+=np.sum(error**2*weight[:,:,None]*(period[1]-period[0])/sd[:,None,None])
            partial=np.pad(2*(period[1]-period[0])*np.cumsum(error[:,:,::-1],axis=-1)[:,:,::-1]/sd[:,None,None],((0,0),(0,0),(0,1)))
            derivative+=mass*(partial-(partial*mass).sum(-1)[:,:,None])
        if ranking:
            ranked=score-fixed if ranking_without_prior else score
            positive=(ranked*teacher[None]).sum(-1)
            rival=np.take_along_axis(ranked,np.broadcast_to(rivals[None],(len(train),)+rivals.shape),axis=2)
            error=rival-positive[:,:,None]
            value+=np.sum(np.logaddexp(0,error)*negative[None]/negative_n[None,:,None]*weight[:,:,None])
            amount=(1/(1+np.exp(np.clip(-error,-700,700))))*negative[None]/negative_n[None,:,None]
            derivative-=teacher[None]*amount.sum(-1)[:,:,None]
            np.add.at(derivative,(np.arange(len(train))[:,None,None],np.arange(len(target))[None,:,None],rivals[None]),amount)
        weighted_derivative=derivative*weight[:,:,None]
        if cohort_moments is not None:
            individual=mass@period
            group=individual@cohort_moments["average"].T
            error=group-cohort_moments["reference"][None]
            precision=cohort_moments["weight"]/cohort_moments["variance"][None]
            value+=.5*np.sum(precision*error**2)
            mean_derivative=(precision*error)@cohort_moments["average"]
            weighted_derivative+=mean_derivative[:,:,None]*mass*(period[None,None]-individual[:,:,None])
        if paired_moments is not None:
            individual=mass@period
            group=individual@paired_moments["average"].T
            left=paired_moments["left"];right=paired_moments["right"]
            error=group[:,right]-group[:,left]-paired_moments["effect"][None]
            precision=paired_moments["weight"]/paired_moments["variance"]
            value+=.5*np.sum(precision*error**2)
            group_derivative=np.zeros_like(group)
            np.add.at(group_derivative,(np.arange(len(train))[:,None],right[None]),precision*error)
            np.add.at(group_derivative,(np.arange(len(train))[:,None],left[None]),-precision*error)
            mean_derivative=group_derivative@paired_moments["average"]
            weighted_derivative+=mean_derivative[:,:,None]*mass*(period[None,None]-individual[:,:,None])
        grad=np.einsum("fnp,npd->fd",weighted_derivative,x,optimize=True)/scale
        gradient=np.einsum("fji,fj->fi",transform,grad,optimize=True)+np.einsum("fji,fj->fi",prior_transform,prior_residual,optimize=True)
        return float(value),gradient.ravel()

    initial=np.zeros(scale.size) if initial_coefficient is None else np.linalg.solve(transform,(initial_coefficient[:,indices]*scale)[:,:,None])[:,:,0].ravel()
    fitted=minimize(objective,initial,jac=True,method="L-BFGS-B",
        options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-4,"maxcor":25})
    coefficient=np.einsum("fij,fj->fi",transform,fitted.x.reshape(scale.shape),optimize=True)/scale
    output=np.zeros((len(train),features.shape[-1]))
    output[:,indices]=coefficient
    print("UNIFIED_SCORE_FIT",len(indices),ordered,ranking,fitted.success,fitted.nit,float(np.max(abs(fitted.jac))),flush=True)
    return output,{"success":bool(fitted.success),"iterations":fitted.nit,"gradient":float(np.max(abs(fitted.jac)))}


def _fit_core(base,target,qualified,train,period,maxiter,log_prior,return_phase=False,quality=None,interaction_units=None,reference_count=None,companion_base=None,initial_core=None,training_weight=None,sequential_reference=False):
    x=quality_design(_polynomial(base),quality) if companion_base is None else np.concatenate([_polynomial(base),_polynomial(companion_base)],axis=-1)
    left,right=np.triu_indices(13)
    phase=np.concatenate([np.arange(13)==12,(left==12)|(right==12)])
    if quality is not None or companion_base is not None:
        phase=np.tile(phase,2)
    prior=np.broadcast_to(log_prior[:,None],(len(train),len(target),len(period)))
    options={"quality":quality,"interaction_units":interaction_units,"effective_sample_size":reference_count,
        "prior_multiplier":2. if companion_base is not None else 1.,"training_weight":training_weight}
    likelihood_status=[]
    if sequential_reference:
        likelihood,likelihood_fit=_fit_score(x,target,qualified,train,period,np.flatnonzero(~phase),parent=prior,ordered=False,maxiter=maxiter,**options)
        initial_core=likelihood
        likelihood_status=[likelihood_fit]
    anchor,s1=_fit_score(x,target,qualified,train,period,np.flatnonzero(~phase),parent=prior,ordered=True,maxiter=maxiter,initial_coefficient=initial_core,**options)
    fixed=prior+np.einsum("npd,fd->fnp",x,anchor,optimize=True)
    signed,s2=_fit_score(x,target,qualified,train,period,np.flatnonzero(phase),parent=fixed,ordered=True,maxiter=maxiter,**options)
    parent=anchor+signed
    fixed=prior+np.einsum("npd,fd->fnp",x,parent,optimize=True)
    own_rival=base[:,:,0].argmax(1)
    peer_rival=np.where(np.any(base[:,:,4]!=0,axis=1),base[:,:,4].argmax(1),own_rival)
    rivals=np.stack([own_rival,peer_rival],axis=-1)
    refinement,s3=_fit_score(x,target,qualified,train,period,parent=fixed,ordered=True,ranking=True,rivals=rivals,maxiter=maxiter,**options)
    result=(anchor,parent+refinement,likelihood_status+[s1,s2,s3])
    return result+(parent,) if return_phase else result


def _fit_covariance(bundle,target,qualified,train,background_index,device,maxiter,quality=None,interaction_units=None,reference_count=None,initial_covariance=None,training_weight=None):
    """Fit the original two-view Gaussian amplitude/noise law on allowed references."""
    scale=float(bundle["period_scale"])
    t=(bundle["time"]-bundle["time"][0])/scale
    mask=bundle["mask"]
    count=bundle["count"]
    centered=(t[None]-(mask*t[None]).sum(1)[:,None]/count[:,None])*mask
    trend=np.stack([mask/np.sqrt(count[:,None]),centered/np.sqrt((centered**2).sum(1))[:,None]],axis=-1)
    reference=np.where(qualified,target/scale,1.)
    angle=2*np.pi*t[None]/reference[:,None]
    foreground=np.stack([np.cos(angle),np.sin(angle)],axis=-1)*mask[:,:,None]
    foreground-=np.einsum("nti,nij->ntj",trend,np.einsum("nti,ntj->nij",trend,foreground,optimize=True),optimize=True)
    background=bundle["harmonic_design"][bundle["layout"],background_index]
    design=np.concatenate([foreground,background],axis=-1)
    gram=np.einsum("nti,ntj->nij",design,design,optimize=True)
    signals=np.stack([bundle["original_unit"],bundle["original_peer"]])
    rhs=np.einsum("nti,vnt->vni",design,signals,optimize=True)
    dtype=torch.float64
    tg=torch.as_tensor(gram,dtype=dtype,device=device)
    tr=torch.as_tensor(rhs,dtype=dtype,device=device)
    te=torch.as_tensor(np.sum(signals**2,axis=-1),dtype=dtype,device=device)
    tc=torch.as_tensor(count-2,dtype=dtype,device=device)
    tw=torch.as_tensor(train/train.sum(1)[:,None] if training_weight is None else training_weight,dtype=dtype,device=device)
    tn=torch.as_tensor(train.sum(1) if reference_count is None else reference_count,dtype=dtype,device=device)
    shape=(len(train),2,3) if quality is None else (len(train),2,2,3)
    if quality is not None:
        tq=torch.as_tensor(np.column_stack([1.-quality,quality]),dtype=dtype,device=device)
        t_units=torch.as_tensor(interaction_units,dtype=dtype,device=device)
        quality_center=(tw*tq[None,:,1]).sum(-1)
        quality_variance=(tw*tq[None,:,1].square()).sum(-1)-quality_center.square()
        quality_low=torch.where(tw>0.,tq[None,:,1],torch.inf).amin(-1)
        quality_high=torch.where(tw>0.,tq[None,:,1],-torch.inf).amax(-1)
        quality_variance=torch.where((quality_variance>0.)&(quality_high>quality_low),quality_variance,torch.ones_like(quality_variance))

    def objective(parameter):
        log=parameter.reshape(shape)
        if quality is not None:
            current=torch.einsum("nc,fcvk->fvnk",tq,log)
            h=current.exp()
            sd=torch.repeat_interleave(h[:,:,:,:2],2,dim=-1).sqrt()
            precision=torch.eye(4,dtype=dtype,device=device)[None,None,None]+tg[None,None]*sd[:,:,:,:,None]*sd[:,:,:,None,:]/h[:,:,:,2,None,None]
            scaled=tr[None]*sd/h[:,:,:,2,None]
            gain=(scaled*torch.linalg.solve(precision,scaled[:,:,:,:,None])[:,:,:,:,0]).sum(-1)
            nll=.5*(tc[None,None]*current[:,:,:,2]+torch.linalg.slogdet(precision)[1]+te[None]/h[:,:,:,2]-gain)
            shared=(1.-quality_center[:,None,None])*log[:,0]+quality_center[:,None,None]*log[:,1]
            change=log[:,1]-log[:,0]
            penalty=shared.square().sum((1,2))/(2*tn)+change.square().sum((1,2))*quality_variance/(2*t_units)
            return (nll*tw[:,None]).sum()+penalty.sum()
        h=log.exp()
        sd=torch.repeat_interleave(h[:,:,:2],2,dim=-1).sqrt()
        precision=torch.eye(4,dtype=dtype,device=device)[None,None,None]+tg[None,None]*sd[:,:,None,:,None]*sd[:,:,None,None,:]/h[:,:,2,None,None,None]
        scaled=tr[None]*sd[:,:,None]/h[:,:,2,None,None]
        gain=(scaled*torch.linalg.solve(precision,scaled[:,:,:,:,None])[:,:,:,:,0]).sum(-1)
        nll=.5*(tc[None,None]*log[:,:,2,None]+torch.linalg.slogdet(precision)[1]+te[None]/h[:,:,2,None]-gain)
        return ((nll*tw[:,None]).sum(-1)+log.square().sum(-1)/(2*tn[:,None])).sum()

    gradient=torch.func.grad_and_value(objective)
    scipy_objective=lambda parameter:(pair:=gradient(torch.as_tensor(parameter,dtype=dtype,device=device)),(pair[1].item(),pair[0].detach().cpu().numpy()))[-1]
    initial=np.zeros(np.prod(shape)) if initial_covariance is None else np.log(initial_covariance).ravel()
    fitted=minimize(scipy_objective,initial,jac=True,method="L-BFGS-B",options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-4,"maxcor":25})
    print("UNIFIED_COVARIANCE",fitted.success,fitted.nit,float(np.max(abs(fitted.jac))),flush=True)
    return np.exp(fitted.x.reshape(shape)),{"success":bool(fitted.success),"iterations":fitted.nit,"gradient":float(np.max(abs(fitted.jac)))}


def _fit_grid_source_covariance(bundle,target,qualified,train,background_index,device,maxiter,quality,reference_count):
    """Calibrate raw, peer and balanced roles on the same period grid."""
    axis=bundle["period"]/float(bundle["period_scale"])
    position=(np.where(qualified,target/float(bundle["period_scale"]),axis[0])-axis[0])/(axis[1]-axis[0])
    lower=np.clip(np.floor(position).astype(int),0,len(axis)-1);upper=np.minimum(lower+1,len(axis)-1)
    fraction=np.clip(position-lower,0.,1.)
    foreground=np.column_stack([lower,upper]);teacher=np.column_stack([1.-fraction,fraction])
    rhs=np.concatenate([bundle["source_rhs"],bundle["companion_rhs"][:,:,:,None]],axis=-1)
    row=np.arange(len(target))
    gram=bundle["joint_gram"][bundle["layout"][:,None],foreground,background_index[:,None]]
    foreground_rhs=rhs[row[:,None],foreground]
    background_rhs=np.broadcast_to(rhs[row,background_index][:,None],foreground_rhs.shape)
    sufficient=np.concatenate([foreground_rhs,background_rhs],axis=2).transpose(3,0,1,2)
    views=rhs.shape[-1];dtype=torch.float64
    tg=torch.as_tensor(gram,dtype=dtype,device=device);tr=torch.as_tensor(sufficient,dtype=dtype,device=device)
    tc=torch.as_tensor(bundle["count"]-2,dtype=dtype,device=device)
    te=torch.as_tensor(bundle["count"],dtype=dtype,device=device)
    tw=torch.as_tensor(train/train.sum(1)[:,None],dtype=dtype,device=device)
    tt=torch.as_tensor(teacher,dtype=dtype,device=device);tn=torch.as_tensor(reference_count,dtype=dtype,device=device)
    shape=(len(train),views,3) if quality is None else (len(train),2,views,3)
    if quality is not None:
        tq=torch.as_tensor(np.column_stack([1.-quality,quality]),dtype=dtype,device=device)
        center=(tw*tq[None,:,1]).sum(-1)
        variance=(tw*(tq[None,:,1]-center[:,None]).square()).sum(-1)
        variance=torch.where(variance>0.,variance,torch.ones_like(variance))
    def objective(parameter):
        log=parameter.reshape(shape)
        if quality is None:
            current=log[:,:,None].expand(-1,-1,len(target),-1)
            penalty=log.square().sum((1,2))/(2*tn)
        else:
            current=torch.einsum("nc,fcvk->fvnk",tq,log)
            shared=(1.-center[:,None,None])*log[:,0]+center[:,None,None]*log[:,1]
            change=log[:,1]-log[:,0]
            penalty=(shared.square().sum((1,2))+change.square().sum((1,2))*variance)/(2*tn)
        h=current.exp();sd=torch.repeat_interleave(h[:,:,:,:2],2,dim=-1).sqrt();noise=h[:,:,:,-1]
        precision=torch.eye(4,dtype=dtype,device=device)[None,None,None,None]+tg[None,None]*sd[:,:,:,None,:,None]*sd[:,:,:,None,None,:]/noise[:,:,:,None,None,None]
        scaled=tr[None]*sd[:,:,:,None]/noise[:,:,:,None,None]
        gain=(scaled*torch.linalg.solve(precision,scaled[:,:,:,:, :,None])[:,:,:,:,:,0]).sum(-1)
        nll=.5*(tc[None,None,:,None]*current[:,:,:,-1,None]+torch.linalg.slogdet(precision)[1]+te[None,None,:,None]/noise[:,:,:,None]-gain)
        return (nll*tt[None,None]*tw[:,None,:,None]).sum()+penalty.sum()
    gradient=torch.func.grad_and_value(objective)
    scipy_objective=lambda parameter:(pair:=gradient(torch.as_tensor(parameter,dtype=dtype,device=device)),(pair[1].item(),pair[0].detach().cpu().numpy()))[-1]
    fit=minimize(scipy_objective,np.zeros(np.prod(shape)),jac=True,method="L-BFGS-B",options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-4,"maxcor":25})
    print("GRID_SOURCE_COVARIANCE",views,bool(fit.success),fit.nit,float(np.max(abs(fit.jac))),flush=True)
    return np.exp(fit.x.reshape(shape)),{"success":bool(fit.success),"iterations":fit.nit,"gradient":float(np.max(abs(fit.jac)))}


def _base_variant(bundle,variant):
    base=bundle["base_features"].copy()
    if variant==2:
        base[:,:,4:13]=0.
    return base


def _identity_base_variant(bundle,variant):
    base=bundle["identity_base_features"].copy()
    if variant==2:
        base[:,:,4:13]=0.
    return base


def _reference_prior(target,qualified,train,period,unit_membership):
    """A training-only source-period law; bandwidth counts independent units."""
    allowed=train&qualified[None]
    count=allowed.sum(1)
    weight=allowed/count[:,None]
    center=weight@target
    variance=np.maximum(weight@(target**2)-center**2,(period[1]-period[0])**2/12)
    units=(allowed.astype(float)@unit_membership.T)>0
    independent=units.sum(1)
    bandwidth=np.maximum(np.sqrt(variance)*(4/(3*independent))**.2,(period[1]-period[0])/np.sqrt(12))
    log_weight=np.full(allowed.shape,-np.inf)
    np.log(weight,out=log_weight,where=allowed)
    score=logsumexp(log_weight[:,:,None]-.5*((period[None,None]-target[None,:,None])/bandwidth[:,None,None])**2,axis=1)
    return score-logsumexp(score,axis=1,keepdims=True),independent


def cohort_reference_prior(bundle,reference,qualified,training_run,independent_unit):
    """Training-unit-balanced population-location and within-cohort priors."""
    axis=bundle["period"]/float(bundle["period_scale"])
    target=np.where(qualified,reference/float(bundle["period_scale"]),0.)
    membership=bundle["membership"];run=bundle["run_index"]
    number=membership@qualified.astype(float)
    mean=np.divide(membership@target,number,out=np.zeros(len(number)),where=number>0.)
    code,units=pd.factorize(independent_unit,sort=True)
    unit_record=(np.arange(len(units))[:,None]==code[None]).astype(float)
    group_unit=(membership@unit_record.T)>0
    assert np.all(group_unit.sum(1)==1)
    group_code=group_unit.argmax(1)
    eligible=training_run&(number>0.)[None]
    number_in_unit=eligible.astype(float)@group_unit
    effective_units=(number_in_unit>0.).sum(1)
    weight=eligible/np.maximum(number_in_unit[:,group_code],1.)/effective_units[:,None]
    center=weight@mean
    between=np.maximum(weight@(mean**2)-center**2,0.)
    bandwidth=np.maximum(np.sqrt(between)*(4./(3.*effective_units))**.2,(axis[1]-axis[0])/np.sqrt(12.))
    log_weight=np.full(weight.shape,-np.inf);np.log(weight,out=log_weight,where=weight>0.)
    score=logsumexp(log_weight[:,:,None]-.5*((axis[None,None]-mean[None,:,None])/bandwidth[:,None,None])**2,axis=1)
    if "integration_weight" in bundle:
        score+=np.log(bundle["integration_weight"])[None]
    mean_log_prior=score-logsumexp(score,axis=-1,keepdims=True)
    centered=(target-mean[run])*qualified
    within_sum=membership@(centered**2)
    variance=np.divide(within_sum,number-1.,out=np.zeros_like(number),where=number>1.)
    variance_eligible=training_run&(number>1.)[None]
    variance_in_unit=variance_eligible.astype(float)@group_unit
    variance_units=(variance_in_unit>0.).sum(1)
    assert np.all(variance_units>0)
    variance_weight=variance_eligible/np.maximum(variance_in_unit[:,group_code],1.)/variance_units[:,None]
    within_sd=np.sqrt(np.maximum(variance_weight@variance,0.))
    return {"mean_log_prior":mean_log_prior,"foreground_within_sd":within_sd,
        "effective_training_units":effective_units,"reference_group_mean":mean,
        "reference_group_count":number,"mean_prior_weight":weight,"mean_prior_bandwidth":bandwidth}


def joint_cohort_period_posterior(log_likelihood,period_axis,run_index,mean_log_prior,within_sd,integration_weight=None):
    """Infer population location and finite-cohort periods with one mean prior.

    period_axis and within_sd use the same units (normally period/period_scale).
    A zero within spread is an exact point mass, with no artificial variance.
    """
    log_likelihood=np.asarray(log_likelihood,dtype=float)
    axis=np.asarray(period_axis,dtype=float);run_index=np.asarray(run_index,dtype=int)
    prior=np.asarray(mean_log_prior,dtype=float)
    within=np.asarray(within_sd,dtype=float)
    assert log_likelihood.ndim==3 and prior.shape==(len(log_likelihood),len(axis))
    assert within.shape==(len(log_likelihood),) and np.all(np.isfinite(within)) and np.all(within>=0.)
    prior=prior-logsumexp(prior,axis=-1,keepdims=True)
    if np.all(within==0.):
        groups=int(run_index.max())+1
        member=(np.arange(groups)[:,None]==run_index[None]).astype(float)
        log_mean=prior[:,None]+np.einsum("gn,fnp->fgp",member,log_likelihood,optimize=True)
        log_mean-=logsumexp(log_mean,axis=-1,keepdims=True)
        mean_mass=np.exp(log_mean)
        mean=mean_mass@axis
        individual=mean[:,run_index]
        return {"mean_probability":mean_mass,"period_probability":mean_mass[:,run_index],
            "population_location":mean,"finite_cohort_mean":mean,"period_mean":individual,
            "deviation":np.zeros_like(individual)}
    difference=axis[:,None]-axis[None]
    precision=np.divide(1.,within**2,out=np.zeros_like(within),where=within>0.)
    transition=-.5*difference[None]**2*precision[:,None,None]
    if integration_weight is not None:
        transition+=np.log(integration_weight)[None,:,None]
    transition-=logsumexp(transition,axis=1,keepdims=True)
    point=np.full((len(axis),len(axis)),-np.inf);np.fill_diagonal(point,0.)
    transition=np.where((within==0.)[:,None,None],point[None],transition)
    ngroup=int(run_index.max())+1
    mean_probability=np.empty((len(prior),ngroup,len(axis)))
    period_probability=np.empty_like(log_likelihood)
    population_location=np.empty((len(prior),ngroup))
    finite_mean=np.empty_like(population_location)
    period_mean=np.empty(log_likelihood.shape[:2])

    def cohort(group):
        selected=np.flatnonzero(run_index==group)
        local=log_likelihood[:,selected]
        integrated=logsumexp(local[:,:,:,None]+transition[:,None],axis=2)
        log_mean=prior+integrated.sum(1)
        log_mean-=logsumexp(log_mean,axis=-1,keepdims=True)
        mean_mass=np.exp(log_mean)
        cavity=log_mean[:,None]-integrated
        conditional=logsumexp(cavity[:,:,None]+transition[:,None],axis=-1)
        log_period=local+conditional
        log_period-=logsumexp(log_period,axis=-1,keepdims=True)
        mass=np.exp(log_period)
        mass=np.where((within==0.)[:,None,None],mean_mass[:,None],mass)
        location=mean_mass@axis
        individual=mass@axis
        individual=np.where((within==0.)[:,None],location[:,None],individual)
        finite=np.where(within==0.,location,individual.mean(1))
        mean_probability[:,group]=mean_mass
        period_probability[:,selected]=mass
        population_location[:,group]=location
        finite_mean[:,group]=finite
        period_mean[:,selected]=individual
        return None

    list(map(cohort,np.arange(ngroup)))
    deviation=period_mean-finite_mean[:,run_index]
    deviation=np.where((within==0.)[:,None],0.,deviation)
    return {"mean_probability":mean_probability,"period_probability":period_probability,
        "population_location":population_location,"finite_cohort_mean":finite_mean,
        "period_mean":period_mean,"deviation":deviation}


def source_period_grid(time,mask,period_min,period_max,harmonics,oversampling=4.):
    """Resolve the fastest retained harmonic over the longest observed window."""
    scale=np.sqrt(period_min*period_max)
    span=np.max(np.where(mask,np.asarray(time)[None],-np.inf),axis=1)-np.min(np.where(mask,np.asarray(time)[None],np.inf),axis=1)
    normalized_span=span.max()/scale
    frequency_span=scale/period_min-scale/period_max
    candidates=int(np.ceil(frequency_span*oversampling*harmonics*normalized_span))+1
    frequency=np.linspace(scale/period_min,scale/period_max,candidates)
    period=scale/frequency
    step=np.diff(period)
    weight=np.r_[step[0]/2.,(step[:-1]+step[1:])/2.,step[-1]/2.]
    return period,weight/weight.sum()


def known_source_harmonic_statistics(time,mask,signals,foreground_period,background_period,foreground_harmonics,background_harmonics,device="cuda",record_chunk=64):
    """Exact observed-time sufficient statistics at paired source reference periods."""
    time=np.asarray(time,dtype=float);mask=np.asarray(mask,dtype=bool)
    signals=np.asarray(signals,dtype=float)
    df=2*int(foreground_harmonics);db=2*int(background_harmonics)
    n=len(mask);views=signals.shape[-1]
    output={"fg_gram":np.empty((n,df,df)),"bg_gram":np.empty((n,db,db)),
        "cross":np.empty((n,df,db)),"fg_rhs":np.empty((n,df,views)),
        "bg_rhs":np.empty((n,db,views)),"energy":np.empty((n,views)),"degrees":mask.sum(1)-2.}
    dtype=torch.float64
    tensor=lambda value:torch.as_tensor(value,dtype=dtype,device=device)
    def block(start):
        stop=min(start+record_chunk,n);m=mask[start:stop];count=m.sum(1)
        first=np.min(np.where(m,time[None],np.inf),axis=1)
        actual=time[None]-first[:,None]
        centered=(actual-(actual*m).sum(1)[:,None]/count[:,None])*m
        trend=np.stack([m/np.sqrt(count[:,None]),centered/np.sqrt(np.sum(centered**2,axis=1))[:,None]],axis=-1)
        tz=tensor(trend);tm=tensor(m)
        def design(period,harmonics):
            angle=2*np.pi*actual[:,:,None]*np.arange(1,harmonics+1)[None,None]/np.asarray(period)[start:stop,None,None]
            h=tensor(np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(stop-start,len(time),2*harmonics))*tm[:,:,None]
            return h-torch.einsum("nti,nij->ntj",tz,torch.einsum("nti,ntj->nij",tz,h))
        foreground=design(foreground_period,int(foreground_harmonics))
        background=design(background_period,int(background_harmonics))
        y=tensor(signals[start:stop])*tm[:,:,None]
        y-=torch.einsum("nti,niv->ntv",tz,torch.einsum("nti,ntv->niv",tz,y))
        output["fg_gram"][start:stop]=torch.einsum("nti,ntj->nij",foreground,foreground).cpu().numpy()
        output["bg_gram"][start:stop]=torch.einsum("nti,ntj->nij",background,background).cpu().numpy()
        output["cross"][start:stop]=torch.einsum("nti,ntj->nij",foreground,background).cpu().numpy()
        output["fg_rhs"][start:stop]=torch.einsum("nti,ntv->niv",foreground,y).cpu().numpy()
        output["bg_rhs"][start:stop]=torch.einsum("nti,ntv->niv",background,y).cpu().numpy()
        output["energy"][start:stop]=y.square().sum(1).cpu().numpy()
        return None
    list(map(block,np.arange(0,n,record_chunk)))
    return output


def period_reference_mass(period,target,qualified):
    """Barycentric reference mass on an increasing, possibly uneven grid."""
    period=np.asarray(period,dtype=float);target=np.asarray(target,dtype=float)
    lower=np.clip(np.searchsorted(period,target,side="right")-1,0,len(period)-2)
    fraction=np.clip((target-period[lower])/(period[lower+1]-period[lower]),0.,1.)
    identity=np.eye(len(period))
    return ((1.-fraction)[:,None]*identity[lower]+fraction[:,None]*identity[lower+1])*qualified[:,None]


def fit_reference_source_filter(bundle,statistics,qualified,training_run,independent_unit,reference_source_id=None,maxiter=900,device="cuda"):
    """Learn foreground-preserving departures from conditional peer removal.

    Paired FG waveforms supervise signal correlation, including its polarity
    nuisance. Four coefficients adapt the two cancellation weights to observed
    mixing quality. Prediction uses No and its measured peers alone.
    """
    context={"qualified":qualified,"origin_run":bundle["run"],"independent_unit":independent_unit,
        "view_weight":np.ones(len(qualified)),"is_original":np.ones(len(qualified),dtype=bool)}
    weights=source_reference_weights(bundle,context,training_run)
    identifiers=bundle["trace_id"] if reference_source_id is None else reference_source_id
    code,names=pd.factorize(identifiers,sort=True)
    member=(np.arange(len(names))[:,None]==code[None]).astype(float)
    n=((weights["train"].astype(float)@member.T)>0.).sum(1)
    dtype=torch.float64
    covariance=torch.as_tensor(statistics["raw_covariance"],dtype=dtype,device=device)
    cross=torch.as_tensor(statistics["reference_cross"],dtype=dtype,device=device)
    reference_energy=torch.as_tensor(statistics["reference_energy"],dtype=dtype,device=device)
    initial=np.column_stack([np.ones(len(qualified)),-statistics["nuisance_coefficient"]])
    base=torch.as_tensor(initial,dtype=dtype,device=device)
    quality=np.column_stack([np.ones(len(qualified)),bundle["mixing_quality"]])
    tq=torch.as_tensor(quality,dtype=dtype,device=device)
    def case(index):
        weight=torch.as_tensor(weights["training_weight"][index],dtype=dtype,device=device)
        def objective(parameter,readout=False):
            theta=torch.as_tensor(parameter,dtype=dtype,device=device).requires_grad_(not readout)
            correction=tq@theta.reshape(2,2)
            mixing=torch.cat([base[:,:1],base[:,1:]+correction],dim=-1)
            energy=torch.einsum("ni,nij,nj->n",mixing,covariance,mixing)
            matched=(mixing*cross).sum(-1)
            r2=matched.square()/(energy*reference_energy).clamp_min(torch.finfo(dtype).tiny)
            r2=r2.clamp(0.,1.)
            if readout:
                return mixing.detach().cpu().numpy(),r2.detach().cpu().numpy()
            loss=(weight*(1.-r2)).sum()+.5*theta.square().sum()/n[index]
            gradient=torch.autograd.grad(loss,theta)[0]
            return float(loss.detach().cpu()),gradient.detach().cpu().numpy()
        initial_weight=weights["training_weight"][index]
        initial_parameter=np.linalg.pinv(quality.T@(quality*initial_weight[:,None]))@(quality.T@(statistics["nuisance_coefficient"]*initial_weight[:,None]))
        fit=minimize(objective,initial_parameter.ravel(),jac=True,method="L-BFGS-B",options={"maxiter":maxiter,"ftol":1e-11,"gtol":1e-7})
        mixing,r2=objective(fit.x,True)
        _,original=objective(np.zeros(4),True)
        print("REFERENCE_SOURCE_FILTER",index,bool(fit.success),float(weights["training_weight"][index]@original),float(weights["training_weight"][index]@r2),flush=True)
        return {"source_filter_delta":fit.x.reshape(2,2),"source_filter":mixing,"success":np.asarray(fit.success),
            "training_source_r2":weights["training_weight"][index]@r2,"training_projected_r2":weights["training_weight"][index]@original}
    fitted=list(map(case,np.arange(len(training_run))))
    return {key:np.stack(list(map(lambda value:value[key],fitted))) for key in fitted[0]}


def fit_joint_template_temperature(bundle,score,reference,qualified,training_run,independent_unit,reference_source_id=None,maxiter=900,device="cuda",view_context=None,prior_source=None):
    """Calibrate one template likelihood inside one coherent cohort posterior.

    The training references determine the population prior and within spread.
    A single nonnegative likelihood temperature is fitted per exclusion case.
    Individual periods and finite-cohort means are read from the same posterior.
    """
    period=bundle["period"];scale=float(bundle["period_scale"]);axis=period/scale
    integration=np.asarray(bundle.get("integration_weight",np.ones(len(period))),dtype=float)
    integration=integration/integration.sum()
    from torch.utils.checkpoint import checkpoint
    prior=cohort_reference_prior(bundle,reference,qualified,training_run,independent_unit) if prior_source is None else prior_source
    context={"qualified":qualified,"origin_run":bundle["run"],"independent_unit":independent_unit,
        "view_weight":np.ones(len(reference)),"is_original":np.ones(len(reference),dtype=bool)}
    context=context if view_context is None else view_context
    weights=source_reference_weights(bundle,context,training_run)
    identifiers=bundle["trace_id"] if reference_source_id is None else reference_source_id
    source_code,source_names=pd.factorize(identifiers,sort=True)
    source_member=(np.arange(len(source_names))[:,None]==source_code[None]).astype(float)
    original_train=weights["train"]&context["is_original"][None]
    n=((original_train.astype(float)@source_member.T)>0.).sum(1)
    centered=score-np.einsum("fnp,p->fn",score,integration,optimize=True)[:,:,None]
    scale_weight=weights["training_weight"]*context["is_original"][None]
    scale_weight/=scale_weight.sum(1)[:,None]
    feature_scale=np.sqrt(np.einsum("fn,fnp,p->f",scale_weight,centered**2,integration,optimize=True))
    feature_scale=np.maximum(feature_scale,np.finfo(float).tiny)
    target=np.where(qualified,reference/scale,0.)
    teacher=period_reference_mass(axis,target,qualified)
    reference_number=bundle["membership"]@qualified.astype(float)
    reference_mean=np.divide(bundle["membership"]@target,reference_number,out=np.zeros(len(reference_number)),where=reference_number>0.)
    sd=np.sqrt(np.maximum(weights["training_weight"]@(target**2)-(weights["training_weight"]@target)**2,(axis[1]-axis[0])**2/12.))
    dtype=torch.float64
    tteacher=torch.as_tensor(teacher,dtype=dtype,device=device)
    taxis=torch.as_tensor(axis,dtype=dtype,device=device)
    member=torch.as_tensor(bundle["membership"],dtype=dtype,device=device)
    average=torch.as_tensor(bundle["average"],dtype=dtype,device=device)
    run=torch.as_tensor(bundle["run_index"],dtype=torch.long,device=device)
    difference=axis[:,None]-axis[None]
    point=np.full((len(axis),len(axis)),-np.inf);np.fill_diagonal(point,0.)
    group_records=list(map(lambda group:torch.as_tensor(np.flatnonzero(bundle["run_index"]==group),dtype=torch.long,device=device),np.arange(len(bundle["run_names"]))))
    grouped_order=np.concatenate(list(map(lambda group:np.flatnonzero(bundle["run_index"]==group),np.arange(len(bundle["run_names"])))))
    original_order=torch.as_tensor(np.argsort(grouped_order),dtype=torch.long,device=device)

    def case(index):
        within=float(prior["foreground_within_sd"][index])
        kernel=point if within==0. else -.5*(difference/within)**2
        if "integration_weight" in bundle:
            kernel=kernel+np.log(integration)[None]
        kernel=kernel-logsumexp(kernel,axis=-1,keepdims=True)
        tk=torch.as_tensor(kernel,dtype=dtype,device=device)
        ts=torch.as_tensor(centered[index]/feature_scale[index],dtype=dtype,device=device)
        tp=torch.as_tensor(prior["mean_log_prior"][index],dtype=dtype,device=device)
        tw=torch.as_tensor(weights["training_weight"][index],dtype=dtype,device=device)
        tg=torch.as_tensor(weights["group_weight"][index],dtype=dtype,device=device)
        tr=torch.as_tensor(reference_mean,dtype=dtype,device=device)
        pair=None
        if view_context is not None:
            left=context["pair_left"];right=context["pair_right"]
            eligible=context["pair_eligible"]&training_run[index,left]&training_run[index,right]
            pair_weight=(weights["group_weight"][index,left]+weights["group_weight"][index,right])*eligible/2.
            pair_weight/=max(pair_weight.sum(),np.finfo(float).tiny)
            pair=(torch.as_tensor(left,dtype=torch.long,device=device),torch.as_tensor(right,dtype=torch.long,device=device),
                torch.as_tensor(context["pair_effect"],dtype=dtype,device=device),torch.as_tensor(pair_weight,dtype=dtype,device=device))

        def cohort_value(likelihood,group):
            records=group_records[group]
            local=likelihood[records]
            integrated=torch.logsumexp(local[:,None,:]+tk[None],dim=-1)
            log_mean=torch.log_softmax(tp+integrated.sum(0),dim=-1)
            cavity=log_mean[None]-integrated
            value=local+torch.logsumexp(cavity[:,:,None]+tk[None],dim=-2)
            return torch.log_softmax(value,dim=-1)

        def objective(parameter,readout=False):
            thermal=torch.as_tensor(parameter,dtype=dtype,device=device).requires_grad_(not readout)
            likelihood=thermal[0]*ts
            if within==0.:
                log_mean=torch.log_softmax(tp[None]+member@likelihood,dim=-1)
                log=log_mean[run]
            else:
                parts=list(map(lambda group:checkpoint(cohort_value,likelihood,int(group),use_reentrant=False) if likelihood.requires_grad else cohort_value(likelihood,int(group)),np.arange(len(group_records))))
                log=torch.cat(parts,dim=0)[original_order]
            log=torch.log_softmax(log,dim=-1)
            probability=log.exp();individual=probability@taxis
            mean=average@individual
            if readout:
                return probability.detach().cpu().numpy(),mean.detach().cpu().numpy()
            ce=-(log*tteacher*tw[:,None]).sum()
            crps=((probability.cumsum(-1)[:,:-1]-tteacher.cumsum(-1)[:,:-1]).square()*tw[:,None]*torch.as_tensor(np.diff(axis),dtype=dtype,device=device)[None]).sum()/sd[index]
            moment=.5*((mean-tr).square()*tg).sum()/sd[index]**2
            loss=ce+crps+moment+.5*thermal[0].square()/n[index]
            if pair is not None:
                left,right,effect,pair_weight=pair
                loss=loss+.5*(((mean[right]-mean[left]-effect)/sd[index]).square()*pair_weight).sum()
            gradient=torch.autograd.grad(loss,thermal)[0]
            return float(loss.detach().cpu()),gradient.detach().cpu().numpy()

        fitted=minimize(objective,np.asarray([0.]),jac=True,bounds=[(0.,None)],method="L-BFGS-B",
            options={"maxiter":maxiter,"ftol":1e-10,"gtol":1e-6})
        probability,mean=objective(fitted.x,True)
        print("JOINT_TEMPLATE_CASE",index,bool(fitted.success),float(fitted.x[0]/feature_scale[index]),flush=True)
        return {"posterior":probability,"group_mean":mean*scale,"estimate":probability@period,
            "temperature":fitted.x[0]/feature_scale[index],"success":np.asarray(fitted.success)}

    fitted=list(map(case,np.arange(len(training_run))))
    output={key:np.stack(list(map(lambda item:item[key],fitted))) for key in fitted[0]}
    output.update({"mean_log_prior":prior["mean_log_prior"],"foreground_within_sd":prior["foreground_within_sd"],
        "reference_source_count":n,"effective_training_units":prior["effective_training_units"],
        "readout_kind":np.asarray("joint_reference_template"),"mean_prior_weight":prior["mean_prior_weight"],
        "mean_prior_bandwidth":prior["mean_prior_bandwidth"],"reference_group_mean":prior["reference_group_mean"]})
    output["source_period"]=output["estimate"].copy()
    output["deviation"]=output["estimate"]-output["group_mean"][:,bundle["run_index"]]
    return output


def fit_joint_reference_source(bundle,statistics,template_coefficient,initial_filter,initial_temperature,reference,qualified,training_run,independent_unit,reference_source_id=None,maxiter=900,device="cuda"):
    """Jointly fit source preservation and period inference with five parameters."""
    from torch.utils.checkpoint import checkpoint
    period=bundle["period"];scale=float(bundle["period_scale"]);axis=period/scale
    integration=np.asarray(bundle.get("integration_weight",np.ones(len(period))),dtype=float)
    integration=integration/integration.sum()
    prior=cohort_reference_prior(bundle,reference,qualified,training_run,independent_unit)
    context={"qualified":qualified,"origin_run":bundle["run"],"independent_unit":independent_unit,
        "view_weight":np.ones(len(reference)),"is_original":np.ones(len(reference),dtype=bool)}
    weights=source_reference_weights(bundle,context,training_run)
    identifiers=bundle["trace_id"] if reference_source_id is None else reference_source_id
    code,names=pd.factorize(identifiers,sort=True)
    source_member=(np.arange(len(names))[:,None]==code[None]).astype(float)
    n=((weights["train"].astype(float)@source_member.T)>0.).sum(1)
    target=np.where(qualified,reference/scale,0.)
    teacher=period_reference_mass(axis,target,qualified)
    sd=np.sqrt(np.maximum(weights["training_weight"]@(target**2)-(weights["training_weight"]@target)**2,(axis[1]-axis[0])**2/12.))
    dtype=torch.float64
    tensor=lambda value:torch.as_tensor(value,dtype=dtype,device=device)
    tax=tensor(axis);ty=tensor(teacher);member=tensor(bundle["membership"]);average=tensor(bundle["average"])
    run=torch.as_tensor(bundle["run_index"],dtype=torch.long,device=device)
    tq=tensor(np.column_stack([np.ones(len(reference)),bundle["mixing_quality"]]))
    covariance=tensor(statistics["raw_covariance"]);fg_cross=tensor(statistics["reference_cross"]);fg_energy=tensor(statistics["reference_energy"])
    ols=tensor(statistics["nuisance_coefficient"])
    difference=axis[:,None]-axis[None]
    point=np.full((len(axis),len(axis)),-np.inf);np.fill_diagonal(point,0.)
    ordinary_rhs=tensor(statistics["ordinary_rhs"])
    nuisance_rhs=tensor(statistics["nuisance_rhs"])
    layout=torch.as_tensor(statistics["ordinary_layout"],dtype=torch.long,device=device)
    degrees=tensor(statistics["count"]-2.)
    def case(index):
        c=tensor(template_coefficient[index])
        norm=tensor(np.einsum("ki,lpij,kj->lpk",template_coefficient[index],statistics["ordinary_gram"],template_coefficient[index],optimize=True)).clamp_min(0.)
        tw=tensor(weights["training_weight"][index]);gw=tensor(prior["mean_prior_weight"][index])
        tp=tensor(prior["mean_log_prior"][index]);tr=tensor(prior["reference_group_mean"])
        within=prior["foreground_within_sd"][index]
        kernel=point if within==0. else -.5*(difference/within)**2
        if "integration_weight" in bundle:
            kernel=kernel+np.log(integration)[None]
        kernel=kernel-logsumexp(kernel,axis=-1,keepdims=True);tk=tensor(kernel)
        def source_score(delta):
            cancellation=-ols+tq@delta.reshape(2,2)
            mixing=torch.cat([torch.ones_like(cancellation[:,:1]),cancellation],dim=-1)
            energy=torch.einsum("ni,nij,nj->n",mixing,covariance,mixing)
            def block(parameter,start):
                stop=min(start+32,len(reference))
                current=-ols[start:stop]+tq[start:stop]@parameter.reshape(2,2)
                current_mixing=torch.cat([torch.ones_like(current[:,:1]),current],dim=-1)
                current_energy=torch.einsum("ni,nij,nj->n",current_mixing,covariance[start:stop],current_mixing)
                rhs=ordinary_rhs[start:stop]+torch.einsum("npdi,ni->npd",nuisance_rhs[start:stop],current)
                amplitude=torch.einsum("kd,npd->npk",c,rhs)
                denominator=norm[layout[start:stop]]*current_energy[:,None,None]
                r2=torch.where(denominator>0.,amplitude.square()/denominator.clamp_min(torch.finfo(dtype).tiny),0.).clamp(0.,1.)
                d=degrees[start:stop,None,None]
                log=-.5*torch.log1p(d)-.5*d*torch.log1p(-d/(1.+d)*r2)
                return torch.logsumexp(log,dim=-1)-np.log(template_coefficient.shape[1])
            parts=list(map(lambda start:checkpoint(block,delta,int(start),use_reentrant=False) if delta.requires_grad else block(delta,int(start)),np.arange(0,len(reference),32)))
            score=torch.cat(parts,dim=0)
            matched=(mixing*fg_cross).sum(-1)
            preservation=(matched.square()/(energy*fg_energy).clamp_min(torch.finfo(dtype).tiny)).clamp(0.,1.)
            return score,preservation
        initial_score,_=source_score(tensor(initial_filter[index].ravel()))
        centered=initial_score-(initial_score*tensor(integration)[None]).sum(-1,keepdim=True)
        feature_scale=float(torch.sqrt((centered.square()*tw[:,None]*tensor(integration)[None]).sum()).cpu())
        feature_scale=max(feature_scale,np.finfo(float).tiny)
        def objective(parameter,readout=False):
            theta=tensor(parameter).requires_grad_(not readout)
            score,preservation=source_score(theta[:4])
            likelihood=theta[4]/feature_scale*score
            if within==0.:
                log_mean=torch.log_softmax(tp[None]+member@likelihood,dim=-1)
                log=log_mean[run]
            else:
                integrated=torch.logsumexp(likelihood[:,None]+tk[None],dim=-1)
                log_mean=torch.log_softmax(tp[None]+member@integrated,dim=-1)
                cavity=log_mean[run]-integrated
                log=likelihood+torch.logsumexp(cavity[:,:,None]+tk[None],dim=-2)
            log=torch.log_softmax(log,dim=-1);mass=log.exp();individual=mass@tax;mean=average@individual
            if readout:
                return mass.detach().cpu().numpy(),mean.detach().cpu().numpy(),preservation.detach().cpu().numpy()
            ce=-(log*ty*tw[:,None]).sum()
            ordered=((mass.cumsum(-1)[:,:-1]-ty.cumsum(-1)[:,:-1]).square()*tw[:,None]*tensor(np.diff(axis))[None]).sum()/sd[index]
            moment=.5*((mean-tr).square()*gw).sum()/sd[index]**2
            waveform=((1.-preservation)*tw).sum()
            penalty=.5*theta.square().sum()/n[index]
            loss=ce+ordered+moment+waveform+penalty
            gradient=torch.autograd.grad(loss,theta)[0]
            return float(loss.detach().cpu()),gradient.detach().cpu().numpy()
        initial=np.r_[initial_filter[index].ravel(),initial_temperature[index]*feature_scale]
        fit=minimize(objective,initial,jac=True,bounds=[(None,None)]*4+[(0.,None)],method="L-BFGS-B",
            options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-5,"maxcor":25})
        mass,mean,preservation=objective(fit.x,True)
        print("JOINT_REFERENCE_SOURCE_CASE",index,bool(fit.success),fit.nit,float(np.max(abs(fit.jac))),float(weights["training_weight"][index]@preservation),flush=True)
        return {"posterior":mass,"estimate":mass@period,"group_mean":mean*scale,"temperature":fit.x[4]/feature_scale,
            "source_filter_delta":fit.x[:4].reshape(2,2),"training_source_r2":weights["training_weight"][index]@preservation,"success":np.asarray(fit.success)}
    fitted=list(map(case,np.arange(len(training_run))))
    output={key:np.stack(list(map(lambda value:value[key],fitted))) for key in fitted[0]}
    output.update({"mean_log_prior":prior["mean_log_prior"],"foreground_within_sd":prior["foreground_within_sd"],
        "reference_source_count":n,"effective_training_units":prior["effective_training_units"],"readout_kind":np.asarray("joint_reference_source"),
        "mean_prior_weight":prior["mean_prior_weight"],"mean_prior_bandwidth":prior["mean_prior_bandwidth"],"reference_group_mean":prior["reference_group_mean"]})
    output["source_period"]=output["estimate"].copy()
    output["deviation"]=output["estimate"]-output["group_mean"][:,bundle["run_index"]]
    return output


def fit_joint_source_field(bundle,template_score,role,reference,qualified,training_run,independent_unit,reference_source_id=None,maxiter=900,device="cuda"):
    """Fit local and cohort source energies through one conditional posterior.

    Shared measurements enter once at the population location. Centered local
    measurements determine individual deviations. All readouts are marginals
    of this same conditional distribution.
    """
    period=bundle["period"];scale=float(bundle["period_scale"]);axis=period/scale
    base=bundle["source_base"].copy();base[:,:,11]=bundle["source_coherent_power"]
    phi=_polynomial(base)
    left,right=np.triu_indices(13)
    shared=np.r_[np.isin(np.arange(13),np.arange(4,9)),np.isin(left,np.arange(4,9))&np.isin(right,np.arange(4,9))]
    prior=cohort_reference_prior(bundle,reference,qualified,training_run,independent_unit)
    context={"qualified":qualified,"origin_run":bundle["run"],"independent_unit":independent_unit,
        "view_weight":np.ones(len(reference)),"is_original":np.ones(len(reference),dtype=bool)}
    weights=source_reference_weights(bundle,context,training_run)
    identifiers=bundle["trace_id"] if reference_source_id is None else reference_source_id
    code,names=pd.factorize(identifiers,sort=True)
    source_member=(np.arange(len(names))[:,None]==code[None]).astype(float)
    n=((weights["train"].astype(float)@source_member.T)>0.).sum(1)
    target=np.where(qualified,reference/scale,0.)
    teacher=np.maximum(1.-abs(axis[None]-target[:,None])/(axis[1]-axis[0]),0.)*qualified[:,None]
    teacher/=np.maximum(teacher.sum(1),1e-12)[:,None]
    sd=np.sqrt(np.maximum(weights["training_weight"]@(target**2)-(weights["training_weight"]@target)**2,(axis[1]-axis[0])**2/12.))
    dtype=torch.float64
    tax=torch.as_tensor(axis,dtype=dtype,device=device)
    tm=torch.as_tensor(bundle["membership"],dtype=dtype,device=device)
    ta=torch.as_tensor(bundle["average"],dtype=dtype,device=device)
    run=torch.as_tensor(bundle["run_index"],dtype=torch.long,device=device)
    ty=torch.as_tensor(teacher,dtype=dtype,device=device)
    difference=axis[:,None]-axis[None]
    point=np.full((len(axis),len(axis)),-np.inf);np.fill_diagonal(point,0.)
    def case(index):
        local=np.concatenate([phi[:,:,~shared],template_score[index,:,:,None],role[index,:,:,:1]],axis=-1)
        local-=np.einsum("gn,npd->gpd",bundle["average"],local,optimize=True)[bundle["run_index"]]
        contextual=np.concatenate([phi,template_score[index,:,:,None],role[index]],axis=-1)
        contextual=np.einsum("gn,npd->gpd",bundle["average"],contextual,optimize=True)
        local-=local.mean(1,keepdims=True);contextual-=contextual.mean(1,keepdims=True)
        local_scale=np.sqrt(np.einsum("n,npd->d",weights["training_weight"][index],local**2,optimize=True)/len(period))
        group_scale=np.sqrt(np.einsum("g,gpd->d",prior["mean_prior_weight"][index],contextual**2,optimize=True)/len(period))
        local_scale=np.where(local_scale>1e-12,local_scale,1.)
        group_scale=np.where(group_scale>1e-12,group_scale,1.)
        local/=local_scale;contextual/=group_scale
        tx=torch.as_tensor(local,dtype=dtype,device=device)
        tg=torch.as_tensor(contextual,dtype=dtype,device=device)
        tw=torch.as_tensor(weights["training_weight"][index],dtype=dtype,device=device)
        gw=torch.as_tensor(prior["mean_prior_weight"][index],dtype=dtype,device=device)
        tr=torch.as_tensor(prior["reference_group_mean"],dtype=dtype,device=device)
        tp=torch.as_tensor(prior["mean_log_prior"][index],dtype=dtype,device=device)
        within=prior["foreground_within_sd"][index]
        kernel=point if within==0. else -.5*(difference/within)**2
        kernel-=logsumexp(kernel,axis=-1,keepdims=True)
        tk=torch.as_tensor(kernel,dtype=dtype,device=device)
        dimension=local.shape[-1]
        def objective(parameter,readout=False):
            theta=torch.as_tensor(parameter,dtype=dtype,device=device).requires_grad_(not readout)
            node=tx@theta[:dimension]
            integrated=torch.logsumexp(node[:,None,:]+tk[None],dim=-1)
            mean_log=torch.log_softmax(tp[None]+tg@theta[dimension:]+tm@integrated,dim=-1)
            cavity=mean_log[run]-integrated
            log=node+torch.logsumexp(cavity[:,:,None]+tk[None],dim=-2)
            log=torch.log_softmax(log,dim=-1);mass=log.exp()
            individual=mass@tax;mean=ta@individual
            if readout:
                return mass.detach().cpu().numpy(),mean.detach().cpu().numpy()
            ce=-(log*ty*tw[:,None]).sum()
            ordered=((mass.cumsum(-1)[:,:-1]-ty.cumsum(-1)[:,:-1]).square()*tw[:,None]).sum()*float(axis[1]-axis[0])/sd[index]
            moment=.5*((mean-tr).square()*gw).sum()/sd[index]**2
            penalty=.5*theta[:dimension].square().sum()/n[index]+.5*theta[dimension:].square().sum()/prior["effective_training_units"][index]
            loss=ce+ordered+moment+penalty
            gradient=torch.autograd.grad(loss,theta)[0]
            return float(loss.detach().cpu()),gradient.detach().cpu().numpy()
        fit=minimize(objective,np.zeros(dimension+contextual.shape[-1]),jac=True,method="L-BFGS-B",
            options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-5,"maxcor":25})
        mass,mean=objective(fit.x,True)
        print("JOINT_SOURCE_FIELD_CASE",index,bool(fit.success),fit.nit,float(np.max(abs(fit.jac))),flush=True)
        return {"posterior":mass,"estimate":mass@period,"group_mean":mean*scale,
            "local_coefficient":fit.x[:dimension]/local_scale,"group_coefficient":fit.x[dimension:]/group_scale,
            "success":np.asarray(fit.success)}
    fitted=list(map(case,np.arange(len(training_run))))
    output={key:np.stack(list(map(lambda value:value[key],fitted))) for key in fitted[0]}
    output.update({"mean_log_prior":prior["mean_log_prior"],"foreground_within_sd":prior["foreground_within_sd"],
        "reference_source_count":n,"effective_training_units":prior["effective_training_units"],"readout_kind":np.asarray("joint_source_field")})
    output["source_period"]=output["estimate"].copy()
    output["deviation"]=output["estimate"]-output["group_mean"][:,bundle["run_index"]]
    return output


def _source_base_variant(bundle,variant):
    key={1:"unadjusted_base",8:"base_features",9:"independent_base"}.get(variant,"source_base")
    base=bundle[key].copy()
    if variant==2:
        base[:,:,4:13]=0.
    return base


def _core_base_variant(bundle,variant,parent=False):
    base=_base_variant(bundle,variant) if parent else _source_base_variant(bundle,variant)
    if variant not in [2,6]:
        if parent:
            base[:,:,11]=bundle["parent_coherent_power"]
        else:
            field={1:"unadjusted",8:"original",9:"independent"}.get(variant,"source")
            base[:,:,11]=bundle["source_coherent_global" if variant==15 else field+"_coherent_half"]
    return base


def _geometry_coordinate(bundle):
    if "geometry_coordinate" in bundle:
        return bundle["geometry_coordinate"]
    clock=bundle["time"]
    mask=bundle["mask"]
    span=np.max(np.where(mask,clock[None],-np.inf),axis=1)-np.min(np.where(mask,clock[None],np.inf),axis=1)
    return np.log(span/float(bundle["period_scale"]))


def retired_reference_view_features(path,context,period_min,period_max,candidates=73):
    """Read each reference view once and retain only period sufficient statistics."""
    view_ids=np.unique(context["view_id"])
    record_keys=["trace_id","run","condition","count","base_features","source_base","source_rhs","original_rhs","source_coherent_half"]

    def view(tag):
        print("REFERENCE_VIEW_FEATURES",str(tag),flush=True)
        frame=pd.read_parquet(path,columns=["trace_id","run","condition","x","signal_y"],filters=[("view_id","=",str(tag))])
        current=feature_bundle(frame,period_min,period_max,candidates,include_parent=False,include_controls=False)
        compact=dict(zip(record_keys,map(current.__getitem__,record_keys)))
        compact["geometry_coordinate"]=_geometry_coordinate(current)
        compact["joint_gram"]=current["joint_gram"]
        compact["layout"]=current["layout"]
        return compact

    parts=list(map(view,view_ids))
    keys=record_keys+["geometry_coordinate"]
    bundle=dict(zip(keys,map(lambda key:np.concatenate(list(map(lambda part:part[key],parts)),axis=0),keys)))
    offset=np.r_[0,np.cumsum(np.asarray(list(map(lambda part:len(part["joint_gram"]),parts))))[:-1]]
    bundle["joint_gram"]=np.concatenate(list(map(lambda part:part["joint_gram"],parts)),axis=0)
    bundle["layout"]=np.concatenate(list(map(lambda pair:pair[1]["layout"]+pair[0],zip(offset,parts))))
    code,runs=pd.factorize(bundle["run"],sort=True)
    membership=(np.arange(len(runs))[:,None]==code[None]).astype(float)
    bundle.update({"run_index":code,"run_names":np.asarray(runs,dtype=str),"membership":membership,"average":membership/membership.sum(1)[:,None],
        "period":np.linspace(period_min,period_max,candidates),"period_scale":np.asarray(np.sqrt(period_min*period_max)),
        "source_operator":np.asarray(SOURCE_OPERATOR)})
    record_order=pd.Index(context["trace_id"]).get_indexer(bundle["trace_id"])
    group_order=pd.Index(context["run_name"]).get_indexer(bundle["run_names"])
    assert np.all(record_order>=0) and np.all(group_order>=0)
    selected={"is_original":context["is_original"][record_order],"view_weight":context["view_weight"][record_order],
        "origin_run":context["origin_run"][record_order],
        "target":context["target"][record_order],"qualified":context["qualified"][record_order],
        "group_reference":context["group_reference"][group_order],
        "training_run":np.concatenate([np.ones((1,len(group_order)),dtype=bool),context["outer_train_run"][:,group_order]])}
    old_to_new=pd.Index(bundle["run_names"]).get_indexer(context["run_name"])
    left=context["pair_left_run"].ravel();right=context["pair_right_run"].ravel()
    valid=context["pair_eligible"].ravel()&(left>=0)&(right>=0)
    selected["pair_left"]=np.where(valid,old_to_new[np.maximum(left,0)],0)
    selected["pair_right"]=np.where(valid,old_to_new[np.maximum(right,0)],0)
    selected["pair_eligible"]=valid
    selected["pair_effect"]=context["pair_effect"].ravel()
    return bundle,selected


def _source_design(bundle,base,center,scale,variant):
    # Timing enters the measured Gram/OU geometry; only control 13 learns
    # a separate duration interaction. Zero padding keeps one parameter schema.
    geometry=np.tanh((_geometry_coordinate(bundle)-center)/scale)*(variant==13)
    poly=_polynomial(base)
    return np.concatenate([poly,poly*geometry[:,None,None]],axis=-1)


def _role_features(bundle,covariance,background,variant,field=None):
    """Evidence for a foreground clock beyond the same background-only model."""
    field=field or {1:"unadjusted_rhs",8:"original_rhs",9:"independent_rhs"}.get(variant,"source_rhs")
    rhs=bundle[field]
    gram=bundle["joint_gram"]
    result=np.empty((len(covariance),len(rhs),len(bundle["period"]),2))

    def case(index):
        h=covariance[index]
        ratio=np.sqrt(np.repeat(h[:,:2],2,axis=1)/h[:,2,None])
        precision=np.eye(4)[None,None,None,None]+gram[None]*ratio[:,None,None,None,:,None]*ratio[:,None,None,None,None,:]
        inverse=np.linalg.inv(precision)
        determinant=np.linalg.slogdet(precision)[1]
        background_precision=np.eye(2)[None,None,None]+gram[None,:,0,:,2:,2:]*h[:,1,None,None,None,None]/h[:,2,None,None,None,None]
        background_inverse=np.linalg.inv(background_precision)
        background_determinant=np.linalg.slogdet(background_precision)[1]

        def block(indices):
            r=rhs[indices]
            selected=inverse[:,bundle["layout"][indices]]
            pp=np.einsum("npiv,vnpqij,npjv->vnpq",r,selected[...,:2,:2],r,optimize=True)
            qq=np.einsum("nqiv,vnpqij,nqjv->vnpq",r,selected[...,2:,2:],r,optimize=True)
            pq=np.einsum("npiv,vnpqij,nqjv->vnpq",r,selected[...,:2,2:],r,optimize=True)
            gain=(pp*h[:,0,None,None,None]+qq*h[:,1,None,None,None]+2*pq*np.sqrt(h[:,0]*h[:,1])[:,None,None,None])/h[:,2,None,None,None]**2
            pair=.5*(gain-determinant[:,bundle["layout"][indices]])
            log_bg=np.log(np.maximum(background[index,indices],1e-300))
            full=logsumexp(pair+log_bg[None,:,None],axis=-1)
            background_gain=np.einsum("nqiv,vnqij,nqjv->vnq",r,background_inverse[:,bundle["layout"][indices]],r,optimize=True)
            background_gain*=h[:,1,None,None]/h[:,2,None,None]**2
            background_only=.5*(background_gain-background_determinant[:,bundle["layout"][indices]])
            # This null is independent of the proposed foreground period P.
            # An unused foreground at a remote P therefore pays its evidence
            # cost rather than gaining from a worse role-swapped alternative.
            null=logsumexp(background_only+log_bg[None],axis=-1)
            result[index,indices]=(full-null[:,:,None]).transpose(1,2,0)
            return None

        list(map(block,np.array_split(np.arange(len(rhs)),max(1,int(np.ceil(len(rhs)/32))))))
        return None

    list(map(case,np.arange(len(covariance))))
    if variant==2:
        result[:,:,:,1]=0.
    if variant==3:
        result[:]=0.
    return result


def _continue_pipeline(bundle,state,reference,qualified,training_run,variant,maxiter,device,view_bundle,view_context):
    """Continue failed mean/role cases while preserving verified fitted stages."""
    assert state["reference_success_by_case"].all()
    period=bundle["period"];scale=float(bundle["period_scale"])
    target=np.where(qualified,reference/scale,0.)
    train=training_run[:,bundle["run_index"]]&qualified[None]
    weight=train/train.sum(1)[:,None]
    group_reference=(bundle["membership"]@target)/np.maximum(bundle["membership"]@qualified,1.)
    group_reference=np.where(bundle["membership"]@qualified>0,group_reference,np.nan)
    sd=np.sqrt(np.maximum(weight@(target**2)-(weight@target)**2,(period[1]-period[0])**2/(12*scale**2)))
    center=float(state["geometry_center"]);geometry_scale=float(state["geometry_scale"])
    changed_prior=np.flatnonzero(~state["view_success_by_case"])
    if len(changed_prior):
        assert view_bundle is not None and view_context is not None
        base=_source_base_variant(view_bundle,variant);raw=_base_variant(view_bundle,variant)
        blocks=changed_prior[:,None]
        core_width=208 if variant==13 else 104
        core_base=_core_base_variant(view_bundle,variant)
        core_design=_source_design(view_bundle,core_base,center,geometry_scale,variant) if variant==13 else _polynomial(core_base)

        def continue_view(selected):
            print("CONTINUE_SOURCE_REFERENCE",str(METHODS[variant]),selected.tolist(),flush=True)
            role,fixed=_source_fields(view_bundle,base,raw,state["parent_core_coefficient"][selected],state["identity_core_coefficient"][selected],
                state["background_coefficient"][selected],state["role_covariance"][selected],state["foreground_log_prior"][selected],
                state["background_log_prior"][selected],variant,center,geometry_scale)
            view_training=view_context["training_run"][selected]
            eligible=view_context["pair_eligible"][None]&view_training[:,view_context["pair_left"]]&view_training[:,view_context["pair_right"]]
            current=dict(view_context)
            current["pair_weight"]=eligible/np.maximum(eligible.sum(1),1)[:,None]
            current["pair_effect"]=view_context["pair_effect"]/scale
            fit=_fit_mean_role(view_bundle,base,role,fixed,state["raw_role_coefficient"][selected],view_context["group_reference"]/scale,
                view_training,(state["calibration_prediction"][selected],state["calibration_mean"][selected],state["calibration_experiments"][selected]),
                sd[selected],state["effective_training_units"][selected],maxiter,device,current,initial_coefficient=state["prior_role_coefficient"][selected],
                period_reference=(view_context["target"]/scale,view_context["qualified"]),
                core_fit=(core_design,state["parent_core_coefficient"][selected,:core_width],state["core_coefficient"][selected,:core_width]))
            state["core_coefficient"][selected,:core_width]=fit["core_coefficient"]
            state["prior_role_coefficient"][selected]=fit["role_coefficient"]
            state["prior_mean_bias"][selected]=fit["mean_bias"]
            state["view_success_by_case"][selected]=fit["case_success"]
            state["case_message"][selected]=np.char.add("Source-reference prior: ",fit["case_message"])
            state["iterations"]+=fit["iterations"]
            return None

        list(map(continue_view,blocks))
    selected_native=np.union1d(np.flatnonzero(~state["case_success"]),changed_prior)
    base=_source_base_variant(bundle,variant);raw=_base_variant(bundle,variant)
    blocks=np.array_split(selected_native,max(1,int(np.ceil(len(selected_native)/2))))

    def continue_native(selected):
        if not len(selected):
            return None
        print("CONTINUE_NATIVE_ROLE",str(METHODS[variant]),selected.tolist(),flush=True)
        role,fixed=_source_fields(bundle,base,raw,state["core_coefficient"][selected],state["identity_core_coefficient"][selected],
            state["background_coefficient"][selected],state["role_covariance"][selected],state["foreground_log_prior"][selected],
            state["background_log_prior"][selected],variant,center,geometry_scale)
        fit=_fit_mean_role(bundle,base,role,fixed,state["prior_role_coefficient"][selected],group_reference,training_run[selected],
            (state["calibration_prediction"][selected],state["calibration_mean"][selected],state["calibration_experiments"][selected]),
            sd[selected],state["effective_training_units"][selected],maxiter,device,initial_coefficient=state["role_coefficient"][selected],
            period_reference=(target,qualified))
        state["role_coefficient"][selected]=fit["role_coefficient"]
        state["mean_bias"][selected]=fit["mean_bias"]
        state["source_period"][selected]=fit["source_period"]*scale
        state["group_mean"][selected]=np.clip(fit["group_mean"]*scale,period.min(),period.max())
        if state.get("posterior") is not None:
            state["posterior"][selected]=fit["posterior"]
        state["case_success"][selected]=state["reference_success_by_case"][selected]&state["view_success_by_case"][selected]&fit["case_success"]
        state["case_message"][selected]=np.char.add("Native continuation: ",fit["case_message"])
        state["native_iterations_by_case"][selected]=fit["case_iterations"]
        state["native_gradient_by_case"][selected]=fit["case_gradient"]
        state["native_solver_by_case"][selected]=fit["case_solver"]
        state["iterations"]+=fit["iterations"]
        return None

    list(map(continue_native,blocks))
    state["success"]=bool(state["case_success"].all())
    state["gradient"]=float(np.nanmax(state["native_gradient_by_case"]))
    return state


def _refine_without_views(bundle,state,target,qualified,training_run,maxiter,device):
    state=dict(state)
    state["core_coefficient"]=state["parent_core_coefficient"].copy()
    scale=float(bundle["period_scale"])
    train=training_run[:,bundle["run_index"]]&qualified[None]
    weight=train/train.sum(1)[:,None]
    label=np.where(qualified,target/scale,0.)
    reference=bundle["membership"]@label/np.maximum(bundle["membership"]@qualified,1.)
    reference=np.where(bundle["membership"]@qualified>0,reference,np.nan)
    sd=np.sqrt(np.maximum(weight@(label**2)-(weight@label)**2,np.diff(bundle["period"])[0]**2/(12*scale**2)))
    base=_source_base_variant(bundle,0);raw=_base_variant(bundle,0)
    blocks=np.array_split(np.arange(len(training_run)),max(1,int(np.ceil(len(training_run)/2))))

    def block(selected):
        role,fixed=_source_fields(bundle,base,raw,state["core_coefficient"][selected],state["identity_core_coefficient"][selected],state["background_coefficient"][selected],
            state["role_covariance"][selected],state["foreground_log_prior"][selected],state["background_log_prior"][selected],0,float(state["geometry_center"]),float(state["geometry_scale"]))
        return _fit_mean_role(bundle,base,role,fixed,state["raw_role_coefficient"][selected],reference,training_run[selected],
            (state["calibration_prediction"][selected],state["calibration_mean"][selected],state["calibration_experiments"][selected]),sd[selected],state["effective_training_units"][selected],maxiter,device,
            period_reference=(label,qualified))

    fitted=list(map(block,blocks))
    value=dict(state)
    value["role_coefficient"]=np.concatenate(list(map(lambda entry:entry["role_coefficient"],fitted)),axis=0)
    value["mean_bias"]=np.concatenate(list(map(lambda entry:entry["mean_bias"],fitted)))
    value["prior_role_coefficient"]=state["raw_role_coefficient"]
    value["prior_mean_bias"]=np.zeros_like(state["prior_mean_bias"])
    value["case_success"]=state["reference_success_by_case"]&np.concatenate(list(map(lambda entry:entry["case_success"],fitted)))
    value["view_success_by_case"]=np.ones(len(training_run),dtype=bool)
    value["case_message"]=np.concatenate(list(map(lambda entry:entry["case_message"],fitted)))
    value["native_gradient_by_case"]=np.concatenate(list(map(lambda entry:entry["case_gradient"],fitted)))
    value["native_iterations_by_case"]=np.concatenate(list(map(lambda entry:entry["case_iterations"],fitted)))
    value["native_solver_by_case"]=np.concatenate(list(map(lambda entry:entry["case_solver"],fitted)))
    value["success"]=bool(value["case_success"].all())
    value["iterations"]=sum(map(lambda entry:entry["iterations"],fitted))
    value["gradient"]=max(map(lambda entry:entry["gradient"],fitted))
    return value


POOL_PARAMETER_KEYS=["anchor_coefficient","foreground_log_prior","measurement_coefficient","measurement_duration_coefficient",
    "measurement_log_prior","mean_log_prior","foreground_within_sd","within_scale","own_emission_covariance","background_log_prior"]


def reference_view_features(path,context,period_min,period_max,candidates=73,include_controls=True):
    return pooled_reference_measurement_features(path,context,period_min,period_max,candidates,include_controls)


def _measurement_design(bundle,variant):
    key="identity_base_features" if variant in [1,2] else "cellwise_base_features" if variant==3 else "measurement_base_features"
    base=bundle[key].copy()
    if variant==4:
        base[:,:,4:13]=0.
    poly=_polynomial(base)
    return np.concatenate([poly,poly*bundle["duration_coordinate"][:,None,None]],axis=-1) if variant==12 else poly


def _weighted_period_prior(target,weight,period,effective_units):
    center=weight@target
    variance=np.maximum(weight@(target**2)-center**2,(period[1]-period[0])**2/12.)
    bandwidth=np.maximum(np.sqrt(variance)*(4./(3.*effective_units))**.2,(period[1]-period[0])/np.sqrt(12.))
    log_weight=np.full(weight.shape,-np.inf);np.log(weight,out=log_weight,where=weight>0.)
    value=logsumexp(log_weight[:,:,None]-.5*((period[None,None]-target[None,:,None])/bandwidth[:,None,None])**2,axis=1)
    return value-logsumexp(value,axis=-1,keepdims=True)


def _fit_own_emission(gram,rhs,energy,degrees,weight,effective_n,maxiter,device):
    dtype=torch.float64
    tg=torch.as_tensor(gram,dtype=dtype,device=device);tr=torch.as_tensor(rhs,dtype=dtype,device=device)
    te=torch.as_tensor(energy,dtype=dtype,device=device);td=torch.as_tensor(degrees,dtype=dtype,device=device)
    tw=torch.as_tensor(weight,dtype=dtype,device=device);tn=torch.as_tensor(effective_n,dtype=dtype,device=device)
    def objective(parameter):
        log=parameter.reshape(len(weight),3);h=log.exp();sd=torch.repeat_interleave(h[:,:2],2,dim=-1).sqrt()
        precision=torch.eye(4,dtype=dtype,device=device)[None,None]+tg[None]*sd[:,None,:,None]*sd[:,None,None,:]/h[:,2,None,None,None]
        scaled=tr[None]*sd[:,None]/h[:,2,None,None]
        gain=(scaled*torch.linalg.solve(precision,scaled[:,:,:,None])[:,:,:,0]).sum(-1)
        loss=.5*(td[None]*log[:,2,None]+torch.linalg.slogdet(precision)[1]+te[None]/h[:,2,None]-gain)
        return ((loss*tw).sum(-1)+log.square().sum(-1)/(2.*tn)).sum()
    gradient=torch.func.grad_and_value(objective)
    scipy_objective=lambda parameter:(pair:=gradient(torch.as_tensor(parameter,dtype=dtype,device=device)),(pair[1].item(),pair[0].detach().cpu().numpy()))[-1]
    fit=minimize(scipy_objective,np.zeros(len(weight)*3),jac=True,method="L-BFGS-B",options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-4,"maxcor":25})
    print("OWN_EMISSION_FIT",fit.success,fit.nit,float(np.max(abs(fit.jac))),flush=True)
    return np.exp(fit.x.reshape(len(weight),3)),{"success":bool(fit.success),"iterations":fit.nit,"gradient":float(np.max(abs(fit.jac)))}


def periodic_component_candidates(data,period_min,period_max,seed=12345,band_nuisance=False):
    """Input-only cycle hypotheses for a source-recovery diagnostic."""
    from sklearn.decomposition import FastICA
    from scipy.signal import find_peaks
    values=detrend(data["signal"],data["time"],data["mask"])
    def cohort(group):
        records=np.flatnonzero(data["run_index"]==group)
        mask=data["mask"][records]
        np.testing.assert_array_equal(mask,np.broadcast_to(mask[:1],mask.shape))
        actual=data["time"][mask[0]]
        dt=np.median(np.diff(actual))
        signal=values[records][:,mask[0]]
        if band_nuisance:
            window=actual[-1]-actual[0]+dt
            modes=np.arange(1,int(np.ceil(window/period_max)))
            coordinate=(actual-actual[0])/window
            angle=2.*np.pi*coordinate[:,None]*modes[None]
            nuisance=np.column_stack([np.ones(len(actual)),coordinate,np.cos(angle),np.sin(angle)])
            basis=np.linalg.qr(nuisance,mode="reduced")[0]
            signal-=signal@basis@basis.T
        estimator=FastICA(n_components=len(records),whiten="unit-variance",random_state=seed,max_iter=500,tol=1e-4)
        source=estimator.fit_transform(signal.T).T
        def events(index):
            component=index//3;sign=np.asarray([1.,-1.,0.])[index%3]
            y=abs(source[component]) if sign==0. else source[component]*sign
            peaks,_=find_peaks(y,distance=max(1,int(np.ceil(period_min/dt))),prominence=np.std(y))
            if len(peaks)<3:
                return (group,component,sign,np.nan,np.inf,-np.inf,int(estimator.n_iter_))
            intervals=np.diff(actual[peaks]);initial=np.median(intervals)
            cycles=np.maximum(np.rint(intervals/initial),1.)
            period=intervals.sum()/cycles.sum()
            variance=max(float(np.sum((intervals-cycles*period)**2)/cycles.sum()),dt**2/6.)
            quality=len(intervals)*np.log((period_max-period_min)/np.sqrt(2*np.pi*np.e*variance))-.5*np.sum(np.log(cycles))-.5*np.log(len(intervals))
            return (group,component,sign,float(period),float(np.sqrt(variance)),float(quality),int(estimator.n_iter_))
        return np.asarray(list(map(events,np.arange(len(source)*3))))
    result=np.concatenate(list(map(cohort,np.arange(len(data["runs"])))),axis=0)
    return pd.DataFrame(result,columns=["run_index","component","sign","period","cycle_sd","quality","ica_iterations"])


def observed_cepstral_features(values,data,period):
    """Whole-spectrum fundamental spacing for both measured streams.

    Complete regular observations use their native FFT. Irregular observations
    use the same Fourier frequencies evaluated on their actual timestamps.
    The inverse log spectrum is normalized by its nonzero-quefrency scatter.
    """
    cell,peer=_physical_peer(values,data)
    signal=np.stack([cell,peer],axis=-1)
    output=np.empty((len(cell),len(period),2))
    patterns,owner=np.unique(data["mask"],axis=0,return_inverse=True)
    def acquisition(layout):
        records=np.flatnonzero(owner==layout);actual=data["time"][patterns[layout]]
        spacing=np.diff(actual);window=actual[-1]-actual[0]+np.median(spacing)
        count=len(actual);y=signal[records][:,patterns[layout]]
        taper=np.hanning(count)
        if np.allclose(spacing,spacing[0],rtol=1e-9,atol=abs(spacing[0])*1e-10):
            spectrum=rfft(y*taper[None,:,None],axis=1)
        else:
            frequency=np.arange(count//2+1)/window
            quadrature=np.exp(-2j*np.pi*(actual-actual[0])[:,None]*frequency[None])
            exposure=np.r_[spacing[0],(spacing[:-1]+spacing[1:])/2.,spacing[-1]]
            exposure/=exposure.mean()
            spectrum=np.einsum("ntv,tk,t->nkv",y*taper[None,:,None],quadrature,exposure,optimize=True)
        amplitude=abs(spectrum)
        floor=np.maximum(amplitude.max(axis=1,keepdims=True)*np.finfo(float).eps,np.finfo(float).tiny)
        logarithm=np.log(np.maximum(amplitude,floor))
        logarithm[:,0]=logarithm[:,1:].mean(axis=1)
        cepstrum=irfft(logarithm,n=count,axis=1)
        scatter=np.sqrt(np.mean(cepstrum[:,1:count//2]**2,axis=1))
        cepstrum=np.divide(cepstrum,scatter[:,None],out=np.zeros_like(cepstrum),where=scatter[:,None]>0.)
        coordinate=period*count/window
        left=np.floor(coordinate).astype(int)
        valid=coordinate<count/2.
        left=np.clip(left,0,count-2);fraction=coordinate-left
        value=cepstrum[:,left]*(1.-fraction)[None,:,None]+cepstrum[:,left+1]*fraction[None,:,None]
        output[records]=value*valid[None,:,None]
        return None
    list(map(acquisition,np.arange(len(patterns))))
    return output


def cohort_recurrence_spectrum(values,data,period,non_gaussian=False):
    """Regularized multichannel recurrence on every observed time coordinate.

    Candidate-specific paired Gram matrices use all available channel pairs.
    Positive OAS ridge bounds unstable source directions at short acquisitions.
    The two largest eigenvalues retain full-waveform recurrence information.
    """
    values=detrend(np.asarray(values,dtype=float),data["time"],data["mask"])
    rms=np.sqrt(np.sum(values**2,axis=1)/data["count"])
    values=np.divide(values,rms[:,None],out=np.zeros_like(values),where=rms[:,None]>0.)
    scale=np.sqrt(period.min()*period.max())
    t=(data["time"]-data["time"][0])/scale
    periods=period/scale
    nodes,quadrature=np.polynomial.hermite.hermgauss(32)
    gaussian_logcosh=np.sum(quadrature*(np.logaddexp(np.sqrt(2.)*nodes,-np.sqrt(2.)*nodes)-np.log(2.)))/np.sqrt(np.pi)
    output=np.empty((len(data["runs"]),len(period),2))
    def cohort(group):
        records=np.flatnonzero(data["run_index"]==group)
        y=values[records];mask=data["mask"][records]
        shifted=np.zeros((len(records),len(period),len(t)))
        weight=np.zeros_like(shifted)
        patterns,owner=np.unique(mask,axis=0,return_inverse=True)
        def acquisition(layout):
            selected=np.flatnonzero(owner==layout)
            observed=np.flatnonzero(patterns[layout]);actual=t[observed]
            query=actual[None]+periods[:,None]
            tolerance=np.finfo(float).eps*32.*max(1.,float(np.max(abs(t))))
            valid=query<=actual[-1]+tolerance
            query=np.minimum(query,actual[-1])
            right=np.clip(np.searchsorted(actual,query),1,len(actual)-1);left=right-1
            at_left=abs(query-actual[left])<=tolerance;at_right=abs(query-actual[right])<=tolerance
            query=np.where(at_left,actual[left],np.where(at_right,actual[right],query))
            gap=actual[right]-actual[left]
            fraction=np.clip((query-actual[left])/gap,0.,1.)
            reliability=valid*np.where(at_left|at_right,1.,np.median(np.diff(actual))/gap)
            signal=y[selected][:,observed]
            interpolation=signal[:,left]*(1.-fraction)[None]+signal[:,right]*fraction[None]
            address=(selected[:,None,None],np.arange(len(period))[None,:,None],observed[None,None])
            shifted[address]=interpolation
            weight[address]=reliability[None]
            return None
        list(map(acquisition,np.arange(len(patterns))))
        number=weight.sum(-1)
        inverse=np.divide(1.,number,out=np.zeros_like(number),where=number>0.)
        mean0=np.sum(y[:,None]*weight,axis=-1)*inverse
        mean1=np.sum(shifted*weight,axis=-1)*inverse
        factor=np.sqrt(weight*inverse[:,:,None])
        first=(y[:,None]-mean0[:,:,None])*factor
        second=(shifted-mean1[:,:,None])*factor
        zero=(np.einsum("cpt,dpt->pcd",first,first,optimize=True)+np.einsum("cpt,dpt->pcd",second,second,optimize=True))/2.
        cross=np.einsum("cpt,dpt->pcd",first,second,optimize=True)
        lag=(cross+cross.swapaxes(-1,-2))/2.
        trace=np.trace(zero,axis1=-2,axis2=-1);square=np.sum(zero**2,axis=(-1,-2));channels=len(records)
        denominator=(number.mean(0)+1.)*(square-trace**2/channels)
        alpha=np.clip(np.divide(square+trace**2,denominator,out=np.ones_like(trace),where=denominator>0.),0.,1.)
        eigen,basis=np.linalg.eigh(zero)
        regularized=(1.-alpha[:,None])*np.maximum(eigen,0.)+alpha[:,None]*trace[:,None]/channels
        inverse_root=np.sqrt(np.divide(1.-alpha[:,None],regularized,out=np.zeros_like(eigen),where=regularized>0.))
        root=np.einsum("pik,pk,pjk->pij",basis,inverse_root,basis,optimize=True)
        recurrence=root@lag@root
        result,vectors=np.linalg.eigh((recurrence+recurrence.swapaxes(-1,-2))/2.)
        if non_gaussian:
            coefficient=root@vectors
            projection=np.einsum("pck,cpt->pkt",coefficient,first,optimize=True)
            available=weight.mean(axis=0)
            total=available.sum(-1)
            center=np.divide(np.sum(projection*available[:,None],axis=-1),total[:,None],out=np.zeros(projection.shape[:2]),where=total[:,None]>0.)
            projection-=center[:,:,None]
            variance=np.divide(np.sum(projection**2*available[:,None],axis=-1),total[:,None],out=np.zeros(projection.shape[:2]),where=total[:,None]>0.)
            standardized=np.divide(projection,np.sqrt(np.maximum(variance,0.))[:,:,None],out=np.zeros_like(projection),where=variance[:,:,None]>0.)
            logcosh=np.logaddexp(standardized,-standardized)-np.log(2.)
            expectation=np.divide(np.sum(logcosh*available[:,None],axis=-1),total[:,None],out=np.zeros(projection.shape[:2]),where=total[:,None]>0.)
            contrast=(expectation-gaussian_logcosh)**2
            score=np.maximum(result,0.)**2*contrast
            selected=score.argmax(1)
            output[group]=np.stack([score[np.arange(len(period)),selected],result[np.arange(len(period)),selected]],axis=-1)
        else:
            output[group]=np.clip(result[:,-2:][:,::-1],-1.,1.)
        return None
    list(map(cohort,np.arange(len(data["runs"]))))
    return output[data["run_index"]]


def lag_source_features(values,data,period,record_chunk=4):
    """Candidate-period source recurrence in the cell/peer measurement plane.

    The symmetric lag covariance and paired zero-lag covariance are congruent
    under a nonsingular mixing change. Their generalized eigenvalues therefore
    measure recurrence without fitting a waveform or fixing source amplitudes.
    The selected projection is also evaluated at half and twice its period.
    """
    cell,peer=_physical_peer(values,data)
    signals=np.stack([cell,peer],axis=-1)
    periods=np.concatenate([period,period/2.,period*2.])
    output=np.empty((len(cell),len(period),4))
    patterns,owner=np.unique(data["mask"],axis=0,return_inverse=True)
    def acquisition(layout):
        records=np.flatnonzero(owner==layout)
        actual=data["time"][patterns[layout]]
        query=actual[None]+periods[:,None]
        right=np.searchsorted(actual,query)
        valid=right<len(actual)
        right=np.clip(right,1,len(actual)-1);left=right-1
        gap=actual[right]-actual[left]
        fraction=(query-actual[left])/gap
        weight=valid*np.median(np.diff(actual))/gap
        number=weight.sum(1)
        def block(indices):
            x=signals[indices][:,patterns[layout]]
            shifted=x[:,left]*(1.-fraction)[None,:,:,None]+x[:,right]*fraction[None,:,:,None]
            first=np.einsum("lt,bti->bli",weight,x,optimize=True)
            second=np.einsum("lt,blti->bli",weight,shifted,optimize=True)
            inverse=np.divide(1.,number,out=np.zeros_like(number),where=number>0.)
            cov0=np.einsum("bti,lt,btj->blij",x,weight,x,optimize=True)-first[:,:,:,None]*first[:,:,None,:]*inverse[None,:,None,None]
            cov1=np.einsum("blti,lt,bltj->blij",shifted,weight,shifted,optimize=True)-second[:,:,:,None]*second[:,:,None,:]*inverse[None,:,None,None]
            cross=np.einsum("bti,lt,bltj->blij",x,weight,shifted,optimize=True)-first[:,:,:,None]*second[:,:,None,:]*inverse[None,:,None,None]
            zero=(cov0+cov1)/2.
            lag=(cross+cross.swapaxes(-1,-2))/2.
            eigen,basis=np.linalg.eigh((zero+zero.swapaxes(-1,-2))/2.)
            tolerance=np.finfo(float).eps*len(actual)*np.maximum(eigen[:,:,-1],np.finfo(float).tiny)
            inverse_root=np.divide(1.,np.sqrt(np.maximum(eigen,0.)),out=np.zeros_like(eigen),where=eigen>tolerance[:,:,None])
            root=np.einsum("blik,blk,bljk->blij",basis,inverse_root,basis,optimize=True)
            recurrence=root@lag@root
            value,vector=np.linalg.eigh((recurrence+recurrence.swapaxes(-1,-2))/2.)
            p=len(period)
            coefficient=np.einsum("bpij,bpj->bpi",root[:,:p],vector[:,:p,:,-1],optimize=True)
            half=np.einsum("bpi,bpij,bpj->bp",coefficient,lag[:,p:2*p],coefficient,optimize=True)
            half_variance=np.einsum("bpi,bpij,bpj->bp",coefficient,zero[:,p:2*p],coefficient,optimize=True)
            twice=np.einsum("bpi,bpij,bpj->bp",coefficient,lag[:,2*p:],coefficient,optimize=True)
            twice_variance=np.einsum("bpi,bpij,bpj->bp",coefficient,zero[:,2*p:],coefficient,optimize=True)
            half=np.divide(half,half_variance,out=np.zeros_like(half),where=half_variance>0.)
            twice=np.divide(twice,twice_variance,out=np.zeros_like(twice),where=twice_variance>0.)
            output[indices]=np.clip(np.stack([value[:,:p,-1],value[:,:p,0],half,twice],axis=-1),-1.,1.)
            return None
        list(map(block,np.array_split(records,max(1,int(np.ceil(len(records)/record_chunk))))))
        return None
    list(map(acquisition,np.arange(len(patterns))))
    return output


def covariance_source_balance(values,data):
    """Regularized symmetric whitening preserves full observed waveforms.

    Covariance is a Gram matrix on observed coordinates. Its OAS shrinkage uses
    the effective common observation count, with no specified number of clocks,
    period, sampling rate, source waveform, or source ordering.
    """
    values=detrend(np.asarray(values,dtype=float),data["time"],data["mask"])
    output=np.zeros_like(values)
    shrinkage=np.empty(len(data["runs"]));rank=np.empty_like(shrinkage)
    def cohort(group):
        selected=np.flatnonzero(data["run_index"]==group)
        mask=data["mask"][selected];number=len(selected)
        x=values[selected]/np.sqrt(data["count"][selected,None])
        covariance=x@x.T
        trace=float(np.trace(covariance));square=float(np.sum(covariance**2));mu=trace/number
        effective=float(np.sum(mask.sum(0,dtype=float)**2)/number**2)
        denominator=(effective+1.)*(square-trace**2/number)
        alpha=np.clip((square+trace**2)/max(denominator,np.finfo(float).tiny),0.,1.)
        eigen,basis=np.linalg.eigh(covariance)
        regularized=(1.-alpha)*np.maximum(eigen,0.)+alpha*mu
        floor=np.finfo(float).eps*max(mu,np.finfo(float).tiny)
        whitening=(basis/np.sqrt(np.maximum(regularized,floor))[None])@basis.T
        transformed=whitening@values[selected]
        availability=np.sqrt((whitening**2)@mask.astype(float))
        full=np.sqrt(np.sum(whitening**2,axis=1))
        transformed=np.divide(transformed*full[:,None],availability,out=np.zeros_like(transformed),where=availability>0.)
        transformed*=mask
        output[selected]=detrend(transformed,data["time"],mask)
        shrinkage[group]=alpha
        rank[group]=trace**2/max(square,np.finfo(float).tiny)
        return None
    list(map(cohort,np.arange(len(data["runs"]))))
    return {"signal":output,"covariance_shrinkage":shrinkage,"covariance_effective_rank":rank}


def observed_fundamental_support(values,peer,data,period,scale,tau):
    """The validated nested H1-plus-H2 source test on observed OU geometry."""
    from scipy.special import expit
    t=(data["time"]-data["time"][0])/scale
    angle=2*np.pi*t[None,:,None]*np.arange(1,3)[None,None]/(period/scale)[:,None,None]
    raw=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(len(period),len(t),4)
    patterns,owner=np.unique(data["mask"],axis=0,return_inverse=True)
    geometry,layout=np.unique(np.column_stack([owner,tau]),axis=0,return_inverse=True)
    signal=np.stack([values,peer],axis=-1);result=np.empty((len(values),len(period),2))
    def acquisition(index):
        records=np.flatnonzero(layout==index);mask=patterns[int(geometry[index,0])][None];count=int(mask.sum())
        current_tau=np.asarray([geometry[index,1]])
        center=(t-(mask*t[None]).sum()/count)*mask[0]
        z=np.stack([mask[0]/np.sqrt(count),center/np.sqrt(np.sum(center**2))],axis=-1)
        wz=ou_whiten(z[None],data["time"],mask,current_tau)[0]
        inverse=np.linalg.inv(wz.T@wz)
        h=ou_whiten(raw.transpose(1,0,2).reshape(len(t),-1)[None],data["time"],mask,current_tau)[0]
        h-=wz@(inverse@(wz.T@h));h=h.reshape(len(t),len(period),4).transpose(1,0,2)
        y=ou_whiten(signal[records],data["time"],data["mask"][records],tau[records])
        y-=np.einsum("ti,niv->ntv",wz,np.einsum("ij,tj,ntv->niv",inverse,wz,y,optimize=True),optimize=True)
        gram=np.einsum("pti,ptj->pij",h,h,optimize=True);rhs=np.einsum("pti,ntv->npiv",h,y,optimize=True)
        full=np.einsum("npiv,pij,npjv->npv",rhs,np.linalg.pinv(gram,hermitian=True),rhs,optimize=True)
        half=np.einsum("npiv,pij,npjv->npv",rhs[:,:,2:],np.linalg.pinv(gram[:,2:,2:],hermitian=True),rhs[:,:,2:],optimize=True)
        energy=np.sum(y**2,axis=1)
        partial=np.clip(np.divide(full-half,energy[:,None]-half,out=np.zeros_like(full),where=(energy[:,None]-half)>0.),0.,1.)
        degree=count-4.
        log_bf=-np.log1p(degree)-.5*degree*np.log1p(-degree/(1.+degree)*partial)
        result[records]=expit(log_bf)
        return None
    list(map(acquisition,np.arange(len(geometry))))
    return result


def raw_background_features(frame,period,scale):
    """Measured collective-reference terms, before source canonicalization."""
    data=observations(frame)
    geometry=_role_geometry(data["signal"],data,period,scale,peer_reference=data["signal"],normalize_reference_peer=True)
    fitted=_source_clock_fit(geometry["unit"],data,period,scale)
    tau,_=_source_noise_tau(fitted["two_residual"],data,scale)
    base=profile_features(data["signal"],data,period,scale,peer_reference=data["signal"],normalize_reference_peer=True)[0]
    support=observed_fundamental_support(geometry["unit"],geometry["peer"],data,period,scale,tau)
    base[:,:,5]*=support[:,:,1]
    return {"trace_id":data["metadata"].index.to_numpy(dtype=str),"features":_background_features(base),"noise_tau":tau}


def general_background_reference_features(path,context,period,view_ids):
    """Training-only measured-peer views with origin-level exclusions."""
    scale=np.sqrt(period.min()*period.max())
    def view(name):
        frame=pd.read_parquet(path,columns=["trace_id","run","condition","x","signal_y"],filters=[("view_id","=",str(name))])
        print("BACKGROUND_REFERENCE_VIEW",str(name),flush=True)
        return raw_background_features(frame,period,scale)
    parts=list(map(view,view_ids))
    names=np.concatenate(list(map(lambda item:item["trace_id"],parts)))
    order=pd.Index(context["trace_id"]).get_indexer(names)
    assert np.all(order>=0)
    result={key:context[key][order] for key in ["origin_trace_id","origin_run","is_original","qualified","view_id","background_reference"]}
    result.update({"trace_id":names,"features":np.concatenate(list(map(lambda item:item["features"],parts))),
        "noise_tau":np.concatenate(list(map(lambda item:item["noise_tau"],parts)))})
    result["background_is_proxy"]=~np.isfinite(result["background_reference"])
    result["background_reference"]=np.where(result["background_is_proxy"],period[result["features"][:,:,0].argmax(1)],result["background_reference"])
    result["background_index"]=abs(period[None]-result["background_reference"][:,None]).argmin(1)
    groups=np.char.add(np.char.add(result["view_id"].astype(str),":"),result["origin_run"].astype(str))
    code,names=pd.factorize(groups,sort=True)
    membership=(np.arange(len(names))[:,None]==code[None]).astype(float)
    energy=(membership/membership.sum(1)[:,None])@result["features"][:,:,0]
    result["mixing_quality"]=np.clip(energy.max(1),0.,1.)[code]
    return result


def posterior_clock_role_features(frame,bundle=None,period_min=12.,period_max=48.,candidates=73,period_scale=None,source_balancing=True,noise_cache=None,paired_core=False):
    """One frequency-marginal source operator with the measured peer retained."""
    data=observations(frame)
    if bundle is None:
        period=np.linspace(period_min,period_max,candidates);scale=np.sqrt(period_min*period_max) if period_scale is None else float(period_scale)
        metadata=data["metadata"]
        bundle={"period":period,"period_scale":np.asarray(scale),"trace_id":metadata.index.to_numpy(dtype=str),
            "run":metadata.run.to_numpy(dtype=str),"condition":metadata.condition.to_numpy(dtype=str),"count":data["count"],
            "time":data["time"],"mask":data["mask"],"run_names":data["runs"],"run_index":data["run_index"],
            "membership":data["membership"],"average":data["average"],"identity_base_features":profile_features(data["signal"],data,period,scale)[0]}
    period=bundle["period"];scale=float(bundle["period_scale"])
    if source_balancing:
        original,adjusted=shared_background(data,scale)
        source,diagnostic=canonical_source_signal(adjusted,data,period,scale,share_order=False,marginal_order=False)
        representation="frequency_marginal_retained_peer";policy="observed_OU_BIC_per_record"
    else:
        source=data["signal"];adjusted=source
        unit=detrend(source,data["time"],data["mask"])
        rms=np.sqrt(np.sum(unit**2,axis=1)/data["count"])
        unit=np.divide(unit,rms[:,None],out=np.zeros_like(unit),where=rms[:,None]>0.)
        if noise_cache is None:
            fitted=_source_clock_fit(unit,data,period,scale)
            noise_tau,_=_source_noise_tau(fitted["two_residual"],data,scale)
        else:
            order=pd.Index(noise_cache["trace_id"]).get_indexer(data["metadata"].index)
            assert np.all(order>=0)
            noise_tau=noise_cache["noise_tau"][order]
        diagnostic={"noise_tau":noise_tau,"two_clock_probability":np.full(len(source),np.nan)}
        representation="observed_physical_peer";policy="observed_geometry_gaussian_roles"
    geometry=_role_geometry(source,data,period,scale,peer_reference=adjusted,normalize_reference_peer=True)
    base=profile_features(source,data,period,scale,peer_reference=adjusted,normalize_reference_peer=True)[0]
    support=observed_fundamental_support(geometry["unit"],geometry["peer"],data,period,scale,diagnostic["noise_tau"])
    base[:,:,1]*=support[:,:,0];base[:,:,5]*=support[:,:,1]
    raw_geometry=_role_geometry(data["signal"],data,period,scale)
    result=bundle.copy()
    result["mixing_quality"]=measured_mixing_quality(data,period,scale)
    result.update({"base_features":bundle["identity_base_features"],"original_unit":raw_geometry["unit"],
        "original_peer":raw_geometry["peer"],"original_rhs":raw_geometry["rhs"],"harmonic_design":raw_geometry["design"],
        "observed_signal":data["signal"],"original_coherent_power":generic_coherent_fwl_power(data["time"],data["mask"],raw_geometry["unit"],raw_geometry["peer"],period,half_gram=True)})
    result.update({"source_base":base,"source_rhs":geometry["rhs"],"joint_gram":geometry["joint"],"layout":geometry["layout"],
        "source_coherent_power":generic_coherent_fwl_power(data["time"],data["mask"],geometry["unit"],geometry["peer"],period,half_gram=True),
        "source_unit":geometry["unit"],"source_peer":geometry["peer"],
        "source_model_probability":diagnostic["two_clock_probability"],"source_representation":np.asarray(representation),
        "source_order_policy":np.asarray(policy)})
    if not source_balancing:
        result.update({"base_features":base,"original_unit":geometry["unit"],"original_peer":geometry["peer"],
            "original_rhs":geometry["rhs"],"harmonic_design":geometry["design"],"original_coherent_power":result["source_coherent_power"]})
    observation=np.stack([geometry["unit"],geometry["peer"]],axis=-1)
    result["source_observation_energy"]=np.einsum("nti,ntj->nij",observation,observation,optimize=True)
    physical=detrend(adjusted,data["time"],data["mask"])
    cohort_rms=np.sqrt(data["average"]@(np.sum(physical**2,axis=1)/data["count"]))
    _,physical_peer=_physical_peer(adjusted,data)
    peer_rms=np.sqrt(np.sum(physical_peer**2,axis=1)/data["count"])
    if source_balancing:
        own_scale=cohort_rms[data["run_index"]]/(diagnostic["operator_input_rms"]*diagnostic["operator_output_rms"])
    else:
        own_rms=np.sqrt(np.sum(detrend(source,data["time"],data["mask"])**2,axis=1)/data["count"])
        own_scale=cohort_rms[data["run_index"]]/own_rms
    result["source_pair_scale"]=np.column_stack([own_scale,1./peer_rms])
    available=data["membership"]@data["mask"].astype(float)
    denominator=available[data["run_index"]]-data["mask"]
    peer_square=np.divide(1.,denominator,out=np.zeros_like(denominator),where=denominator>0.)
    result["source_peer_weight_square"]=np.sum(peer_square*data["mask"],axis=1)/data["count"]
    if paired_core:
        companion=posterior_clock_role_features(frame,bundle=bundle,source_balancing=True,paired_core=False)
        result["companion_base_features"]=companion["source_base"]
        result["companion_coherent_power"]=companion["source_coherent_power"]
        result["companion_rhs"]=companion["source_rhs"][:,:,:,0]
    return result


def general_source_role_features(frame,bundle,include_original=True):
    """The validated role and coherent-projection measurements on actual masks."""
    data=observations(frame);period=bundle["period"];scale=float(bundle["period_scale"])
    np.testing.assert_array_equal(data["metadata"].index.to_numpy(),bundle["trace_id"])
    source_values=bundle["balanced_signal"] if "balanced_signal" in bundle else bundle["pooled_unit"]
    source=_role_geometry(source_values,data,period,scale)
    result=bundle.copy()
    result["mixing_quality"]=measured_mixing_quality(data,period,scale)
    result.update({"base_features":bundle["identity_base_features"],"source_base":bundle["measurement_base_features"],
        "source_rhs":source["rhs"],"joint_gram":source["joint"],"layout":source["layout"],"harmonic_design":source["design"],"observed_signal":data["signal"]})
    if include_original:
        original=_role_geometry(data["signal"],data,period,scale)
        result.update({"original_unit":original["unit"],"original_peer":original["peer"],"original_rhs":original["rhs"]})
        result["original_coherent_power"]=generic_coherent_fwl_power(data["time"],data["mask"],original["unit"],original["peer"],period,half_gram=True)
    result["source_coherent_power"]=generic_coherent_fwl_power(data["time"],data["mask"],source["unit"],source["peer"],period,half_gram=True)
    return result


def reference_cepstral_features(path,view_id,period,trace_id):
    """Match whole-spectrum measurements to the existing reference-view rows."""
    def view(name):
        data=observations(pd.read_parquet(path,columns=["trace_id","run","condition","x","signal_y"],filters=[("view_id","=",str(name))]))
        print("CEPSTRAL_REFERENCE_VIEW",str(name),len(data["signal"]),flush=True)
        return data["metadata"].index.to_numpy(dtype=str),observed_cepstral_features(data["signal"],data,period)
    parts=list(map(view,np.unique(view_id)))
    names=np.concatenate(list(map(lambda item:item[0],parts)))
    features=np.concatenate(list(map(lambda item:item[1],parts)),axis=0)
    order=pd.Index(names).get_indexer(trace_id)
    assert np.all(order>=0) and len(np.unique(order))==len(names)
    return features[order]


def general_reference_role_features(path,context,period,clock_cache=None,posterior_clock=False,source_balancing=True,noise_cache=None,paired_core=False):
    """Compact role measurements for the existing declared waveform views."""
    tags=np.unique(context["view_id"])
    keys=["trace_id","run","condition","count","source_base","source_rhs","source_coherent_power","mixing_quality"]
    if paired_core:
        keys+=["companion_base_features","companion_coherent_power","companion_rhs"]
    if posterior_clock:
        keys+=["source_observation_energy","source_pair_scale","source_peer_weight_square"]
    def view(tag):
        print("ROLE_REFERENCE_VIEW",str(tag),flush=True)
        frame=pd.read_parquet(path,columns=["trace_id","run","condition","x","signal_y"],filters=[("view_id","=",str(tag))])
        if posterior_clock:
            current_noise=None
            if noise_cache is not None and str(tag) in noise_cache["view_id"]:
                current_noise=noise_cache
            current=posterior_clock_role_features(frame,period_min=float(period.min()),period_max=float(period.max()),candidates=len(period),source_balancing=source_balancing,noise_cache=current_noise,paired_core=paired_core)
        else:
            measured=pooled_measurement_features(frame,float(period.min()),float(period.max()),len(period),
                include_anchor=clock_cache is None,include_controls=False,clock_cache=clock_cache,include_measurement=clock_cache is None)
            if clock_cache is not None:
                order=pd.Index(clock_cache["trace_id"]).get_indexer(measured["trace_id"])
                assert np.all(order>=0)
                measured["identity_base_features"]=clock_cache["identity_base_features"][order]
                measured["measurement_base_features"]=clock_cache["measurement_base_features"][order]
            current=general_source_role_features(frame,measured,include_original=False)
        return {key:current[key] for key in keys+["joint_gram","layout"]}
    parts=list(map(view,tags))
    result={key:np.concatenate(list(map(lambda item:item[key],parts)),axis=0) for key in keys}
    sizes=np.asarray(list(map(lambda item:len(item["joint_gram"]),parts)))
    offset=np.r_[0,np.cumsum(sizes)[:-1]]
    result["layout"]=np.concatenate(list(map(lambda item:item[0]["layout"]+item[1],zip(parts,offset))))
    result["joint_gram"]=np.concatenate(list(map(lambda item:item["joint_gram"],parts)),axis=0)
    code,runs=pd.factorize(result["run"],sort=True)
    member=(np.arange(len(runs))[:,None]==code[None]).astype(float)
    result.update({"run_index":code,"run_names":np.asarray(runs,dtype=str),"membership":member,"average":member/member.sum(1)[:,None],
        "period":period,"period_scale":np.asarray(np.sqrt(period.min()*period.max()))})
    order=pd.Index(context["trace_id"]).get_indexer(result["trace_id"])
    selected={key:context[key][order] for key in ["target","qualified","origin_run","origin_trace_id","independent_unit","view_weight","is_original","view_id"]}
    names=pd.Index(result["run_names"])
    left=context["pair_left_run"].ravel();right=context["pair_right_run"].ravel()
    eligible=context["pair_eligible"].ravel()&(left>=0)&(right>=0)
    selected["pair_left"]=names.get_indexer(context["run_name"][np.maximum(left,0)])
    selected["pair_right"]=names.get_indexer(context["run_name"][np.maximum(right,0)])
    selected["pair_eligible"]=eligible&(selected["pair_left"]>=0)&(selected["pair_right"]>=0)
    selected["pair_left"]=np.maximum(selected["pair_left"],0)
    selected["pair_right"]=np.maximum(selected["pair_right"],0)
    effect=context["pair_effect"].ravel()/float(result["period_scale"])
    selected["pair_eligible"]&=np.isfinite(effect)
    selected["pair_effect"]=np.where(selected["pair_eligible"],effect,0.)
    assert np.all(order>=0) and len(np.unique(order))==len(context["trace_id"])
    return result,selected


def refine_general_source_views(bundle,view_context,parent,original_bundle,reference,qualified,training_run,maxiter=900,device="cuda",source_invariance=False,refine_core=True,evidence_cache=None):
    """One joint source, cohort-mean and exact-gain-contrast training objective."""
    from source_assignment_core import role_assignment_evidence
    period=bundle["period"];scale=float(bundle["period_scale"])
    origins=pd.DataFrame({"run":bundle["run"],"origin":view_context["origin_run"]}).drop_duplicates().set_index("run").loc[bundle["run_names"],"origin"]
    origin_index=pd.Index(original_bundle["run_names"]).get_indexer(origins)
    assert np.all(origin_index>=0)
    view_training=training_run[:,origin_index]
    weights=source_reference_weights(bundle,view_context,view_training)
    scale_allowed=weights["train"]&view_context["is_original"][None]
    scale_weight=scale_allowed/scale_allowed.sum(1)[:,None]
    view_code,view_names=pd.factorize(view_context["view_id"],sort=True)
    view_member=(np.arange(len(view_names))[:,None]==view_code[None]).astype(float)
    view_count=weights["train"].astype(float)@view_member.T
    view_share=np.divide((weights["train"]*view_context["view_weight"][None])@view_member.T,view_count,out=np.zeros_like(view_count),where=view_count>0.)
    proper_weight=weights["train"]*view_share[:,view_code]/np.maximum(view_count[:,view_code],1.)
    proper_weight/=proper_weight.sum(1)[:,None]
    count=bundle["membership"]@view_context["qualified"].astype(float)
    target=np.where(view_context["qualified"],view_context["target"]/scale,0.)
    means=np.divide(bundle["membership"]@target,count,out=np.full(len(count),np.nan),where=count>0.)
    base=bundle["source_base"];coherent=base.copy();coherent[:,:,11]=bundle["source_coherent_power"]
    quality=model_role_quality(bundle,parent)
    paired=str(parent.get("core_measurements","single"))=="paired"
    design=source_core_design(bundle,coherent,quality,coherent=True,paired=paired)
    role_prior=parent["role_coefficient"]
    reference_count=parent.get("reference_source_count",weights["effective_sample_size"])
    if "source_shape_features" in bundle:
        old_dimension=2;dimension=old_dimension+bundle["source_shape_features"].shape[-1];base_dimension=base.shape[-1]
        old_left,old_right=np.triu_indices(old_dimension);left,right=np.triu_indices(dimension)
        mapping=np.r_[np.arange(old_dimension),dimension+np.arange(old_dimension*base_dimension),
            dimension*(base_dimension+1)+np.searchsorted(left*dimension+right,old_left*dimension+old_right)]
        role_prior=np.zeros((len(training_run),dimension*(base_dimension+1)+len(left)))
        role_prior[:,mapping]=parent["role_coefficient"]
    original_groups=(bundle["membership"]@view_context["is_original"].astype(float))>0.
    original_index=pd.Index(bundle["trace_id"]).get_indexer(original_bundle["trace_id"])
    run_index=pd.Index(bundle["run_names"]).get_indexer(original_bundle["run_names"])
    assert np.all(original_index>=0) and np.all(run_index>=0)
    cache={} if evidence_cache is None else evidence_cache
    def case(index):
        kind=str(parent.get("role_kind","gaussian"))
        key=(id(base),kind,parent["background_coefficient"][index].tobytes(),parent["role_covariance"][index].tobytes(),
            parent["core_coefficient"][index].tobytes() if kind=="density" else b"",
            foreground_reference_log_prior(bundle,parent,np.asarray([index])).tobytes() if kind=="density" else b"",str(parent.get("role_reduction","expected_bf")))
        if key not in cache:
            background=softmax(np.einsum("npd,d->np",quality_design(_polynomial(base),quality),parent["background_coefficient"][index],optimize=True),axis=-1)[None]
            cache[key]=model_role_evidence(bundle,parent,background,np.asarray([index]))[0]
        role=cache[key]
        if "source_shape_features" in bundle:
            role=np.concatenate([role,bundle["source_shape_features"][None]],axis=-1)
        core=parent["core_coefficient"][index:index+1]
        fixed=foreground_reference_log_prior(bundle,parent,np.asarray([index]))[:,None]+np.einsum("npd,fd->fnp",design,core,optimize=True)
        allowed=training_run[index,original_bundle["run_index"]]&qualified
        y=np.where(qualified,reference/scale,0.)
        w=allowed/allowed.sum()
        sd=np.asarray([np.sqrt(max(w@(y**2)-(w@y)**2,(period[1]-period[0])**2/(12*scale**2)))])
        context=dict(view_context,group_weight=weights["group_weight"][index:index+1])
        origin_weight=context["group_weight"]*original_groups[None]
        context["origin_group_weight"]=origin_weight/origin_weight.sum(1)[:,None]
        left=context["pair_left"];right=context["pair_right"]
        eligible=context["pair_eligible"]&view_training[index,left]&view_training[index,right]
        pair_weight=(context["group_weight"][:,left]+context["group_weight"][:,right])*eligible[None]/2.
        context["pair_weight"]=pair_weight/np.maximum(pair_weight.sum(1),np.finfo(float).tiny)[:,None]
        context["pair_scale"]=sd[:,None]
        context["pair_readout"]="source" if source_invariance else "mean"
        context["proper_ranking"]=True
        context["mean_location"]=False
        identity=(period[[0,-1]]/scale)[None]
        fitted=_fit_mean_role(bundle,base,role,fixed,role_prior[index:index+1],means,view_training[index:index+1],
            (identity,identity,np.asarray([2])),
            sd,reference_count[index:index+1],maxiter,device,context,
            period_reference=(target,view_context["qualified"]),core_fit=(design,core,None) if refine_core else None,
            prior_scale_weight=scale_weight[index:index+1],proper_weight_override=proper_weight[index:index+1],
            interaction_units=reference_count[index:index+1])
        print("GENERAL_SOURCE_VIEW_CASE",index,fitted["success"],flush=True)
        return {"core_coefficient":fitted["core_coefficient"][0] if refine_core else core[0],"role_coefficient":fitted["role_coefficient"][0],
            "mean_bias":fitted["mean_bias"][0],"success":np.asarray(fitted["success"]),
            "source_period":fitted["source_period"][0,original_index]*scale,"posterior":fitted["posterior"][0,original_index],
            "group_mean":fitted["group_mean"][0,run_index]*scale}
    fitted=list(map(case,np.arange(len(training_run))))
    result={key:value.copy() for key,value in parent.items()}
    result.update({key:np.stack(list(map(lambda item:item[key],fitted))) for key in fitted[0]})
    result["success"]&=parent["success"]
    membership=original_bundle["membership"].astype(bool)
    minimum=np.min(np.where(membership&qualified[None],reference[None],np.inf),axis=1)
    maximum=np.max(np.where(membership&qualified[None],reference[None],-np.inf),axis=1)
    zero_within=~np.any(training_run&(maximum>minimum)[None],axis=1)
    result["within_scale"]=np.where(zero_within,0.,result["within_scale"])
    uncalibrated=(result["source_period"]/scale)@original_bundle["average"].T
    mean=np.clip((_curve_numpy(uncalibrated,parent["calibration_prediction"],parent["calibration_mean"],parent["calibration_experiments"])+result["mean_bias"][:,None])*scale,period.min(),period.max())
    raw=result["original_period"];average=original_bundle["average"];run=original_bundle["run_index"]
    point=mean[:,run]+result["within_scale"][:,None]*(raw-(raw@average.T)[:,run])
    result["estimate"]=project_periods_with_mean(point,mean,original_bundle["membership"],period.min(),period.max())
    result["group_mean"]=mean
    result["role_shape_kind"]=np.asarray("cepstrum" if "source_shape_features" in bundle else "none")
    result["gain_contrast_readout"]=np.asarray("source" if source_invariance else "mean")
    result["core_refinement"]=np.asarray(refine_core)
    result["source_prior_readout"]=np.asarray("uncalibrated_period_expectation")
    result["source_prior_reference_count"]=reference_count
    result["source_prior_observation_count"]=weights["effective_sample_size"]
    return result


def calibrate_general_source_measurement(bundle,parent,reference,qualified,training_run,independent_unit,maxiter=900,device="cuda",direct_mean=False,mean_only=False,identity_mean=False):
    """Match the same reference law to the adopted physical source measurement."""
    from source_assignment_core import role_assignment_evidence
    period=bundle["period"];scale=float(bundle["period_scale"])
    base=bundle["source_base"];coherent=base.copy();coherent[:,:,11]=bundle["source_coherent_power"]
    quality=model_role_quality(bundle,parent)
    paired=str(parent.get("core_measurements","single"))=="paired"
    design=source_core_design(bundle,coherent,quality,coherent=True,paired=paired)
    label=np.where(qualified,reference/scale,0.)
    number=bundle["membership"]@qualified.astype(float)
    target=np.divide(bundle["membership"]@label,number,out=np.full(len(number),np.nan),where=number>0.)
    codes,names=pd.factorize(independent_unit,sort=True)
    unit=(np.arange(len(names))[:,None]==codes[None]).astype(float)
    group_unit=(bundle["membership"]@unit.T)>0.
    identity=direct_mean or identity_mean
    curve_x=np.broadcast_to((period[[0,-1]]/scale)[None],(len(training_run),2)).copy() if identity else parent["calibration_prediction"]
    curve_y=curve_x.copy() if identity else parent["calibration_mean"]
    curve_n=np.full(len(training_run),2,dtype=int) if identity else parent["calibration_experiments"]
    def case(index):
        bg=softmax(np.einsum("npd,d->np",quality_design(_polynomial(base),quality),parent["background_coefficient"][index],optimize=True),axis=-1)[None]
        role,_=model_role_evidence(bundle,parent,bg,np.asarray([index]))
        fixed=foreground_reference_log_prior(bundle,parent,np.asarray([index]))[:,None]+np.einsum("npd,d->np",design,parent["core_coefficient"][index],optimize=True)[None]
        allowed=training_run[index,bundle["run_index"]]&qualified;weight=allowed/allowed.sum()
        sd=np.asarray([np.sqrt(max(weight@(label**2)-(weight@label)**2,(period[1]-period[0])**2/(12*scale**2)))])
        eligible=training_run[index:index+1]&(number>0.)[None]
        units=((eligible.astype(float)@group_unit)>0.).sum(1)
        value=_fit_mean_role(bundle,base,role,fixed,parent["role_coefficient"][index:index+1],target,training_run[index:index+1],
            (curve_x[index:index+1],curve_y[index:index+1],curve_n[index:index+1]),
            sd,units,maxiter,device,period_reference=None if mean_only else (label,qualified),core_fit=(design,parent["core_coefficient"][index:index+1],None) if direct_mean else None,
            prior_scale_weight=weight[None])
        print("CALIBRATED_SOURCE_MEASUREMENT",index,value["success"],flush=True)
        mean=np.clip(value["group_mean"]*scale,period.min(),period.max())
        raw=parent["original_period"][index:index+1]
        run=bundle["run_index"]
        point=mean[:,run]+parent["within_scale"][index]*(raw-(raw@bundle["average"].T)[:,run])
        point=project_periods_with_mean(point,mean,bundle["membership"],period.min(),period.max())
        return {"role_coefficient":value["role_coefficient"][0],"core_coefficient":value["core_coefficient"][0] if direct_mean else parent["core_coefficient"][index],
            "mean_bias":value["mean_bias"][0],"success":np.asarray(value["success"]),
            "estimate":point[0],"group_mean":mean[0],"source_period":value["source_period"][0]*scale,
            "original_period":raw[0],"posterior":value["posterior"][0]}
    fitted=list(map(case,np.arange(len(training_run))))
    result={key:value.copy() for key,value in parent.items()}
    result.update({key:np.stack(list(map(lambda item:item[key],fitted))) for key in fitted[0]})
    result["success"]&=parent["success"]
    result.update({"calibration_prediction":curve_x,"calibration_mean":curve_y,"calibration_experiments":curve_n})
    result["source_representation"]=bundle["source_representation"]
    result["direct_source_mean"]=np.asarray(direct_mean)
    result["observed_mean_only"]=np.asarray(mean_only)
    result["mean_map_kind"]=np.asarray("identity" if identity else "monotone")
    return result


def predict_general_source_roles(bundle,model,cases=None,evidence_cache=None,include_log_probability=False):
    """Read out the shared source-role model using only measured observations."""
    from source_assignment_core import role_assignment_evidence
    cases=np.arange(len(model["core_coefficient"])) if cases is None else np.asarray(cases,dtype=int)
    period=bundle["period"];scale=float(bundle["period_scale"])
    base=bundle["source_base"];coherent=base.copy();coherent[:,:,11]=bundle["source_coherent_power"]
    quality=model_role_quality(bundle,model)
    paired=str(model.get("core_measurements","single"))=="paired"
    background=softmax(np.einsum("npd,fd->fnp",quality_design(_polynomial(base),quality),model["background_coefficient"][cases],optimize=True),axis=-1)
    cache={} if evidence_cache is None else evidence_cache
    kind=str(model.get("role_kind","gaussian"))
    key=(id(base),kind,model["background_coefficient"][cases].tobytes(),model["role_covariance"][cases].tobytes(),
        model["core_coefficient"][cases].tobytes() if kind=="density" else b"",
        foreground_reference_log_prior(bundle,model,cases).tobytes() if kind=="density" else b"",str(model.get("role_reduction","expected_bf")))
    if key not in cache:
        cache[key]=model_role_evidence(bundle,model,background,cases)[0]
    role=cache[key]
    if str(model.get("role_shape_kind","none"))=="cepstrum":
        data={"time":bundle["time"],"mask":bundle["mask"],"count":bundle["count"],"runs":bundle["run_names"],
            "run_index":bundle["run_index"],"membership":bundle["membership"],"average":bundle["average"]}
        shape=observed_cepstral_features(bundle["observed_signal"],data,period)
        role=np.concatenate([role,np.broadcast_to(shape,(len(cases),*shape.shape))],axis=-1)
    period_prior=foreground_reference_log_prior(bundle,model,cases)
    score=period_prior[:,None]+np.einsum("npd,fd->fnp",source_core_design(bundle,coherent,quality,coherent=True,paired=paired),model["core_coefficient"][cases],optimize=True)
    score+=_role_score(role,base,model["role_coefficient"][cases],quality)
    log_probability=score-logsumexp(score,axis=-1,keepdims=True)
    probability=np.exp(log_probability)
    source=probability@period
    source_mean=(probability@(period/scale))@bundle["average"].T
    mean=interpolate_mean(source_mean,model["calibration_prediction"][cases],model["calibration_mean"][cases],
        model["calibration_experiments"][cases],identity_tails=True)+model["mean_bias"][cases,None]
    mean=np.clip(mean*scale,period.min(),period.max())
    raw=softmax(period_prior[:,None]+np.einsum("npd,fd->fnp",_polynomial(bundle["identity_base_features"]),model["anchor_coefficient"][cases],optimize=True),axis=-1)@period
    point=mean[:,bundle["run_index"]]+model["within_scale"][cases,None]*(raw-(raw@bundle["average"].T)[:,bundle["run_index"]])
    point=project_periods_with_mean(point,mean,bundle["membership"],period.min(),period.max())
    result={"estimate":point,"group_mean":mean,"source_period":source,"original_period":raw,"posterior":probability}
    if include_log_probability:
        result["log_posterior"]=log_probability
    return result


def fit_general_background_references(bundle,background_views,qualified,training_run,parent_coefficient,maxiter=900,independent_unit=None,reference_count=None):
    """Fit only the measured-peer nuisance law; source parameters stay fixed."""
    bx=background_views["features"];period=bundle["period"];run=bundle["run_index"]
    conditioned=parent_coefficient.shape[-1]==208
    bx=quality_design(bx,background_views["mixing_quality"] if conditioned else None)
    origins=background_views["origin_index"]
    btarget=period[background_views["background_index"]]
    code,names=pd.factorize(background_views["view_id"],sort=True)
    member=(np.arange(len(names))[:,None]==code[None]).astype(float)
    native_view=(member@background_views["is_original"].astype(float))>0.
    shares=np.where(native_view,1. if len(names)==1 else .5,.5/max(len(names)-1,1))
    units=bundle["run"] if independent_unit is None else independent_unit
    unit_code,unit_names=pd.factorize(units,sort=True)
    unit_record=(np.arange(len(unit_names))[:,None]==unit_code[None]).astype(float)
    left,right=np.triu_indices(13)
    selected=np.r_[np.isin(np.arange(13),[4,5,6,7,8]),np.isin(left,[4,5,6,7,8])&np.isin(right,[4,5,6,7,8])]
    if conditioned:
        selected=np.tile(selected,2)
    def case(index):
        groups=training_run[index:index+1]
        allowed=groups[:,run]&qualified[None]
        btrain=groups[:,run[origins]]&(~background_views["is_original"][None]|qualified[origins][None])
        bcount=btrain@member.T
        bw=btrain*shares[code][None]/np.maximum(bcount[:,code],1.)
        bscale=btrain&background_views["is_original"][None]
        bscale=bscale/bscale.sum(1)[:,None]
        coefficient,status=_fit_score(bx,btarget,np.ones(len(btarget),dtype=bool),btrain,period,
            maxiter=maxiter,training_weight=bw,effective_sample_size=allowed.sum(1) if reference_count is None else reference_count[index:index+1],
            prior_scale_weight=bscale,initial_coefficient=parent_coefficient[index:index+1,selected],
            quality=background_views["mixing_quality"] if conditioned else None,
            interaction_units=allowed.sum(1) if reference_count is None else reference_count[index:index+1])
        output=np.zeros(parent_coefficient.shape[-1]);output[selected]=coefficient[0]
        return output,status
    values=list(map(case,np.arange(len(training_run))))
    return np.stack(list(map(lambda item:item[0],values))),list(map(lambda item:item[1],values))


def fit_general_source_roles(bundle,reference,qualified,training_run,background_reference,maxiter=900,device="cuda",train_on_original=False,independent_unit=None,source_consistent=False,background_views=None,conditioned=False,reference_source_id=None,paired_core=False,paired_roles=False,foreground_prior_kind="reference",initial_model=None,legacy_correlated_roles=False,conditioning_field="mixing_quality",view_context=None):
    """Fit one role law and one mean calibration for every acquisition geometry."""
    from source_assignment_core import role_assignment_evidence
    period=bundle["period"];scale=float(bundle["period_scale"]);average=bundle["average"];run=bundle["run_index"]
    target=np.where(qualified,reference,0.);train=training_run[:,run]&qualified[None]
    if source_consistent:
        bundle=dict(bundle,base_features=bundle["source_base"],original_unit=bundle["source_unit"],original_peer=bundle["source_peer"],
            original_rhs=bundle["source_rhs"],original_coherent_power=bundle["source_coherent_power"])
    base=bundle["base_features"];source_base=base if train_on_original else bundle["source_base"]
    quality=bundle[conditioning_field] if conditioned else None
    source_rhs=bundle["original_rhs"] if train_on_original else bundle["source_rhs"]
    units=bundle["run"] if independent_unit is None else independent_unit
    unit_code,unit_names=pd.factorize(units,sort=True)
    unit_record=(np.arange(len(unit_names))[:,None]==unit_code[None]).astype(float)
    group_unit=(bundle["membership"]@unit_record.T)>0.
    labels=np.where(np.isfinite(background_reference),background_reference,period[base[:,:,4].argmax(1)])
    label_index=abs(period[None]-labels[:,None]).argmin(1)
    group_number=bundle["membership"]@qualified.astype(float)
    group_reference=np.divide(bundle["membership"]@(target/scale),group_number,out=np.full(len(group_number),np.nan),where=group_number>0.)
    original_context={"qualified":qualified,"origin_run":bundle["run"],"independent_unit":units,
        "view_weight":np.ones(len(target)),"is_original":np.ones(len(target),dtype=bool)}
    reference_weight=source_reference_weights(bundle,original_context,training_run)
    foreground_mean=reference_weight["training_weight"]@(target/scale)
    foreground_variance=reference_weight["training_weight"]@((target/scale)**2)-foreground_mean**2
    foreground_sd=np.sqrt(np.maximum(foreground_variance*(1.+1./reference_weight["effective_training_units"]),
        (period[1]-period[0])**2/(12.*scale**2)))
    foreground_prior=-.5*(((period/scale)[None]-foreground_mean[:,None])/foreground_sd[:,None])**2
    foreground_prior-=logsumexp(foreground_prior,axis=-1,keepdims=True)
    if foreground_prior_kind=="uniform":
        foreground_prior=np.zeros_like(foreground_prior)
    identities=bundle["trace_id"] if reference_source_id is None else reference_source_id
    source_code,source_names=pd.factorize(identities,sort=True)
    source_record=(np.arange(len(source_names))[:,None]==source_code[None]).astype(float)
    reference_count=((train.astype(float)@source_record.T)>0.).sum(1)
    training_weight=None;group_weight=None
    original_group=np.ones(len(bundle["run_names"]),dtype=bool)
    if view_context is not None:
        share=np.asarray(view_context["view_weight"],dtype=float)
        weighted=train*share[None]
        source_total=weighted@source_record.T
        training_weight=np.divide(weighted,source_total[:,source_code],out=np.zeros_like(weighted),where=source_total[:,source_code]>0.)
        training_weight/=reference_count[:,None]
        group_share=average@share
        group_weight=training_run*(group_number>0.)[None]*group_share[None]
        group_weight/=group_weight.sum(1)[:,None]
        original_group=(average@np.asarray(view_context["is_original"],dtype=float))>0.
    raw_coherent=base.copy();raw_coherent[:,:,11]=bundle["original_coherent_power"]
    source_coherent=source_base.copy();source_coherent[:,:,11]=bundle["original_coherent_power"] if train_on_original else bundle["source_coherent_power"]
    raw_design=source_core_design(bundle,raw_coherent,quality,coherent=True,paired=paired_core)
    source_design=source_core_design(bundle,source_coherent,quality,coherent=True,paired=paired_core)
    core_quality=None if paired_core else quality
    def case(index):
        allowed=train[index:index+1];groups=training_run[index:index+1]
        eligible=groups&(group_number>0.)[None]
        effective_units=((eligible.astype(float)@group_unit)>0.).sum(1)
        period_prior=foreground_prior[index:index+1]
        initial={} if initial_model is None else {k:v[index:index+1] for k,v in initial_model.items()}
        count=reference_count[index:index+1]
        tw=None if training_weight is None else training_weight[index:index+1]
        gw=None if group_weight is None else group_weight[index:index+1]
        anchor,core,core_status,phase_core=_fit_core(base,target,qualified,allowed,period,maxiter,period_prior,return_phase=True,quality=core_quality,interaction_units=effective_units,reference_count=count,
            companion_base=bundle["companion_base_features"] if paired_core else None,initial_core=initial.get("core_coefficient"),training_weight=tw)
        if source_consistent or conditioned or paired_core:
            left,right=np.triu_indices(bundle["identity_base_features"].shape[-1]);phase=np.r_[np.arange(13)==12,(left==12)|(right==12)]
            anchor,anchor_status=_fit_score(_polynomial(bundle["identity_base_features"]),target,qualified,allowed,period,np.flatnonzero(~phase),parent=period_prior[:,None],ordered=True,maxiter=maxiter,effective_sample_size=count,initial_coefficient=initial.get("anchor_coefficient"),training_weight=tw)
            core_status=core_status+[anchor_status]
        bg_left,bg_right=np.triu_indices(13)
        bg_mask=np.r_[np.isin(np.arange(13),[4,5,6,7,8]),np.isin(bg_left,[4,5,6,7,8])&np.isin(bg_right,[4,5,6,7,8])]
        bg_columns=np.flatnonzero(np.tile(bg_mask,2) if quality is not None else bg_mask) if legacy_correlated_roles else None
        background,bg_status=_fit_score(quality_design(_polynomial(base),quality),labels,qualified,allowed,period,columns=bg_columns,maxiter=maxiter,quality=quality,interaction_units=effective_units,effective_sample_size=count,initial_coefficient=initial.get("background_coefficient"),training_weight=tw)
        if paired_roles:
            covariance,cov_status=_fit_grid_source_covariance(bundle,target,qualified,allowed,label_index,device,maxiter,quality,count)
        else:
            covariance,cov_status=_fit_covariance(bundle,target,qualified,allowed,label_index,device,maxiter,quality=quality,interaction_units=effective_units,reference_count=count,initial_covariance=initial.get("role_covariance"),training_weight=tw)
        raw_bg=softmax(np.einsum("npd,fd->fnp",quality_design(_polynomial(base),quality),background,optimize=True),axis=-1)
        raw_role,_=source_role_evidence(bundle,covariance,raw_bg,rhs=bundle["original_rhs"],correlated=legacy_correlated_roles,quality=quality)
        raw_fixed=period_prior[:,None]+np.einsum("npd,fd->fnp",source_core_design(bundle,base,quality,paired=paired_core),core,optimize=True)
        role,role_status=_fit_period_role(base,raw_role,raw_fixed,target/scale,qualified,allowed,period/scale,maxiter,device,
            cohort=(bundle["membership"],average,group_reference,groups),quality=quality,interaction_units=effective_units,reference_count=count,initial_coefficient=initial.get("role_coefficient"),training_weight=tw,cohort_weight=gw)
        fixed=period_prior[:,None]+np.einsum("npd,fd->fnp",raw_design,phase_core,optimize=True)
        rivals=np.stack([base[:,:,0].argmax(1),base[:,:,4].argmax(1)],axis=-1)
        increment,coherent_status=_fit_score(raw_design,target,qualified,allowed,period,parent=fixed,ordered=True,ranking=True,rivals=rivals,maxiter=maxiter,initial_coefficient=core-phase_core,quality=core_quality,interaction_units=effective_units,effective_sample_size=count,prior_multiplier=2. if paired_core else 1.,training_weight=tw)
        core=phase_core+increment
        raw_probability=softmax(period_prior[:,None]+np.einsum("npd,fd->fnp",raw_design,core,optimize=True)+_role_score(raw_role,base,role,quality),axis=-1)
        parent_mean=(raw_probability@(period/scale))@average.T
        eligible=groups&(group_number>0.)[None]&original_group[None]
        _,cal_x,cal_y,cal_n=fit_monotone_mean(parent_mean,group_reference,eligible)
        if background_views is not None:
            background,matched_bg_status=fit_general_background_references(bundle,background_views,qualified,groups,background,maxiter,independent_unit=units,reference_count=count)
            core_status=core_status+matched_bg_status
        source_bg=softmax(np.einsum("npd,fd->fnp",quality_design(_polynomial(source_base),quality),background,optimize=True),axis=-1)
        source_role,_=source_role_evidence(bundle,covariance,source_bg,rhs=source_rhs,correlated=legacy_correlated_roles,quality=quality)
        source_fixed=period_prior[:,None]+np.einsum("npd,fd->fnp",source_design,core,optimize=True)
        probability=softmax(source_fixed+_role_score(source_role,source_base,role,quality),axis=-1)
        source_period=probability@(period/scale)
        mean=interpolate_mean(source_period@average.T,cal_x,cal_y,cal_n,identity_tails=True)
        alpha,original=_within_scale(bundle,anchor,period_prior,target,qualified,groups&original_group[None],0)
        reference_deviation=np.where(qualified,target/scale-group_reference[run],0.)
        alpha=np.where(np.sum(allowed*reference_deviation[None]**2,axis=1)==0.,0.,alpha)
        mean=np.clip(mean*scale,period.min(),period.max())
        raw=original*scale
        point=mean[:,run]+alpha[:,None]*(raw-(raw@average.T)[:,run])
        point=project_periods_with_mean(point,mean,bundle["membership"],period.min(),period.max())
        success=all(map(lambda item:item["success"],core_status+[bg_status,cov_status,role_status,coherent_status]))
        print("GENERAL_SOURCE_ROLE_CASE",index,success,flush=True)
        return {"estimate":point[0],"group_mean":mean[0],"source_period":source_period[0]*scale,
            "posterior":probability[0],"original_period":raw[0],"core_coefficient":core[0],"anchor_coefficient":anchor[0],
            "background_coefficient":background[0],"role_covariance":covariance[0],"role_coefficient":role[0],
            "calibration_prediction":cal_x[0],"calibration_mean":cal_y[0],"calibration_experiments":cal_n[0],
            "mean_bias":np.asarray(0.),"within_scale":alpha[0],"success":np.asarray(success),
            "foreground_reference_mean":foreground_mean[index],"foreground_reference_sd":foreground_sd[index],
            "reference_source_count":reference_count[index],"effective_training_units":effective_units[0]}
    values=list(map(case,np.arange(len(training_run))))
    output=dict(zip(values[0],map(lambda key:np.stack(list(map(lambda value:value[key],values))),values[0])))
    output["core_measurements"]=np.asarray("paired" if paired_core else "single")
    output["role_measurements"]=np.asarray("raw_and_balanced" if paired_roles else "raw")
    output["foreground_prior_kind"]=np.asarray(foreground_prior_kind)
    return output


def fit_matched_role_control(bundle,parent,reference,qualified,training_run,maxiter=900,device="cuda",role_kind="density",role_reduction="expected_bf"):
    """Refit density roles with the same core, BG, references, views and losses."""
    model=dict(parent,role_kind=np.asarray(role_kind),role_reduction=np.asarray(role_reduction))
    period=bundle["period"];scale=float(bundle["period_scale"]);base=bundle["source_base"]
    coherent=base.copy();coherent[:,:,11]=bundle["source_coherent_power"]
    quality=model_role_quality(bundle,parent)
    paired=str(parent.get("core_measurements","single"))=="paired"
    design=source_core_design(bundle,coherent,quality,coherent=True,paired=paired)
    target=np.where(qualified,reference/scale,0.)
    number=bundle["membership"]@qualified.astype(float)
    group_reference=np.divide(bundle["membership"]@target,number,out=np.full(len(number),np.nan),where=number>0.)
    def case(index):
        groups=training_run[index:index+1];allowed=groups[:,bundle["run_index"]]&qualified[None]
        bg=softmax(np.einsum("npd,d->np",quality_design(_polynomial(base),quality),parent["background_coefficient"][index],optimize=True),axis=-1)[None]
        role,_=model_role_evidence(bundle,model,bg,np.asarray([index]))
        fixed=foreground_reference_log_prior(bundle,parent,np.asarray([index]))[:,None]+np.einsum("npd,d->np",design,parent["core_coefficient"][index],optimize=True)[None]
        n=parent["reference_source_count"][index:index+1]
        coefficient,status=_fit_period_role(base,role,fixed,target,qualified,allowed,period/scale,maxiter,device,
            cohort=(bundle["membership"],bundle["average"],group_reference,groups),quality=quality,interaction_units=n,reference_count=n)
        probability=softmax(fixed+_role_score(role,base,coefficient,quality),axis=-1)
        source_mean=(probability@(period/scale))@bundle["average"].T
        _,cx,cy,cn=fit_monotone_mean(source_mean,group_reference,groups&(number>0.)[None])
        print("MATCHED_ROLE_CASE",role_kind,role_reduction,index,status["success"],flush=True)
        return {"role_coefficient":coefficient[0],"calibration_prediction":cx[0],"calibration_mean":cy[0],
            "calibration_experiments":cn[0],"mean_bias":np.asarray(0.),"success":np.asarray(status["success"])}
    values=list(map(case,np.arange(len(training_run))))
    model.update({key:np.stack(list(map(lambda value:value[key],values))) for key in values[0]})
    model["success"]&=parent["success"]
    model["comparison_policy"]=np.asarray("matched_role_refit_shared_core_BG_source_references_and_losses")
    model.update(predict_general_source_roles(bundle,model))
    return model


def fit_cohort_moment_measurement(bundle,context,training_run,foreground_log_prior,maxiter=900,initial_coefficient=None):
    """One source score learns cell periods and finite-cohort means together.

    Original and gain-balanced measurements remain separate coordinates of the
    same score. Pooling averages their measured statistics, with no multiplication
    of dependent cell likelihoods and no replacement prior at prediction time.
    """
    period=bundle["period"];membership=bundle["membership"];average=bundle["average"]
    target=np.where(context["qualified"],context["target"],0.)
    weights=source_reference_weights(bundle,context,training_run)
    number=membership@context["qualified"].astype(float)
    cohort_target=np.divide(membership@target,number,out=np.zeros(len(number)),where=number>0.)
    original=_polynomial(bundle["identity_base_features"])
    balanced=_polynomial(bundle["measurement_base_features"])
    design=np.concatenate([original,balanced],axis=-1)
    deviations=(target-cohort_target[bundle["run_index"]])*context["qualified"]
    variance=np.divide(membership@(deviations**2),number-1.,out=np.zeros_like(number),where=number>1.)
    mean_variance=np.maximum(np.divide(variance,number,out=np.zeros_like(number),where=number>0.),(period[1]-period[0])**2/12.)
    moments={"average":average,"reference":cohort_target,"variance":mean_variance,"weight":weights["group_weight"]}
    rivals=np.stack([bundle["identity_base_features"][:,:,0].argmax(1),bundle["identity_base_features"][:,:,4].argmax(1)],axis=-1)
    left,right=np.triu_indices(13)
    phase=np.r_[np.arange(13)==12,(left==12)|(right==12)]
    columns=np.r_[np.flatnonzero(~phase),104+np.flatnonzero(~phase)]
    coefficient,status=_fit_score(design,target,context["qualified"],weights["train"],period,columns,
        parent=foreground_log_prior[:,None],ordered=True,ranking=True,rivals=rivals,maxiter=maxiter,
        training_weight=weights["training_weight"],effective_sample_size=weights["effective_sample_size"],
        initial_coefficient=initial_coefficient,cohort_moments=moments,ranking_without_prior=True)
    return coefficient,status


def cohort_moment_prediction(bundle,coefficient,foreground_log_prior,anchor_coefficient,anchor_log_prior,within_scale):
    """Use the exact trained score and prior for both readout levels."""
    original=_polynomial(bundle["identity_base_features"])
    design=np.concatenate([original,_polynomial(bundle["measurement_base_features"])],axis=-1)
    score=np.einsum("npd,fd->fnp",design,coefficient,optimize=True)
    posterior=softmax(score+foreground_log_prior[:,None],axis=-1)
    cohort_probability=np.einsum("gn,fnp->fgp",bundle["average"],posterior,optimize=True)
    mean=cohort_probability@bundle["period"]
    raw=softmax(np.einsum("npd,fd->fnp",original,anchor_coefficient,optimize=True)+anchor_log_prior[:,None],axis=-1)@bundle["period"]
    delta=raw-(raw@bundle["average"].T)[:,bundle["run_index"]]
    unbounded=mean[:,bundle["run_index"]]+within_scale[:,None]*delta
    estimate=project_periods_with_mean(unbounded,mean,bundle["membership"],bundle["period"].min(),bundle["period"].max())
    return {"estimate":estimate,"group_mean":mean,"source_period":posterior@bundle["period"],
        "posterior":posterior,"cohort_probability":cohort_probability,"original_period":raw}


def fit_bundle(bundle,reference,qualified,training_run,maxiter=900,device="cuda",background_reference=None,method_indices=None,initial_fit=None,cache_prefix=None,independent_unit=None,view_bundle=None,view_context=None,refit_source_parent=False):
    """Fit the shared-gain foreground measurement and joint cohort model."""
    assert str(bundle["source_operator"])==SOURCE_OPERATOR
    torch.set_num_threads(__import__("os").cpu_count())
    started=time.perf_counter();period=bundle["period"];ncase=len(training_run)
    indices=np.arange(len(METHODS)) if method_indices is None else np.asarray(method_indices,dtype=int)
    units=bundle["run"] if independent_unit is None else independent_unit
    target=np.where(qualified,reference,0.);train=training_run[:,bundle["run_index"]]&qualified[None]
    unit_code,unit_names=pd.factorize(units,sort=True);unit_member=(np.arange(len(unit_names))[:,None]==unit_code[None]).astype(float)
    foreground_prior,effective_units=_reference_prior(target,qualified,train,period,unit_member)
    mean=cohort_reference_prior(bundle,reference,qualified,training_run,units)
    left,right=np.triu_indices(13);phase=np.r_[np.arange(13)==12,(left==12)|(right==12)];columns=np.flatnonzero(~phase)
    raw_poly=_polynomial(bundle["identity_base_features"])
    saved={}
    if initial_fit is not None:
        assert str(initial_fit["model_kind"])==MODEL_KIND and str(initial_fit["source_operator"])==SOURCE_OPERATOR
        assert str(initial_fit["source_core_policy"])==CORE_POLICY
        assert np.array_equal(initial_fit["trace_id"],bundle["trace_id"]) and np.array_equal(initial_fit["training_run"],training_run)
        assert np.array_equal(initial_fit["qualified"],qualified) and np.allclose(initial_fit["reference"],reference,equal_nan=True)
        saved=dict(zip(initial_fit["method_indices"].tolist(),np.arange(len(initial_fit["methods"])).tolist()))
    anchor=np.zeros((ncase,104));anchor_success=np.ones(ncase,dtype=bool)
    anchor_iterations=np.zeros(ncase,dtype=int);anchor_gradient=np.zeros(ncase)
    if initial_fit is not None:
        first=0  # The original-anchor parameters are shared by every method row.
        anchor=initial_fit["anchor_coefficient"][first].copy()
        np.testing.assert_array_equal(foreground_prior,initial_fit["foreground_log_prior"][first])
        anchor_success=initial_fit["anchor_success_by_case"][first].copy()
    def raw_anchor(case):
        if initial_fit is not None and anchor_success[case]:
            return None
        coefficient,status=_fit_score(raw_poly,target,qualified,train[case:case+1],period,columns,parent=foreground_prior[case:case+1,None],ordered=True,maxiter=maxiter,
            initial_coefficient=anchor[case:case+1] if initial_fit is not None else None)
        anchor[case]=coefficient[0];anchor_success[case]=status["success"];anchor_iterations[case]=status["iterations"];anchor_gradient[case]=status["gradient"]
        return None
    list(map(raw_anchor,np.arange(ncase)))
    alpha,_=_within_scale(bundle,anchor,foreground_prior,target,qualified,training_run,0)
    alpha=np.where(mean["foreground_within_sd"]==0.,0.,alpha)
    original_context={"target":reference,"qualified":qualified,"origin_run":bundle["run"],"origin_trace_id":bundle["trace_id"],"independent_unit":units,
        "view_weight":np.ones(len(reference)),"is_original":np.ones(len(reference),dtype=bool)}
    original_weights=source_reference_weights(bundle,original_context,training_run)
    vb=bundle if view_bundle is None else view_bundle
    vc=original_context if view_context is None else view_context
    if view_bundle is None:
        view_training=training_run
    else:
        origins=pd.DataFrame({"run":vb["run"],"origin":vc["origin_run"]}).drop_duplicates().set_index("run").loc[vb["run_names"],"origin"]
        origin_index=pd.Index(bundle["run_names"]).get_indexer(origins)
        assert np.all(origin_index>=0)
        view_training=training_run[:,origin_index]
    view_weights=source_reference_weights(vb,vc,view_training)
    states={}
    def method(variant):
        if variant in states:
            return states[variant]
        prior_mean=mean["mean_log_prior"].copy();within_sd=mean["foreground_within_sd"].copy()
        value={"anchor_coefficient":anchor.copy(),"foreground_log_prior":foreground_prior.copy(),
            "measurement_coefficient":np.zeros((ncase,104)),"measurement_duration_coefficient":np.zeros((ncase,104)),
            "measurement_log_prior":foreground_prior.copy(),"mean_log_prior":prior_mean,"foreground_within_sd":within_sd,
            "within_scale":alpha.copy(),"own_emission_covariance":np.zeros((ncase,3)),"background_log_prior":np.full((ncase,len(period)),-np.log(len(period))),
            "anchor_success":anchor_success.copy(),"measurement_success":np.ones(ncase,dtype=bool),"iterations":np.zeros(ncase,dtype=int),
            "gradient":np.zeros(ncase),"solver":np.repeat("not_fitted",ncase).astype("<U64"),"message":np.repeat("fixed readout",ncase).astype("<U512")}
        saved_index=saved.get(variant)
        if saved_index is not None and not refit_source_parent:
            value.update(dict(zip(POOL_PARAMETER_KEYS,map(lambda key:initial_fit[key][saved_index].copy(),POOL_PARAMETER_KEYS))))
            value["measurement_success"]=initial_fit["measurement_success_by_case"][saved_index].copy()
            value["iterations"]=initial_fit["measurement_iterations_by_case"][saved_index].copy()
            value["gradient"]=initial_fit["measurement_gradient_by_case"][saved_index].copy()
            value["solver"]=initial_fit["measurement_solver_by_case"][saved_index].copy()
            value["message"]=initial_fit["optimizer_message_by_case"][saved_index].astype("<U512")
        if saved_index is not None and not refit_source_parent and value["measurement_success"].all():
            pass
        elif variant in [2,5,6,7,9,10]:
            parent=method(0)
            value.update(dict(zip(POOL_PARAMETER_KEYS,map(lambda key:parent[key].copy(),POOL_PARAMETER_KEYS))))
            value["measurement_success"]=parent["measurement_success"].copy()
            value["message"]=np.repeat("Fixed main-model parameters; readout or operator sensitivity",ncase).astype("<U512")
            if variant==6:
                value["mean_log_prior"][:]=-np.log(len(period))
            if variant==7:
                value["foreground_within_sd"][:]=0.
        else:
            source_bundle,source_context,weights=(bundle,original_context,original_weights) if variant==8 else (vb,vc,view_weights)
            source_target=np.where(source_context["qualified"],source_context["target"],0.)
            if variant==11:
                if "emission_gram" not in source_bundle:
                    stats=_emission_reference_statistics(bundle["pooled_unit"],{"time":bundle["time"],"mask":bundle["mask"],"count":bundle["count"],
                        "membership":bundle["membership"],"average":bundle["average"],"run_index":bundle["run_index"]},period,float(bundle["period_scale"]),reference,background_reference)
                    source_bundle=source_bundle|stats
                value["background_log_prior"]=_weighted_period_prior(source_bundle["emission_background_period"],weights["training_weight"],period,weights["effective_training_units"])
                def emission(case):
                    cov,status=_fit_own_emission(source_bundle["emission_gram"],source_bundle["emission_rhs"],source_bundle["emission_energy"],source_bundle["emission_degrees"],
                        weights["training_weight"][case:case+1],weights["effective_sample_size"][case:case+1],maxiter,device)
                    value["own_emission_covariance"][case]=cov[0];value["measurement_success"][case]=status["success"]
                    value["iterations"][case]=status["iterations"];value["gradient"][case]=status["gradient"];value["solver"][case]="Gaussian_ML"
                    value["message"][case]="Gaussian own-view covariance maximum likelihood"
                    return None
                list(map(emission,np.arange(ncase)))
            else:
                design=_measurement_design(source_bundle,variant)
                active=np.r_[columns,columns+104] if variant==12 else columns
                def measurement(case):
                    if saved_index is not None and not refit_source_parent and value["measurement_success"][case]:
                        return None
                    current=None
                    if saved_index is not None:
                        current=np.concatenate([value["measurement_coefficient"][case:case+1],value["measurement_duration_coefficient"][case:case+1]],axis=1) if variant==12 else value["measurement_coefficient"][case:case+1]
                    coefficient,status=_fit_score(design,source_target,source_context["qualified"],weights["train"][case:case+1],period,active,
                        parent=foreground_prior[case:case+1,None],ordered=True,maxiter=maxiter,training_weight=weights["training_weight"][case:case+1],
                        effective_sample_size=weights["effective_sample_size"][case:case+1],initial_coefficient=current)
                    value["measurement_coefficient"][case]=coefficient[0,:104]
                    if variant==12:
                        value["measurement_duration_coefficient"][case]=coefficient[0,104:]
                    value["measurement_success"][case]=status["success"];value["iterations"][case]=status["iterations"];value["gradient"][case]=status["gradient"]
                    value["solver"][case]="L-BFGS-B_proper90" if variant!=12 else "L-BFGS-B_proper180"
                    value["message"][case]="Proper source measurement on allowed reference views" if view_bundle is not None and variant!=8 else "Proper source measurement on original observations"
                    return None
                list(map(measurement,np.arange(ncase)))
        states[variant]=value
        return value
    values=list(map(method,indices))
    result=dict(zip(POOL_PARAMETER_KEYS,map(lambda key:np.stack(list(map(lambda value:value[key],values))),POOL_PARAMETER_KEYS)))
    anchor_flags=np.stack(list(map(lambda v:v["anchor_success"],values)));measurement_flags=np.stack(list(map(lambda v:v["measurement_success"],values)))
    case_success=anchor_flags&measurement_flags
    policy={0:"full_fit",1:"matched_raw_refit",2:"fixed_pool_removal",3:"matched_cellwise_refit",4:"matched_peer_removal_refit",5:"independent_readout",6:"uniform_mean_readout",7:"point_within_readout",8:"matched_original_only_refit",9:"source_deviation_readout",10:"unprojected_readout",11:"gaussian_alternative",12:"matched_duration_refit"}
    applicable=np.ones(len(indices),dtype=bool);applicable[indices==7]=np.any(mean["foreground_within_sd"]>0.)
    preferred=DEFAULT_VARIANT if DEFAULT_VARIANT in indices else int(indices[0])
    result.update({"schema":np.asarray(SCHEMA),"source_operator":np.asarray(SOURCE_OPERATOR),"source_core_policy":np.asarray(CORE_POLICY),"model_kind":np.asarray(MODEL_KIND),
        "preferred_variant":np.asarray(preferred),"preferred_method":np.asarray(METHODS[preferred]),
        "methods":METHODS[indices],"method_indices":indices,"period":period,"period_scale":bundle["period_scale"],"trace_id":bundle["trace_id"],"run_names":bundle["run_names"],
        "training_run":training_run,"training_mask":train,"qualified":qualified,"reference":reference,"effective_training_units":effective_units,
        "measurement_original_qualified":original_weights["effective_sample_size"],"optimizer_success_by_method":case_success.all(1),"optimizer_success_by_case":case_success,
        "anchor_success_by_case":anchor_flags,"measurement_success_by_case":measurement_flags,
        "optimizer_message_by_case":np.stack(list(map(lambda v:v["message"],values))),
        "measurement_iterations_by_case":np.stack(list(map(lambda v:v["iterations"],values))),"measurement_gradient_by_case":np.stack(list(map(lambda v:v["gradient"],values))),
        "measurement_solver_by_case":np.stack(list(map(lambda v:v["solver"],values))),
        "iterations_by_method":np.asarray(list(map(lambda v:int(v["iterations"].sum()),values))),"gradient_by_method":np.asarray(list(map(lambda v:float(v["gradient"].max()),values))),
        "method_applicable":applicable,"parameter_policy":np.asarray(list(map(policy.__getitem__,indices.tolist()))),"elapsed_s":np.asarray(time.perf_counter()-started)})
    predictions=prediction_matrix(bundle,result,np.arange(ncase),include_probabilities=True)
    result.update(dict(zip(["estimate","source_period","original_period","group_mean","population_location","mean_probability"],
        map(predictions.__getitem__,["estimate","source_period","original_period","group_mean","population_location","mean_probability"]))))
    preferred_index=int(np.flatnonzero(indices==preferred).item())
    result["source_probability"]=predictions["source_probability"][preferred_index]
    result["source_probability_method"]=np.asarray(result["methods"][preferred_index])
    return result


def _predict_variant(bundle,model,index,cases,evidence_cache=None):
    assert str(model["model_kind"])==MODEL_KIND and str(model["source_operator"])==SOURCE_OPERATOR
    assert str(model["source_core_policy"])==CORE_POLICY
    variant=int(model["method_indices"][index]);period=bundle["period"];scale=float(bundle["period_scale"]);axis=period/scale
    raw_poly=_polynomial(bundle["identity_base_features"])
    raw_score=model["foreground_log_prior"][index,cases,None]+np.einsum("npd,fd->fnp",raw_poly,model["anchor_coefficient"][index,cases],optimize=True)
    raw_probability=softmax(raw_score,axis=-1);original=raw_probability@period
    if variant==11:
        cov=model["own_emission_covariance"][index,cases];both=np.repeat(cov[:,None],2,axis=1)
        background=np.broadcast_to(np.exp(model["background_log_prior"][index,cases,None]),(len(cases),len(original[0]),len(period)))
        likelihood=_role_features(bundle,both,background,0,field="pooled_rhs")[:,:,:,0]
    else:
        design=_measurement_design(bundle,variant)
        coefficient=np.concatenate([model["measurement_coefficient"][index,cases],model["measurement_duration_coefficient"][index,cases]],axis=1) if variant==12 else model["measurement_coefficient"][index,cases]
        likelihood=np.einsum("npd,fd->fnp",design,coefficient,optimize=True)
    prior=model["mean_log_prior"][index,cases];sd=model["foreground_within_sd"][index,cases]
    difference=axis[:,None]-axis[None];inv=np.divide(1.,sd**2,out=np.zeros_like(sd),where=sd>0.)
    transition=-.5*difference[None]**2*inv[:,None,None];transition-=logsumexp(transition,axis=1,keepdims=True)
    point=np.full((len(axis),len(axis)),-np.inf);np.fill_diagonal(point,0.);transition=np.where((sd==0.)[:,None,None],point[None],transition)
    cell_prior=logsumexp(transition+prior[:,None],axis=-1)
    if variant==5:
        posterior=softmax(likelihood+cell_prior[:,None],axis=-1);source=posterior@period;mean=source@bundle["average"].T
        population=np.full_like(mean,np.nan);mean_probability=np.full((*mean.shape,len(period)),np.nan)
    else:
        joint=joint_cohort_period_posterior(likelihood,axis,bundle["run_index"],prior,sd)
        posterior=joint["period_probability"];source=joint["period_mean"]*scale;mean=joint["finite_cohort_mean"]*scale
        population=joint["population_location"]*scale;mean_probability=joint["mean_probability"]
    delta=original-(original@bundle["average"].T)[:,bundle["run_index"]]
    estimate=mean[:,bundle["run_index"]]+model["within_scale"][index,cases,None]*delta
    if variant==9:
        estimate=source.copy()
    bounded=estimate if variant==10 else project_periods_with_mean(estimate,mean,bundle["membership"],period.min(),period.max())
    return {"estimate":bounded,"unbounded_period":estimate,"source_period":source,"original_period":original,"group_mean":mean,
        "population_location":population,"mean_probability":mean_probability,"posterior":posterior,
        "measurement_log_likelihood":likelihood,"raw_anchor_probability":raw_probability,"cell_log_prior":cell_prior}


def prediction_matrix(bundle,model,cases=None,include_probabilities=False):
    selected=np.asarray([0] if cases is None else cases,dtype=int)
    values=list(map(lambda index:_predict_variant(bundle,model,index,selected),np.arange(len(model["methods"]))))
    keys=["estimate","source_period","original_period","group_mean","population_location","mean_probability"]
    result=dict(zip(keys,map(lambda key:np.stack(list(map(lambda value:value[key],values))),keys)))
    if include_probabilities:
        result["source_probability"]=np.stack(list(map(lambda value:value["posterior"],values)))
    return result|{"cases":selected,"methods":model["methods"]}


def prediction_diagnostics(bundle,model,case=0,variant=None):
    variant=int(model.get("preferred_variant",DEFAULT_VARIANT)) if variant is None else int(variant)
    index=int(np.flatnonzero(model["method_indices"]==variant).item());value=_predict_variant(bundle,model,index,np.asarray([case]));likelihood=value["measurement_log_likelihood"][0]
    return {"period":bundle["period"],"period_scale":bundle["period_scale"],"raw_anchor_probability":value["raw_anchor_probability"][0],
        "measurement_log_likelihood":likelihood,"measurement_likelihood_probability":softmax(likelihood,axis=-1),
        "measurement_training_probability":softmax(likelihood+model["measurement_log_prior"][index,case][None],axis=-1),
        "period_probability":value["posterior"][0],"joint_period_probability":value["posterior"][0],
        "reference_period_probability":np.exp(value["cell_log_prior"][0]),"mean_probability":value["mean_probability"][0],
        "source_period":value["source_period"][0],"finite_cohort_mean":value["group_mean"][0],"population_location":value["population_location"][0],
        "foreground_within_sd":model["foreground_within_sd"][index,case],
        "foreground_within_sd_period":model["foreground_within_sd"][index,case]*float(bundle["period_scale"]),
        "pool_period_pairs":bundle["pool_period_pairs"],"pool_gains":bundle["pool_gains"],"raw_noise_tau":bundle["raw_noise_tau"]}


def predict_bundle(bundle,model,case=0,variant=None):
    variant=int(model.get("preferred_variant",DEFAULT_VARIANT)) if variant is None else int(variant)
    index=int(np.flatnonzero(model["method_indices"]==variant).item());value=_predict_variant(bundle,model,index,np.asarray([case]));run=bundle["run_index"];mean=value["group_mean"][0,run]
    success=bool(model["optimizer_success_by_case"][index,case]);feature_success=bundle["pool_period_fit_success"][run]
    frame=pd.DataFrame({"trace_id":bundle["trace_id"],"run":bundle["run"],"condition":bundle["condition"],"estimate_period":value["estimate"][0],
        "experiment_mean":mean,"individual_deviation":value["estimate"][0]-mean,"source_period":value["source_period"][0],
        "source_mean":mean,"population_location":value["population_location"][0,run],"mean_link_shift":0.,
        "support_shift":value["estimate"][0]-value["unbounded_period"][0],
        "original_period":value["original_period"][0],"source_score_entropy":-np.sum(value["posterior"][0]*np.log(np.maximum(value["posterior"][0],1e-300)),axis=1),
        "method":str(METHODS[variant]),"model_schema":SCHEMA,"training_case":case,
        "fit_status":np.where(feature_success&success,"completed","not_converged"),"applicable":bool(model["method_applicable"][index])})
    return frame,value["posterior"][0]


def retired_fit_bundle(bundle,reference,qualified,training_run,maxiter=900,device="cuda",background_reference=None,method_indices=None,initial_fit=None,cache_prefix=None,independent_unit=None,view_bundle=None,view_context=None,refit_source_parent=False):
    """Native reference parent, waveform parameter prior, then constrained adaptation."""
    started=time.perf_counter()
    assert str(bundle["source_operator"])==SOURCE_OPERATOR
    if view_bundle is not None:
        assert str(view_bundle["source_operator"])==SOURCE_OPERATOR
    torch.set_num_threads(__import__("os").cpu_count())
    background=np.full(len(reference),np.nan) if background_reference is None else background_reference
    units=bundle["run"] if independent_unit is None else independent_unit
    indices=np.arange(len(METHODS)) if method_indices is None else np.asarray(method_indices,dtype=int)
    raw_cache={};background_cache={};states={}
    keys=["core_coefficient","parent_core_coefficient","identity_core_coefficient","anchor_coefficient","background_coefficient","role_covariance","role_coefficient",
        "foreground_log_prior","background_log_prior","calibration_prediction","calibration_mean","calibration_experiments",
        "raw_role_coefficient","prior_role_coefficient","mean_bias","prior_mean_bias","within_scale"]
    saved_methods={}
    parent_changed=False
    if initial_fit is not None:
        assert str(initial_fit["model_kind"])=="coherent_reference_parent_source_prior"
        assert str(initial_fit.get("source_operator",""))==SOURCE_OPERATOR,"Source coordinates changed; refit the reference parent"
        if not refit_source_parent:
            assert str(initial_fit.get("source_core_policy",""))==CORE_POLICY,"Core policy changed; refit the source parent"
        assert np.array_equal(initial_fit["trace_id"],bundle["trace_id"]) and np.array_equal(initial_fit["training_run"],training_run)
        assert np.array_equal(initial_fit["qualified"],qualified) and np.allclose(initial_fit["reference"],reference,equal_nan=True)
        previous=int(np.flatnonzero(initial_fit["method_indices"]==0).item())
        if "optimizer_success_by_case" not in initial_fit:
            assert initial_fit["optimizer_success_by_method"].all(),"Declare old-schema failed stages with --resume-failure before continuation"
            initial_fit=dict(initial_fit)
            initial_fit["optimizer_success_by_case"]=np.ones(initial_fit["core_coefficient"].shape[:2],dtype=bool)
            initial_fit["reference_success_by_method_case"]=initial_fit["optimizer_success_by_case"].copy()
            initial_fit["view_success_by_method_case"]=initial_fit["optimizer_success_by_case"].copy()
        assert initial_fit["reference_success_by_method_case"].all(),"A failed reference stage needs its own refit"
        if not refit_source_parent:
            saved_methods=dict(zip(initial_fit["method_indices"].tolist(),np.arange(len(initial_fit["methods"])).tolist()))
            parent_changed=not bool(initial_fit["optimizer_success_by_case"][previous].all())
        reused={"success":True,"iterations":0,"gradient":0.}
        raw_cache[(False,False)]=(initial_fit["anchor_coefficient"][previous],initial_fit["identity_core_coefficient"][previous],[reused,reused,reused])
        background_cache[((False,False),False)]=(initial_fit["background_coefficient"][previous],initial_fit["role_covariance"][previous],initial_fit["background_log_prior"][previous],reused,reused)
        print("REFERENCE_BACKBONE_REUSED",len(training_run),len(reference),flush=True)
        if refit_source_parent:
            print("SOURCE_PARENT_CHAIN_REFIT",CORE_POLICY,flush=True)

    def method(index):
        if index in states:
            return states[index]
        if index in saved_methods and (not parent_changed or index in [0,7,10,12,13,14]):
            saved_index=saved_methods[index]
            value=dict(zip(keys,map(lambda key:initial_fit[key][saved_index].copy(),keys)))
            value.update(dict(zip(["estimate","source_period","original_period","group_mean"],
                map(lambda key:initial_fit[key][saved_index].copy(),["estimate","source_period","original_period","group_mean"]))))
            value.update({"geometry_center":initial_fit["geometry_center"],"geometry_scale":initial_fit["geometry_scale"],
                "effective_training_units":initial_fit["effective_training_units"],
                "case_success":initial_fit["optimizer_success_by_case"][saved_index].copy(),
                "reference_success_by_case":initial_fit["reference_success_by_method_case"][saved_index].copy(),
                "view_success_by_case":initial_fit["view_success_by_method_case"][saved_index].copy(),
                "case_message":initial_fit["optimizer_message_by_case"][saved_index].astype("<U512") if "optimizer_message_by_case" in initial_fit else np.repeat("Verified original run; unchanged",len(training_run)).astype("<U512"),
                "native_iterations_by_case":initial_fit["native_iterations_by_case"][saved_index].copy() if "native_iterations_by_case" in initial_fit else np.full(len(training_run),-1,dtype=int),
                "native_gradient_by_case":initial_fit["native_gradient_by_case"][saved_index].copy() if "native_gradient_by_case" in initial_fit else np.full(len(training_run),np.nan),
                "native_solver_by_case":initial_fit["native_solver_by_case"][saved_index].astype("<U64") if "native_solver_by_case" in initial_fit else np.repeat("verified_original_run",len(training_run)).astype("<U64"),
                "success":bool(initial_fit["optimizer_success_by_case"][saved_index].all()),
                "iterations":int(initial_fit["iterations_by_method"][saved_index]),"gradient":float(initial_fit["gradient_by_method"][saved_index]),
                "posterior":initial_fit["source_probability"].copy() if str(initial_fit["source_probability_method"])==str(METHODS[index]) else None})
            if not value["success"]:
                value=_continue_pipeline(bundle,value,reference,qualified,training_run,index,maxiter,device,view_bundle,view_context)
            print("SOURCE_STAGE_REUSED",str(METHODS[index]),value["success"],flush=True)
        elif index in [0,7,10,13,14]:
            print("SOURCE_PIPELINE",str(METHODS[index]),flush=True)
            value=_fit_pipeline(bundle,reference,qualified,training_run,background,units,index,maxiter,device,view_bundle,view_context,raw_cache,background_cache)
        else:
            parent=method(0)
            value=dict(parent)
            if index==4:
                value["role_coefficient"]=parent["prior_role_coefficient"]
                value["mean_bias"]=parent["prior_mean_bias"]
                value["case_success"]=parent["reference_success_by_case"]&parent["view_success_by_case"]
                value["success"]=bool(value["case_success"].all())
                value["case_message"]=np.repeat("Native mean adaptation omitted",len(training_run))
                value["native_iterations_by_case"]=np.zeros(len(training_run),dtype=int)
                value["native_gradient_by_case"]=np.zeros(len(training_run))
                value["native_solver_by_case"]=np.repeat("not_fitted",len(training_run))
            if index==11:
                scale=float(bundle["period_scale"])
                label=np.where(qualified,reference/scale,0.)
                group_ref=bundle["membership"]@label/np.maximum(bundle["membership"]@qualified,1.)
                groups=training_run&(bundle["membership"]@qualified>0)[None]
                source_mean=(parent["source_period"]/scale)@bundle["average"].T
                _,x,y,count=fit_monotone_mean(source_mean,group_ref,groups)
                mean=_curve_numpy(source_mean,x,y,count)
                weight=groups/groups.sum(1)[:,None]
                value["calibration_prediction"]=x;value["calibration_mean"]=y;value["calibration_experiments"]=count
                value["mean_bias"]=((group_ref[None]-mean)*weight).sum(1)
            if index==12:
                value=_refine_without_views(bundle,parent,reference,qualified,training_run,maxiter,device)
        states[index]=value
        return value

    values=list(map(method,indices))
    result=dict(zip(keys,map(lambda key:np.stack(list(map(lambda value:value[key],values))),keys)))
    first=values[0]
    result.update({"schema":np.asarray(SCHEMA),"source_operator":np.asarray(SOURCE_OPERATOR),"source_core_policy":np.asarray(CORE_POLICY),"model_kind":np.asarray("coherent_reference_parent_source_prior"),"methods":METHODS[indices],"method_indices":indices,
        "period":bundle["period"],"period_scale":bundle["period_scale"],"geometry_center":first["geometry_center"],"geometry_scale":first["geometry_scale"],
        "trace_id":bundle["trace_id"],"run_names":bundle["run_names"],"training_run":training_run,
        "training_mask":training_run[:,bundle["run_index"]]&qualified[None],"qualified":qualified,"reference":reference,
        "effective_training_units":first["effective_training_units"],"source_probability":first["posterior"],"source_probability_method":np.asarray(METHODS[indices[0]]),
        "optimizer_success_by_method":np.asarray(list(map(lambda value:value["success"],values))),
        "optimizer_success_by_case":np.stack(list(map(lambda value:value["case_success"],values))),
        "reference_success_by_method_case":np.stack(list(map(lambda value:value["reference_success_by_case"],values))),
        "view_success_by_method_case":np.stack(list(map(lambda value:value["view_success_by_case"],values))),
        "optimizer_message_by_case":np.stack(list(map(lambda value:value["case_message"],values))),
        "native_iterations_by_case":np.stack(list(map(lambda value:value["native_iterations_by_case"],values))),
        "native_gradient_by_case":np.stack(list(map(lambda value:value["native_gradient_by_case"],values))),
        "native_solver_by_case":np.stack(list(map(lambda value:value["native_solver_by_case"],values))),
        "iterations_by_method":np.asarray(list(map(lambda value:value["iterations"],values))),"gradient_by_method":np.asarray(list(map(lambda value:value["gradient"],values))),
        "method_applicable":((indices!=7)|bool(np.any(np.isfinite(background))))&((indices!=12)|(view_bundle is not None)),
        "parameter_policy":np.asarray(list(map(lambda index:"matched_refit" if index==10 else "training_control" if index in [7,12,13,14] else "fixed_parameter_removal" if index not in [0,11] else "readout_control" if index==11 else "full_fit",indices))),
        "elapsed_s":np.asarray(time.perf_counter()-started)})
    fresh=prediction_matrix(bundle,result,np.arange(len(training_run)))
    result.update(dict(zip(["estimate","source_period","original_period","group_mean"],map(fresh.__getitem__,["estimate","source_period","original_period","group_mean"]))))
    if result["source_probability"] is None:
        result["source_probability"]=_predict_variant(bundle,result,0,np.arange(len(training_run)))["posterior"]
    return result


def retired_predict_variant(bundle,model,index,cases,evidence_cache=None):
    assert str(model.get("source_operator",""))==SOURCE_OPERATOR
    assert str(model.get("source_core_policy",""))==CORE_POLICY
    variant=int(model["method_indices"][index])
    original_base=_base_variant(bundle,variant)
    base=_source_base_variant(bundle,variant)
    poly=_source_design(bundle,_core_base_variant(bundle,variant),float(model["geometry_center"]),float(model["geometry_scale"]),variant)
    original_poly=_polynomial(original_base)
    identity_poly=_polynomial(_identity_base_variant(bundle,variant))
    core=model["core_coefficient"][index,cases]
    cache={} if evidence_cache is None else evidence_cache
    background_key=("background",variant==2,variant in [1,8],cases.tobytes(),model["background_log_prior"][index,cases].tobytes(),model["background_coefficient"][index,cases].tobytes())
    if background_key not in cache:
        cache[background_key]=softmax(model["background_log_prior"][index,cases,None]+np.einsum("npd,fd->fnp",_background_features(base),model["background_coefficient"][index,cases],optimize=True),axis=-1)
    background=cache[background_key]
    if variant==10:
        role=np.stack([np.log(np.maximum(softmax(model["foreground_log_prior"][index,cases,None]+np.einsum("npd,fd->fnp",original_poly,model["identity_core_coefficient"][index,cases],optimize=True),axis=-1),1e-300)),np.log(np.maximum(background,1e-300))],axis=-1)
    elif variant==3:
        role=np.zeros((len(cases),len(bundle["trace_id"]),len(bundle["period"]),2))
    else:
        field={1:"unadjusted_rhs",8:"original_rhs",9:"independent_rhs"}.get(variant,"source_rhs")
        role_key=("role",background_key,field,model["role_covariance"][index,cases].tobytes())
        if role_key not in cache:
            cache[role_key]=_role_features(bundle,model["role_covariance"][index,cases],background,variant)
        role=cache[role_key]
    score=model["foreground_log_prior"][index,cases,None]+np.einsum("npd,fd->fnp",poly,core,optimize=True)+_role_score(role,base,model["role_coefficient"][index,cases])
    probability=softmax(score,axis=-1)
    source=probability@bundle["period"]
    original=softmax(model["foreground_log_prior"][index,cases,None]+np.einsum("npd,fd->fnp",identity_poly,model["anchor_coefficient"][index,cases],optimize=True),axis=-1)@bundle["period"]
    source_mean=source@bundle["average"].T
    linked=_curve_numpy(source_mean/float(bundle["period_scale"]),model["calibration_prediction"][index,cases],model["calibration_mean"][index,cases],model["calibration_experiments"][index,cases])
    unbounded_mean=linked*float(bundle["period_scale"])+float(bundle["period_scale"])*model["mean_bias"][index,cases,None]
    mean=np.clip(unbounded_mean,bundle["period"].min(),bundle["period"].max())
    anchor=source if variant==5 else original
    deviation=anchor-(anchor@bundle["average"].T)[:,bundle["run_index"]]
    estimate=mean[:,bundle["run_index"]]+model["within_scale"][index,cases,None]*deviation
    bounded=project_periods_with_mean(estimate,mean,bundle["membership"],bundle["period"].min(),bundle["period"].max())
    return {"estimate":bounded,"source_period":source,"original_period":original,"group_mean":mean,
        "source_mean":source_mean,"unbounded_mean":unbounded_mean,"posterior":probability,"role_evidence":role,"background_probability":background}


def retired_prediction_matrix(bundle,model,cases=None):
    selected=np.asarray([0] if cases is None else cases,dtype=int)
    evidence_cache={}
    values=list(map(lambda index:_predict_variant(bundle,model,index,selected,evidence_cache),np.arange(len(model["methods"]))))
    keys=["estimate","source_period","original_period","group_mean"]
    result=dict(zip(keys,map(lambda key:np.stack(list(map(lambda value:value[key],values))),keys)))
    return result|{"cases":selected,"methods":model["methods"]}


def retired_predict_bundle(bundle,model,case=0,variant=0):
    index=int(np.flatnonzero(model["method_indices"]==variant).item())
    value=_predict_variant(bundle,model,index,np.asarray([case]))
    run=bundle["run_index"]
    mean=value["group_mean"][0,run]
    source_mean=value["source_mean"][0,run]
    unbounded=value["unbounded_mean"][0,run]
    frame=pd.DataFrame({"trace_id":bundle["trace_id"],"run":bundle["run"],"condition":bundle["condition"],
        "estimate_period":value["estimate"][0],"experiment_mean":mean,"individual_deviation":value["estimate"][0]-mean,
        "source_period":value["source_period"][0],"source_mean":source_mean,"mean_link_shift":unbounded-source_mean,
        "support_shift":mean-unbounded,"original_period":value["original_period"][0],
        "source_score_entropy":-np.sum(value["posterior"][0]*np.log(np.maximum(value["posterior"][0],1e-300)),axis=1),
        "method":str(METHODS[variant]),"model_schema":SCHEMA,"training_case":case,
        "fit_status":("completed" if bool(model["optimizer_success_by_case"][index,case]) else "not_converged") if "optimizer_success_by_case" in model else "not_recorded",
        "applicable":bool(model["method_applicable"][index])})
    return frame,value["posterior"][0]


def predict_file(args,saved):
    unit=args.time_unit or str(saved.get("time_unit","h"))
    if str(saved.get("readout_kind",""))=="information_conditioned_reference_W2":
        seconds={"h":3600.,"min":60.,"s":1.,"ms":.001}
        factor=seconds[str(saved["time_unit"])]/seconds[unit]
        expected=np.asarray([saved["period"].min(),saved["period"].max()])*factor
        bounds=[args.period_min if args.period_min is not None else expected[0],
            args.period_max if args.period_max is not None else expected[1]]
        np.testing.assert_allclose(bounds,expected,rtol=1e-12,atol=1e-12)
        assert args.variant in {None,0}
        case_id=0 if args.case<0 else args.case
        case=int(np.flatnonzero(saved["case_index"]==case_id)[0])
        frame=pd.read_parquet(args.input);frame=frame.assign(x=frame.x/factor)
        fresh=predict_reference_calibration_mixture(frame,saved,cases=[case],device=getattr(args,"device","cpu"),include_diagnostics=True)
        data=observations(frame);metadata=data["metadata"]
        bundle={"trace_id":metadata.index.to_numpy(dtype=str),"period":saved["period"]*factor}
        probability=fresh["posterior"][0];run=data["run_index"];mean=fresh["group_mean"][0,run]*factor
        original=(fresh["raw_individual_probability"][0]@saved["period"])*factor
        raw_mean=((fresh["source_assignment_probability"][0]@saved["period"])@data["average"].T)*factor
        prediction=pd.DataFrame({"trace_id":bundle["trace_id"],"run":metadata.run.to_numpy(dtype=str),"condition":metadata.condition.to_numpy(dtype=str),
            "estimate_period":fresh["estimate"][0]*factor,"experiment_mean":mean,"individual_deviation":fresh["deviation"][0]*factor,
            "source_period":fresh["source_period"][0]*factor,"source_mean":mean,"original_period":original,
            "mean_link_shift":mean-raw_mean[run],"support_shift":np.nan,
            "source_score_entropy":-np.sum(probability*np.log(np.maximum(probability,np.finfo(float).tiny)),axis=-1),
            "source_information":fresh["source_information"],"method":"unified","model_schema":str(saved["schema"]),
            "readout_kind":str(saved["readout_kind"]),"training_case":case_id,
            "fit_status":"completed" if bool(saved["success"][case]) else "not_converged","applicable":True})
    elif str(saved.get("readout_kind",""))=="reference_phase_temporal_bounded_W2":
        seconds={"h":3600.,"min":60.,"s":1.,"ms":.001}
        factor=seconds[str(saved["time_unit"])]/seconds[unit]
        case_id=0 if args.case<0 else args.case
        case=int(np.flatnonzero(saved["case_index"]==case_id)[0])
        bounds=np.asarray([saved["period"].min(),saved["period"].max()])*factor
        lower=float(args.period_min if args.period_min is not None else bounds[0])
        upper=float(args.period_max if args.period_max is not None else bounds[1])
        np.testing.assert_allclose([lower,upper],bounds,rtol=1e-12,atol=1e-12)
        assert args.variant in {None,0}
        options=json.loads(str(saved["representation_options_json"]))
        # Evaluate learned temporal coordinates in their fitted physical unit.
        frame=pd.read_parquet(args.input)
        frame=frame.assign(x=frame.x/factor)
        bundle=reference_source_observations(frame,saved["period"],options,fundamental_support=bool(saved["fundamental_support"]),device=getattr(args,"device","cpu"))
        fresh=predict_reference_phase_source(frame,saved,source_bundle=bundle,cases=[case],device=getattr(args,"device","cpu"),include_diagnostics=True)
        probability=fresh["posterior"][0];run=bundle["run_index"];mean=fresh["group_mean"][0,run]*factor
        original=(fresh["raw_individual_probability"][0]@bundle["period"])*factor
        source_mean=((fresh["source_assignment_probability"][0]@bundle["period"])@bundle["average"].T)*factor
        prediction=pd.DataFrame({"trace_id":bundle["trace_id"],"run":bundle["run"],"condition":bundle["condition"],
            "estimate_period":fresh["estimate"][0]*factor,"experiment_mean":mean,"individual_deviation":fresh["deviation"][0]*factor,
            "source_period":fresh["source_period"][0]*factor,"source_mean":mean,"original_period":original,
            "mean_link_shift":mean-source_mean[run],"support_shift":np.nan,
            "source_score_entropy":-np.sum(probability*np.log(np.maximum(probability,np.finfo(float).tiny)),axis=1),
            "method":"unified","model_schema":str(saved["schema"]),"readout_kind":str(saved["readout_kind"]),
            "training_case":case_id,"fit_status":"completed" if bool(saved["success"][case]) else "not_converged","applicable":True})
        bundle=dict(bundle,period=bundle["period"]*factor)
    elif str(saved.get("readout_kind",""))=="reference_source_bounded_W2":
        seconds={"h":3600.,"min":60.,"s":1.,"ms":.001}
        factor=seconds[str(saved["time_unit"])]/seconds[unit]
        expected=np.asarray([saved["period"].min(),saved["period"].max()])*factor
        lower=float(args.period_min if args.period_min is not None else expected[0])
        upper=float(args.period_max if args.period_max is not None else expected[1])
        np.testing.assert_allclose([lower,upper],expected,rtol=1e-12,atol=1e-12)
        case_id=0 if args.case<0 else args.case
        case=int(np.flatnonzero(saved.get("case_index",np.arange(len(saved["core_coefficient"])))==case_id)[0])
        assert args.variant in {None,0}
        options=json.loads(str(saved["representation_options_json"]))
        options["common_baseline_period"]*=factor
        frame=pd.read_parquet(args.input)
        bundle=reference_source_observations(frame,saved["period"]*factor,options,
            fundamental_support=bool(saved["fundamental_support"]),device=getattr(args,"device","cpu"))
        fresh=predict_reference_source_w2(bundle,saved,cases=[case],device=getattr(args,"device","cpu"))
        probability=fresh["posterior"][0];run=bundle["run_index"];mean=fresh["group_mean"][0,run]
        raw_score=np.einsum("npd,d->np",_polynomial(bundle["identity_base_features"]),saved["anchor_coefficient"][case],optimize=True)
        original=softmax(raw_score,axis=-1)@bundle["period"]
        prediction=pd.DataFrame({"trace_id":bundle["trace_id"],"run":bundle["run"],"condition":bundle["condition"],
            "estimate_period":fresh["estimate"][0],"experiment_mean":mean,"individual_deviation":fresh["deviation"][0],
            "source_period":fresh["source_period"][0],"source_mean":mean,"original_period":original,
            "mean_link_shift":mean-fresh["cohort_source_mean"][0,run],"support_shift":np.nan,
            "source_score_entropy":-np.sum(probability*np.log(np.maximum(probability,np.finfo(float).tiny)),axis=1),
            "method":"unified","model_schema":str(saved["schema"]),"readout_kind":str(saved["readout_kind"]),
            "training_case":case_id,"fit_status":"completed" if bool(saved["success"][case]) else "not_converged","applicable":True})
    elif str(saved.get("readout_kind",""))=="conditional_latent_representation_density":
        seconds={"h":3600.,"min":60.,"s":1.,"ms":.001}
        factor=seconds[str(saved["time_unit"])]/seconds[unit]
        bounds=np.asarray([saved["period"].min(),saved["period"].max()])*factor
        lower=float(args.period_min if args.period_min is not None else bounds[0])
        upper=float(args.period_max if args.period_max is not None else bounds[1])
        np.testing.assert_allclose([lower,upper],bounds,rtol=1e-12,atol=1e-12,
            err_msg="Prediction must retain the fitted physical period grid; equivalent time-unit conversion is supported")
        case=max(args.case,0)
        assert args.variant in {None,0},"Source-role removals use their fitted model files"
        frame=pd.read_parquet(args.input)
        raw=posterior_clock_role_features(frame,period_min=lower,period_max=upper,
            candidates=len(saved["period"]),period_scale=float(saved["period_scale"])*factor,source_balancing=False)
        balanced=posterior_clock_role_features(frame,bundle=raw,source_balancing=True)
        fresh=predict_latent_representation_density(raw,balanced,saved,cases=[case])
        bundle=raw;probability=fresh["posterior"][0];run=bundle["run_index"];mean=fresh["group_mean"][0,run]
        prediction=pd.DataFrame({"trace_id":bundle["trace_id"],"run":bundle["run"],"condition":bundle["condition"],
            "estimate_period":fresh["estimate"][0],"experiment_mean":mean,"individual_deviation":fresh["deviation"][0],
            "source_period":fresh["source_period"][0],"source_mean":mean,
            "mean_link_shift":np.nan,"support_shift":np.nan,"original_period":np.nan,
            "measured_periodic_fraction":bundle["mixing_quality"],"balanced_probability":fresh["balanced_probability"][0],
            "source_score_entropy":-np.sum(probability*np.log(np.maximum(probability,np.finfo(float).tiny)),axis=1),
            "method":"target_source_period","model_schema":str(saved["schema"]),"training_case":case,
            "fit_status":"completed" if bool(saved["success"][case]) else "not_converged","applicable":True})
    elif str(saved.get("schema",""))=="source_assignment_unified_0.3.1":
        seconds={"h":3600.,"min":60.,"s":1.,"ms":.001}
        factor=seconds[str(saved["time_unit"])]/seconds[unit]
        lower=float(args.period_min if args.period_min is not None else saved["period"].min()*factor)
        upper=float(args.period_max if args.period_max is not None else saved["period"].max()*factor)
        case=max(args.case,0)
        assert args.variant in {None,0},"Source-role removals use their fitted model files"
        bundle=posterior_clock_role_features(pd.read_parquet(args.input),period_min=lower,period_max=upper,
            candidates=len(saved["period"]),period_scale=float(saved["period_scale"])*factor,
            source_balancing=str(saved["source_representation"])!="observed_physical_peer",paired_core=str(saved.get("core_measurements","single"))=="paired")
        fresh=predict_general_source_roles(bundle,saved,cases=[case])
        probability=fresh["posterior"][0]
        source_mean=fresh["source_period"]@bundle["average"].T
        unbounded=(interpolate_mean(source_mean/float(bundle["period_scale"]),saved["calibration_prediction"][[case]],
            saved["calibration_mean"][[case]],saved["calibration_experiments"][[case]],identity_tails=True)+saved["mean_bias"][[case],None])*float(bundle["period_scale"])
        run=bundle["run_index"];mean=fresh["group_mean"][0,run]
        prediction=pd.DataFrame({"trace_id":bundle["trace_id"],"run":bundle["run"],"condition":bundle["condition"],
            "estimate_period":fresh["estimate"][0],"experiment_mean":mean,"individual_deviation":fresh["estimate"][0]-mean,
            "source_period":fresh["source_period"][0],"source_mean":source_mean[0,run],
            "mean_link_shift":unbounded[0,run]-source_mean[0,run],"support_shift":mean-unbounded[0,run],
            "original_period":fresh["original_period"][0],"measured_periodic_fraction":bundle["mixing_quality"],
            "source_score_entropy":-np.sum(probability*np.log(np.maximum(probability,np.finfo(float).tiny)),axis=1),
            "method":"target_source_period","model_schema":str(saved["schema"]),"training_case":case,
            "fit_status":"completed" if bool(saved["success"][case]) else "not_converged","applicable":True})
    else:
        lower=float(args.period_min if args.period_min is not None else saved["period"].min())
        upper=float(args.period_max if args.period_max is not None else saved["period"].max())
        variant=int(saved.get("preferred_variant",DEFAULT_VARIANT)) if args.variant is None else int(args.variant)
        bundle=feature_bundle(pd.read_parquet(args.input),lower,upper,len(saved["period"]),include_controls=variant in [3,11])
        prediction,probability=predict_bundle(bundle,saved,max(args.case,0),variant)
    prediction["period_unit"]=unit
    if unit=="h":
        prediction["estimate_h"]=prediction.estimate_period
        prediction["experiment_mean_h"]=prediction.experiment_mean
        prediction["individual_deviation_h"]=prediction.individual_deviation
        prediction["source_mean_h"]=prediction.source_mean
        prediction["calibration_shift_h"]=prediction.mean_link_shift
        prediction["profile_shift_h"]=0.
        prediction["support_shift_h"]=prediction.support_shift
    args.output.parent.mkdir(parents=True,exist_ok=True)
    prediction.to_csv(args.output,index=False,encoding="utf-8")
    experiments=prediction.groupby(["run","condition"],sort=False).agg(n_input=("trace_id","size"),estimate_mean=("estimate_period","mean")).reset_index()
    if unit=="h":
        experiments["estimate_mean_h"]=experiments.estimate_mean
    experiments.to_csv(args.output.with_name(args.output.stem+"_experiments.csv"),index=False,encoding="utf-8")
    conditions=experiments.groupby("condition",sort=False).agg(n_experiments=("run","size"),estimate_mean=("estimate_mean","mean")).reset_index()
    control=conditions.loc[conditions.condition.eq(args.control_condition),["estimate_mean"]].rename(columns={"estimate_mean":"control_mean"})
    contrasts=conditions.loc[~conditions.condition.eq(args.control_condition)].merge(control,how="cross")
    contrasts["estimate_effect"]=contrasts.estimate_mean-contrasts.control_mean
    contrasts["period_unit"]=unit
    if unit=="h":
        contrasts["estimate_effect_h"]=contrasts.estimate_effect
    contrasts.to_csv(args.output.with_name(args.output.stem+"_contrasts.csv"),index=False,encoding="utf-8")
    np.savez_compressed(args.output.with_name(args.output.stem+"_source.npz"),trace_id=bundle["trace_id"],period=bundle["period"],source_probability=probability)
    print("UNIFIED_PREDICTED",len(prediction),"records",len(experiments),"groups",str(args.output),flush=True)
    return prediction


def _role_terms(role,base,quality=None):
    repeated=np.broadcast_to(base,(*role.shape[:-1],base.shape[-1]))
    left,right=np.triu_indices(role.shape[-1])
    terms=np.concatenate([role,(role[...,None]*repeated[...,None,:]).reshape(*role.shape[:-1],-1),role[...,left]*role[...,right]],axis=-1)
    return quality_design(terms,quality)


def _role_score(role,base,coefficient,quality=None):
    """Contract linear, interaction and quadratic terms of the measured roles."""
    dimension=base.shape[-1];roles=role.shape[-1]
    role_dimension=roles*(dimension+1)+roles*(roles+1)//2
    if coefficient.shape[-1]==2*role_dimension:
        low=_role_score(role,base,coefficient[:,:role_dimension])
        high=_role_score(role,base,coefficient[:,role_dimension:])
        return low*(1.-quality)[None,:,None]+high*quality[None,:,None]
    cross=coefficient[:,roles:roles*(dimension+1)].reshape(len(coefficient),roles,dimension)
    interaction=np.einsum("npd,fkd->fnpk",base,cross,optimize=True)
    left,right=np.triu_indices(roles)
    return np.einsum("fnpk,fk->fnp",role,coefficient[:,:roles],optimize=True)+np.sum(role*interaction,axis=-1)+np.einsum("fnpk,fk->fnp",role[...,left]*role[...,right],coefficient[:,roles*(dimension+1):],optimize=True)


def _curve_numpy(source,x,y,count):
    return interpolate_mean(source,x,y,count,identity_tails=True)


def _curve_torch(source,x,y,count):
    valid=torch.arange(x.shape[1],device=x.device)[None]<count[:,None]
    lower=(torch.sum(source[:,:,None]>=torch.where(valid[:,None],x[:,None],torch.inf),dim=-1)-1).clamp_min(0)
    lower=torch.minimum(lower,count[:,None]-1)
    upper=torch.minimum(lower+1,count[:,None]-1)
    x0=x.gather(1,lower);x1=x.gather(1,upper)
    y0=y.gather(1,lower);y1=y.gather(1,upper)
    last=x.gather(1,(count-1)[:,None])
    clipped=torch.minimum(torch.maximum(source,x[:,0,None]),last)
    fraction=torch.where(x1>x0,(clipped-x0)/(x1-x0).clamp_min(1e-12),torch.zeros_like(source))
    return y0+fraction*(y1-y0)+source-clipped


def _role_conditioning(features,weight,effective_n,scale_weight=None):
    dimension=features.shape[-1]
    native_weight=weight if scale_weight is None else scale_weight
    mean=np.einsum("fn,fnpd->fd",native_weight,features,optimize=True)/features.shape[2]
    variance=np.einsum("fn,fnpd->fd",native_weight,features**2,optimize=True)/features.shape[2]-mean**2
    scale=np.sqrt(np.maximum(variance,1e-12)) if scale_weight is None else np.sqrt(np.maximum(variance,0.))
    scale=np.where(scale>0.,scale,1.)
    centered=features-features.mean(2,keepdims=True)
    covariance=np.einsum("fn,fnpd,fnpe->fde",weight,centered,centered,optimize=True)/features.shape[2]/(scale[:,:,None]*scale[:,None,:])
    covariance+=np.eye(dimension)[None]/effective_n[:,None,None]
    eig,vec=np.linalg.eigh(covariance)
    transform=np.einsum("fik,fk,fjk->fij",vec,1/np.sqrt(np.maximum(eig,1e-12)),vec,optimize=True)
    return scale,transform


def _fit_period_role(base,role,fixed,target,qualified,train,period,maxiter,device,cohort=None,quality=None,interaction_units=None,reference_count=None,training_weight=None,cohort_weight=None,initial_coefficient=None):
    """Native source-role reference model preceding representation adaptation."""
    features=_role_terms(role,base,quality)
    n=train.sum(1) if reference_count is None else reference_count
    weight=train/train.sum(1)[:,None] if training_weight is None else np.asarray(training_weight,dtype=float)
    scale,transform=_role_conditioning(features,weight,n)
    prior_transform=transform/np.sqrt(n[:,None,None])
    if quality is not None:
        prior_transform=conditional_prior_transform(features,quality,weight,scale,transform,n,interaction_units)
    teacher=np.maximum(1-abs(period[None]-target[:,None])/(period[1]-period[0]),0)*qualified[:,None]
    teacher/=np.maximum(teacher.sum(1),1e-12)[:,None]
    sd=np.sqrt(np.maximum(weight@(target**2)-(weight@target)**2,(period[1]-period[0])**2/12))
    own=base[:,:,0].argmax(1)
    peer=np.where(np.any(base[:,:,4]!=0,axis=1),base[:,:,4].argmax(1),own)
    rivals=np.stack([own,peer],axis=-1)
    negative=teacher[np.arange(len(target))[:,None],rivals]==0
    dtype=torch.float64
    tx=torch.as_tensor(features,dtype=dtype,device=device);tf=torch.as_tensor(fixed,dtype=dtype,device=device)
    ts=torch.as_tensor(scale,dtype=dtype,device=device);tt=torch.as_tensor(transform,dtype=dtype,device=device)
    tw=torch.as_tensor(weight,dtype=dtype,device=device);ty=torch.as_tensor(teacher,dtype=dtype,device=device)
    tn=torch.as_tensor(n,dtype=dtype,device=device);tstd=torch.as_tensor(sd,dtype=dtype,device=device)
    tprior_transform=torch.as_tensor(prior_transform,dtype=dtype,device=device)
    tr=torch.as_tensor(rivals,dtype=torch.long,device=device)[None].expand(len(train),-1,-1)
    tneg=torch.as_tensor(negative,dtype=dtype,device=device)

    if cohort is not None:
        membership,average,reference,training_run=cohort
        group_allowed=training_run&((membership@qualified)>0.)[None]
        group_weight=group_allowed/group_allowed.sum(1)[:,None] if cohort_weight is None else np.asarray(cohort_weight,dtype=float)
        tgroup_weight=torch.as_tensor(group_weight,dtype=dtype,device=device)
        taverage=torch.as_tensor(average,dtype=dtype,device=device)
        treference=torch.as_tensor(np.nan_to_num(reference,nan=0.),dtype=dtype,device=device)
        tperiod=torch.as_tensor(period,dtype=dtype,device=device)

    def objective(parameter,include_mean=False):
        normalized=torch.einsum("fij,fj->fi",tt,parameter.reshape(scale.shape))
        score=tf+torch.einsum("fnpd,fd->fnp",tx,normalized/ts)
        log=torch.log_softmax(score,-1);mass=log.exp()
        ce=-(log*ty[None]*tw[:,:,None]).sum((-1,-2))
        ordered=((mass.cumsum(-1)[:,:,:-1]-ty.cumsum(-1)[None,:,:-1]).square().sum(-1)*tw).sum(-1)*float(period[1]-period[0])/tstd
        positive=(score*ty[None]).sum(-1)
        ranking=((torch.nn.functional.softplus(score.gather(-1,tr)-positive[:,:,None])*tneg[None]).sum(-1)/tneg.sum(-1).clamp_min(1.)[None]*tw).sum(-1)
        prior_residual=torch.einsum("fij,fj->fi",tprior_transform,parameter.reshape(scale.shape))
        loss=ce+ordered+ranking+.5*prior_residual.square().sum(-1)
        if include_mean:
            mean=(mass@tperiod)@taverage.T
            loss=loss+.5*((mean-treference[None]).square()*tgroup_weight).sum(-1)/tstd.square()
        return loss.sum()

    gradient=torch.func.grad_and_value(objective)
    scipy_objective=lambda parameter:(pair:=gradient(torch.as_tensor(parameter,dtype=dtype,device=device)),(pair[1].item(),pair[0].detach().cpu().numpy()))[-1]
    initial=np.zeros(scale.size) if initial_coefficient is None else np.linalg.solve(transform,(initial_coefficient*scale)[:,:,None])[:,:,0].ravel()
    fit=minimize(scipy_objective,initial,jac=True,method="L-BFGS-B",options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-4,"maxcor":25})
    proper_success=bool(fit.success)
    if cohort is not None:
        mean_gradient=torch.func.grad_and_value(lambda parameter:objective(parameter,True))
        mean_objective=lambda parameter:(pair:=mean_gradient(torch.as_tensor(parameter,dtype=dtype,device=device)),(pair[1].item(),pair[0].detach().cpu().numpy()))[-1]
        fit=minimize(mean_objective,fit.x,jac=True,method="L-BFGS-B",options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-4,"maxcor":25})
    coefficient=np.einsum("fij,fj->fi",transform,fit.x.reshape(scale.shape),optimize=True)/scale
    print("REFERENCE_ROLE",fit.success,fit.nit,float(np.max(abs(fit.jac))),flush=True)
    return coefficient,{"success":proper_success and bool(fit.success),"iterations":fit.nit,"gradient":float(np.max(abs(fit.jac)))}


def _fit_mean_role(bundle,base,role,fixed,prior,reference,training_run,curve,sd,effective_units,maxiter,device,view_context=None,initial_coefficient=None,period_reference=None,core_fit=None,prior_scale_weight=None,proper_weight_override=None,interaction_units=None):
    """Joint reference-source increments, proper period scores and fixed mean map."""
    roles=role.shape[-1]
    role_dimension=roles*(base.shape[-1]+1)+roles*(roles+1)//2
    quality=bundle["mixing_quality"] if prior.shape[-1]==2*role_dimension else None
    features=_role_terms(role,base,quality)
    core_dimension=0
    if core_fit is not None:
        core_design,core_prior,initial_core=core_fit
        core_dimension=core_design.shape[-1]
        features=np.concatenate([np.broadcast_to(core_design,(len(prior),*core_design.shape)),features],axis=-1)
        prior=np.concatenate([np.zeros_like(core_prior),prior],axis=-1)
        if initial_coefficient is not None:
            initial_coefficient=np.concatenate([initial_core-core_prior,initial_coefficient],axis=-1)
    groups=training_run&np.isfinite(reference)[None]
    average=bundle["average"]
    original_group=np.ones(len(reference),dtype=bool)
    if view_context is not None:
        original_group=(bundle["membership"]@view_context["is_original"].astype(float))>0
    origin_weight=(groups&original_group[None]).astype(float)
    origin_weight/=origin_weight.sum(1)[:,None]
    group_weight=groups.astype(float)
    if view_context is not None:
        view_weight=average@view_context["view_weight"]
        group_weight*=view_weight[None]
    group_weight/=group_weight.sum(1)[:,None]
    if view_context is not None and "group_weight" in view_context:
        group_weight=view_context["group_weight"]
        origin_weight=view_context["origin_group_weight"]
    record_weight=origin_weight@average
    if view_context is not None:
        record_weight=group_weight@average
    scale,transform=_role_conditioning(features,record_weight,effective_units,scale_weight=prior_scale_weight)
    prior_transform=transform/np.sqrt(effective_units[:,None,None])
    if quality is not None and core_fit is None:
        units=effective_units if interaction_units is None else interaction_units
        native_weight=record_weight if prior_scale_weight is None else prior_scale_weight
        prior_transform=conditional_prior_transform(features,quality,native_weight,scale,transform,effective_units,units)
    proper=None
    if period_reference is not None:
        period_axis=bundle["period"]/float(bundle["period_scale"])
        source_target,source_qualified=period_reference
        source_target=np.where(source_qualified,source_target,0.)
        teacher=np.maximum(1.-abs(period_axis[None]-source_target[:,None])/(period_axis[1]-period_axis[0]),0.)*source_qualified[:,None]
        teacher/=np.maximum(teacher.sum(1),1e-12)[:,None]
        qualified_count=bundle["membership"]@source_qualified.astype(float)
        # Each original experiment retains its view share; 777 driver labels
        # do not outweigh the other views' 506 qualified source references.
        proper_group_weight=group_weight*(qualified_count>0.)[None]
        proper_group_weight/=proper_group_weight.sum(1)[:,None]
        proper_weight=proper_group_weight[:,bundle["run_index"]]*source_qualified[None]/np.maximum(qualified_count[bundle["run_index"]],1.)[None]
        if proper_weight_override is not None:
            proper_weight=proper_weight_override
        proper=(teacher,proper_weight)
    dtype=torch.float64
    tx=torch.as_tensor(features,dtype=dtype,device=device);tf=torch.as_tensor(fixed,dtype=dtype,device=device)
    ts=torch.as_tensor(scale,dtype=dtype,device=device);tt=torch.as_tensor(transform,dtype=dtype,device=device)
    tprior=torch.as_tensor(prior,dtype=dtype,device=device);tp=torch.as_tensor(bundle["period"]/float(bundle["period_scale"]),dtype=dtype,device=device)
    ta=torch.as_tensor(average,dtype=dtype,device=device);tr=torch.as_tensor(np.nan_to_num(reference,nan=0.),dtype=dtype,device=device)
    tw=torch.as_tensor(group_weight,dtype=dtype,device=device);torig=torch.as_tensor(origin_weight,dtype=dtype,device=device)
    tcx=torch.as_tensor(curve[0],dtype=dtype,device=device);tcy=torch.as_tensor(curve[1],dtype=dtype,device=device)
    tcn=torch.as_tensor(curve[2],dtype=torch.long,device=device);tstd=torch.as_tensor(sd,dtype=dtype,device=device)
    tn=torch.as_tensor(effective_units,dtype=dtype,device=device)
    tprior_transform=torch.as_tensor(prior_transform,dtype=dtype,device=device)
    if proper is not None:
        tteacher=torch.as_tensor(teacher,dtype=dtype,device=device)
        tproper=torch.as_tensor(proper_weight,dtype=dtype,device=device)
        rivals=np.stack([base[:,:,0].argmax(1),base[:,:,4].argmax(1)],axis=-1)
        negative=teacher[np.arange(len(source_target))[:,None],rivals]==0.
        trivals=torch.as_tensor(rivals,dtype=torch.long,device=device)[None].expand(len(prior),-1,-1)
        tnegative=torch.as_tensor(negative,dtype=dtype,device=device)
    pairs=None
    if view_context is not None:
        pairs=(torch.as_tensor(view_context["pair_left"],dtype=torch.long,device=device),torch.as_tensor(view_context["pair_right"],dtype=torch.long,device=device),
            torch.as_tensor(view_context["pair_weight"],dtype=dtype,device=device),torch.as_tensor(view_context["pair_effect"],dtype=dtype,device=device),
            torch.as_tensor(view_context.get("pair_scale",sd[:,None]),dtype=dtype,device=device))

    locate=view_context is None or view_context.get("mean_location",True)
    def objective(parameter,readout=False):
        normalized=torch.einsum("fij,fj->fi",tt,parameter.reshape(scale.shape))
        coefficient=tprior+normalized/ts
        score=tf+torch.einsum("fnpd,fd->fnp",tx,coefficient)
        mass=torch.softmax(score,-1);source=mass@tp
        calibrated=_curve_torch(source@ta.T,tcx,tcy,tcn)
        location=((tr[None]-calibrated)*torig).sum(-1) if locate else torch.zeros(len(prior),dtype=dtype,device=device)
        mean=calibrated+location[:,None]
        prior_residual=torch.einsum("fij,fj->fi",tprior_transform,parameter.reshape(scale.shape))
        loss=.5*(((mean-tr[None])/tstd[:,None]).square()*tw).sum(-1)+.5*prior_residual.square().sum(-1)
        if proper is not None:
            log=torch.log_softmax(score,-1)
            ce=-(log*tteacher[None]*tproper[:,:,None]).sum((-1,-2))
            ordered=((mass.cumsum(-1)[:,:,:-1]-tteacher.cumsum(-1)[None,:,:-1]).square()*tproper[:,:,None]).sum((-1,-2))*float(period_axis[1]-period_axis[0])/tstd
            loss=loss+ce+ordered
            if view_context is not None and view_context.get("proper_ranking",False):
                positive=(score*tteacher[None]).sum(-1)
                ranking=(torch.nn.functional.softplus(score.gather(-1,trivals)-positive[:,:,None])*tnegative[None]).sum(-1)/tnegative.sum(-1).clamp_min(1.)[None]
                loss=loss+(ranking*tproper).sum(-1)
        if pairs is not None:
            left,right,pair_weight,effect,pair_scale=pairs
            pair_mean=source@ta.T if view_context.get("pair_readout","mean")=="source" else mean
            difference=pair_mean[:,right]-pair_mean[:,left]-effect[None]
            loss=loss+.5*((difference/pair_scale).square()*pair_weight).sum(-1)
        if readout=="loss":
            return loss
        return (coefficient,mass,source,mean,location) if readout else loss.sum()

    gradient=torch.func.grad_and_value(objective)
    scipy_objective=lambda parameter:(pair:=gradient(torch.as_tensor(parameter,dtype=dtype,device=device)),(pair[1].item(),pair[0].detach().cpu().numpy()))[-1]
    initial=np.zeros(scale.shape) if initial_coefficient is None else np.linalg.solve(transform,((initial_coefficient-prior)*scale)[:,:,None])[:,:,0]
    fit=minimize(scipy_objective,initial.ravel(),jac=True,method="L-BFGS-B",options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-4,"maxcor":25})
    parameter=fit.x.reshape(scale.shape).copy()
    case_success=np.full(len(scale),bool(fit.success))
    case_iterations=np.full(len(scale),fit.nit,dtype=int)
    case_gradient=np.max(abs(fit.jac.reshape(scale.shape)),axis=1)
    case_message=np.repeat(str(fit.message),len(scale)).astype(object)
    case_solver=np.repeat("L-BFGS-B",len(scale)).astype(object)
    if not fit.success:
        # The fixed PAV map is piecewise linear. Continue each independent case
        # with exactly the same squared group, pair and Gaussian-prior residuals.
        # The small dense TRF Jacobian is evaluated on CPU to bound GPU memory.
        period_axis=bundle["period"]/float(bundle["period_scale"])
        reference_value=np.nan_to_num(reference,nan=0.)

        def solve_case(index):
            design=np.einsum("npd,dk->npk",features[index]/scale[index],transform[index],optimize=True)
            offset=fixed[index]+np.einsum("npd,d->np",features[index],prior[index],optimize=True)
            knot_count=int(curve[2][index])
            knot_x=curve[0][index,:knot_count]
            knot_y=curve[1][index,:knot_count]
            origin=origin_weight[index]
            group_factor=np.sqrt(group_weight[index])/sd[index]
            root_n=np.sqrt(effective_units[index])

            def residual_jacobian(theta):
                normalized=transform[index]@theta
                score=offset+np.einsum("npd,d->np",design,theta,optimize=True)
                log_probability=score-logsumexp(score,axis=-1,keepdims=True)
                probability=np.exp(log_probability)
                individual=probability@period_axis
                raw_mean=average@individual
                lower=np.clip(np.searchsorted(knot_x,raw_mean,side="right")-1,0,knot_count-1)
                upper=np.minimum(lower+1,knot_count-1)
                dx=knot_x[upper]-knot_x[lower]
                denominator=np.maximum(dx,1e-12)
                clipped=np.clip(raw_mean,knot_x[0],knot_x[-1])
                fraction=np.divide(clipped-knot_x[lower],denominator,out=np.zeros_like(raw_mean),where=dx>0.)
                calibrated=knot_y[lower]+fraction*(knot_y[upper]-knot_y[lower])+raw_mean-clipped
                location=np.sum((reference_value-calibrated)*origin) if locate else 0.
                mean=calibrated+location
                point_jacobian=np.einsum("np,npd->nd",probability*(period_axis[None]-individual[:,None]),design,optimize=True)
                raw_jacobian=average@point_jacobian
                maximum=np.maximum(raw_mean,knot_x[0])
                clip_derivative=((raw_mean>knot_x[0])+.5*(raw_mean==knot_x[0]))*((maximum<knot_x[-1])+.5*(maximum==knot_x[-1]))
                slope=np.divide(knot_y[upper]-knot_y[lower],denominator,out=np.zeros_like(raw_mean),where=dx>0.)
                calibrated_jacobian=raw_jacobian*(slope*clip_derivative+1.-clip_derivative)[:,None]
                mean_jacobian=calibrated_jacobian-(origin@calibrated_jacobian)[None] if locate else calibrated_jacobian
                residual=np.r_[(mean-reference_value)*group_factor,prior_transform[index]@theta]
                jacobian=np.concatenate([mean_jacobian*group_factor[:,None],prior_transform[index]],axis=0)
                if view_context is not None:
                    left=view_context["pair_left"];right=view_context["pair_right"]
                    factor=np.sqrt(view_context["pair_weight"][index])/view_context.get("pair_scale",sd[:,None])[index]
                    pair_mean=raw_mean if view_context.get("pair_readout","mean")=="source" else mean
                    pair_jacobian=raw_jacobian if view_context.get("pair_readout","mean")=="source" else mean_jacobian
                    pair=(pair_mean[right]-pair_mean[left]-view_context["pair_effect"])*factor
                    residual=np.r_[residual,pair]
                    jacobian=np.concatenate([jacobian,(pair_jacobian[right]-pair_jacobian[left])*factor[:,None]],axis=0)
                if proper is not None:
                    selected_period=np.argsort(teacher,axis=1)[:,-2:]
                    selected_weight=proper_weight[index,:,None]*np.take_along_axis(teacher,selected_period,axis=1)
                    selected_log=np.take_along_axis(log_probability,selected_period,axis=1)
                    ce_residual=np.sqrt(np.maximum(-2.*selected_weight*selected_log,0.))
                    expected=np.einsum("np,npd->nd",probability,design,optimize=True)
                    selected_design=np.take_along_axis(design,selected_period[:,:,None],axis=1)-expected[:,None]
                    ce_factor=np.divide(-selected_weight,ce_residual,out=np.zeros_like(ce_residual),where=ce_residual>0.)
                    ce_jacobian=selected_design*ce_factor[:,:,None]
                    ordered_factor=np.sqrt(2.*proper_weight[index,:,None]*(period_axis[1]-period_axis[0])/sd[index])
                    ordered_residual=(probability.cumsum(-1)[:,:-1]-teacher.cumsum(-1)[:,:-1])*ordered_factor
                    ordered_jacobian=np.cumsum(probability[:,:,None]*(design-expected[:,None]),axis=1)[:,:-1]*ordered_factor[:,:,None]
                    residual=np.r_[residual,ce_residual.ravel(),ordered_residual.ravel()]
                    jacobian=np.concatenate([jacobian,ce_jacobian.reshape(-1,design.shape[-1]),ordered_jacobian.reshape(-1,design.shape[-1])],axis=0)
                    if view_context is not None and view_context.get("proper_ranking",False):
                        positive=np.sum(score*teacher,axis=-1)
                        error=score[np.arange(len(score))[:,None],rivals]-positive[:,None]
                        rank_weight=proper_weight[index,:,None]*negative/np.maximum(negative.sum(-1),1.)[:,None]
                        rank_residual=np.sqrt(2.*rank_weight*np.logaddexp(0.,error))
                        positive_design=np.einsum("np,npd->nd",teacher,design,optimize=True)
                        rank_design=design[np.arange(len(score))[:,None],rivals]-positive_design[:,None]
                        rank_factor=np.divide(rank_weight/(1.+np.exp(np.clip(-error,-700.,700.))),rank_residual,out=np.zeros_like(rank_residual),where=rank_residual>0.)
                        residual=np.r_[residual,rank_residual.ravel()]
                        jacobian=np.concatenate([jacobian,(rank_design*rank_factor[:,:,None]).reshape(-1,design.shape[-1])],axis=0)
                return residual,jacobian

            # A fixed deterministic probe avoids comparing opposite one-sided
            # derivatives at the PAV knot that stopped the line search.
            checked_parameter=parameter.copy()
            checked_parameter[index]+=1e-5*np.linspace(1./scale.shape[1],1.,scale.shape[1])
            checked_residual,checked_jacobian=residual_jacobian(checked_parameter[index])
            checked_gradient=scipy_objective(checked_parameter.ravel())[1].reshape(scale.shape)[index]
            np.testing.assert_allclose(checked_jacobian.T@checked_residual,checked_gradient,rtol=1e-7,atol=1e-7)
            checked_loss=objective(torch.as_tensor(checked_parameter.ravel(),dtype=dtype,device=device),"loss").detach().cpu().numpy()[index]
            np.testing.assert_allclose(.5*np.sum(checked_residual**2),checked_loss,rtol=1e-9,atol=1e-9)
            print("SOURCE_OBJECTIVE_CHECK",index,float(abs(.5*np.sum(checked_residual**2)-checked_loss)),
                float(np.max(abs(checked_jacobian.T@checked_residual-checked_gradient))),flush=True)
            refined=least_squares(lambda theta:residual_jacobian(theta)[0],parameter[index],
                jac=lambda theta:residual_jacobian(theta)[1],method="trf",tr_solver="exact",
                max_nfev=maxiter,ftol=1e-10,xtol=1e-10,gtol=1e-6,x_scale="jac")
            parameter[index]=refined.x
            case_success[index]=refined.success
            case_iterations[index]+=refined.nfev
            case_gradient[index]=np.max(abs(refined.grad))
            case_message[index]=str(refined.message)
            case_solver[index]="L-BFGS-B_then_TRF"
            print("FIXED_MEAN_ROLE_CASE",index,refined.success,refined.nfev,float(refined.cost),
                float(refined.optimality),str(refined.message),flush=True)
            return None

        list(map(solve_case,np.arange(len(scale))))
    coefficient,mass,source,mean,location=map(lambda value:value.detach().cpu().numpy(),objective(torch.as_tensor(parameter.ravel(),dtype=dtype,device=device),True))
    fitted_core=None
    if core_fit is not None:
        fitted_core=core_prior+coefficient[:,:core_dimension]
        coefficient=coefficient[:,core_dimension:]
    print("FIXED_MEAN_ROLE","views" if view_context is not None else "native",bool(case_success.all()),
        int(case_iterations.max()),float(case_gradient.max()),flush=True)
    return {"role_coefficient":coefficient,"core_coefficient":fitted_core,"posterior":mass,"source_period":source,"group_mean":mean,"mean_bias":location,
        "success":bool(case_success.all()),"iterations":int(case_iterations.max()),"gradient":float(case_gradient.max()),
        "case_success":case_success,"case_iterations":case_iterations,"case_gradient":case_gradient,
        "case_message":case_message.astype(str),"case_solver":case_solver.astype(str)}


def _source_fields(bundle,base,raw_base,core,identity_core,background_coefficient,covariance,foreground_prior,background_prior,variant,center,geometry_scale,field=None,parent=False):
    raw_poly=_polynomial(raw_base)
    background=softmax(background_prior[:,None]+np.einsum("npd,fd->fnp",_background_features(base),background_coefficient,optimize=True),axis=-1)
    if variant==10:
        foreground=foreground_prior[:,None]+np.einsum("npd,fd->fnp",raw_poly,identity_core,optimize=True)
        role=np.stack([foreground-logsumexp(foreground,axis=-1,keepdims=True),np.log(np.maximum(background,1e-300))],axis=-1)
    else:
        role=_role_features(bundle,covariance,background,variant,field=field)
    core_base=base.copy()
    if variant not in [2,6]:
        if parent:
            core_base[:,:,11]=bundle["parent_coherent_power"]
        else:
            prefix={1:"unadjusted",8:"original",9:"independent"}.get(variant,"source")
            core_base[:,:,11]=bundle["source_coherent_global" if variant==15 else prefix+"_coherent_half"]
    design=_source_design(bundle,core_base,center,geometry_scale,variant)
    fixed=foreground_prior[:,None]+np.einsum("npd,fd->fnp",design,core,optimize=True)
    return role,fixed


def _within_scale(bundle,anchor,foreground_prior,target,qualified,training_run,variant):
    raw=_identity_base_variant(bundle,variant)
    period=bundle["period"]/float(bundle["period_scale"])
    probability=softmax(foreground_prior[:,None]+np.einsum("npd,fd->fnp",_polynomial(raw),anchor,optimize=True),axis=-1)
    original=probability@period
    deviation=original-(original@bundle["average"].T)[:,bundle["run_index"]]
    label=np.where(qualified,target/float(bundle["period_scale"]),0.)
    reference=(bundle["membership"]@label)/np.maximum(bundle["membership"]@qualified,1.)
    train=training_run[:,bundle["run_index"]]&qualified[None]
    n=train.sum(1);weight=train/n[:,None]
    variance=np.maximum(weight@(label**2)-(weight@label)**2,(period[1]-period[0])**2/12)
    residual=label-reference[bundle["run_index"]]
    numerator=np.sum(weight*deviation*residual[None],axis=1)+variance/n
    denominator=np.sum(weight*deviation**2,axis=1)+variance/n
    return np.maximum(numerator/denominator,0.),original


def _fit_pipeline(bundle,reference,qualified,training_run,background_reference,independent_unit,variant,maxiter,device,view_bundle,view_context,raw_cache,background_cache):
    period=bundle["period"]
    scale=float(bundle["period_scale"])
    target=np.where(qualified,reference,0.)
    train=training_run[:,bundle["run_index"]]&qualified[None]
    group_valid=(bundle["membership"]@qualified)>0
    train_groups=training_run&group_valid[None]
    group_reference=(bundle["membership"]@(target/scale))/np.maximum(bundle["membership"]@qualified,1.)
    group_reference=np.where(group_valid,group_reference,np.nan)
    unit_index,units=pd.factorize(independent_unit,sort=True)
    unit_membership=(np.arange(len(units))[:,None]==unit_index[None]).astype(float)
    foreground_prior,effective_units=_reference_prior(target,qualified,train,period,unit_membership)
    if variant==14:
        foreground_prior[:]=-np.log(len(period))
    group_geometry=bundle["average"]@_geometry_coordinate(bundle)
    center=float(np.median(group_geometry));geometry_scale=max(float(np.std(group_geometry)),1e-12)
    raw_base=_base_variant(bundle,variant)
    raw_key=(variant==2,variant==14)
    if raw_key not in raw_cache:
        _,identity,status=_fit_core(raw_base,target,qualified,train,period,maxiter,foreground_prior)
        identity_base=_identity_base_variant(bundle,variant)
        identity_poly=_polynomial(identity_base)
        left,right=np.triu_indices(13)
        phase=np.concatenate([np.arange(13)==12,(left==12)|(right==12)])
        anchor,anchor_status=_fit_score(identity_poly,target,qualified,train,period,np.flatnonzero(~phase),
            parent=foreground_prior[:,None],ordered=True,maxiter=maxiter)
        raw_cache[raw_key]=(anchor,identity,status+[anchor_status])
    anchor,identity_core,core_status=raw_cache[raw_key]
    background_key=(raw_key,variant==7)
    if background_key not in background_cache:
        proxy=period[bundle["peer_spectral_power"].argmax(1)]
        labels=proxy if variant==7 else np.where(np.isfinite(background_reference),background_reference,proxy)
        bg_train=training_run[:,bundle["run_index"]]&qualified[None]
        bg_prior,_=_reference_prior(labels,qualified,bg_train,period,unit_membership)
        if variant==14:
            bg_prior[:]=-np.log(len(period))
        bg_coefficient,bg_status=_fit_score(_background_features(raw_base),labels,qualified,bg_train,period,parent=bg_prior[:,None],maxiter=maxiter)
        bg_index=abs(period[None]-labels[:,None]).argmin(1)
        covariance,cov_status=_fit_covariance(bundle,target,qualified,bg_train,bg_index,device,maxiter)
        background_cache[background_key]=(bg_coefficient,covariance,bg_prior,bg_status,cov_status)
    bg_coefficient,covariance,bg_prior,bg_status,cov_status=background_cache[background_key]
    background=softmax(bg_prior[:,None]+np.einsum("npd,fd->fnp",_background_features(raw_base),bg_coefficient,optimize=True),axis=-1)
    raw_role=_role_features(bundle,covariance,background,variant,field="original_rhs") if variant!=10 else np.stack([
        foreground_prior[:,None]+np.einsum("npd,fd->fnp",_polynomial(raw_base),identity_core,optimize=True),np.log(np.maximum(background,1e-300))],axis=-1)
    if variant==10:
        raw_role[:,:,:,0]-=logsumexp(raw_role[:,:,:,0],axis=-1,keepdims=True)
    raw_fixed=foreground_prior[:,None]+np.einsum("npd,fd->fnp",_polynomial(raw_base),identity_core,optimize=True)
    case_blocks=np.array_split(np.arange(len(training_run)),max(1,int(np.ceil(len(training_run)/2))))

    def reference_role(selected):
        return _fit_period_role(raw_base,raw_role[selected],raw_fixed[selected],target/scale,qualified,train[selected],period/scale,maxiter,device)

    raw_roles=list(map(reference_role,case_blocks))
    raw_role_coefficient=np.concatenate(list(map(lambda value:value[0],raw_roles)),axis=0)
    raw_role_status=list(map(lambda value:value[1],raw_roles))
    parent_base=raw_base.copy()
    if variant not in [2,6]:
        parent_base[:,:,11]=bundle["parent_coherent_power"]
    parent_design=_source_design(bundle,parent_base,center,geometry_scale,variant)
    parent_prior=np.concatenate([identity_core,np.zeros_like(identity_core)],axis=1)
    fixed=foreground_prior[:,None]+np.einsum("npd,fd->fnp",parent_design,parent_prior,optimize=True)+_role_score(raw_role,raw_base,raw_role_coefficient)
    own=raw_base[:,:,0].argmax(1)
    peer=np.where(np.any(raw_base[:,:,4]!=0,axis=1),raw_base[:,:,4].argmax(1),own)
    increment,parent_status=_fit_score(parent_design,target,qualified,train,period,parent=fixed,ordered=True,ranking=True,rivals=np.stack([own,peer],axis=-1),maxiter=maxiter)
    core=parent_prior+increment
    parent_core=core.copy()
    parent_probability=softmax(foreground_prior[:,None]+np.einsum("npd,fd->fnp",parent_design,core,optimize=True)+_role_score(raw_role,raw_base,raw_role_coefficient),axis=-1)
    parent_mean=(parent_probability@(period/scale))@bundle["average"].T
    _,cal_x,cal_y,cal_n=fit_monotone_mean(parent_mean,np.nan_to_num(group_reference,nan=0.),train_groups)
    curve=(cal_x,cal_y,cal_n)
    weight=train/train.sum(1)[:,None]
    sd=np.sqrt(np.maximum(weight@((target/scale)**2)-(weight@(target/scale))**2,(period[1]-period[0])**2/(12*scale**2)))
    prior_coefficient=raw_role_coefficient.copy()
    prior_bias=np.zeros(len(training_run))
    view_status=[]

    if view_bundle is not None and variant!=12:
        view_raw=_base_variant(view_bundle,variant)
        view_base=_source_base_variant(view_bundle,variant)
        view_training=view_context["training_run"]
        view_reference=view_context["group_reference"]/scale
        core_width=208 if variant==13 else 104
        view_core_base=_core_base_variant(view_bundle,variant)
        view_core_design=_source_design(view_bundle,view_core_base,center,geometry_scale,variant) if variant==13 else _polynomial(view_core_base)

        def view_fit(selected):
            role,fixed_view=_source_fields(view_bundle,view_base,view_raw,core[selected],identity_core[selected],bg_coefficient[selected],covariance[selected],foreground_prior[selected],bg_prior[selected],variant,center,geometry_scale)
            eligible=view_context["pair_eligible"][None]&view_training[selected][:,view_context["pair_left"]]&view_training[selected][:,view_context["pair_right"]]
            pair_weight=eligible/np.maximum(eligible.sum(1),1)[:,None]
            current=dict(view_context)
            current["pair_weight"]=pair_weight
            current["pair_effect"]=view_context["pair_effect"]/scale
            return _fit_mean_role(view_bundle,view_base,role,fixed_view,raw_role_coefficient[selected],view_reference,view_training[selected],
                (cal_x[selected],cal_y[selected],cal_n[selected]),sd[selected],effective_units[selected],maxiter,device,current,
                period_reference=(view_context["target"]/scale,view_context["qualified"]),
                core_fit=(view_core_design,parent_core[selected,:core_width],None))

        # A single joint 104+31 case bounds the source-design tensor on the
        # shared 8-GB GPU; independent case objectives remain unchanged.
        view_values=list(map(view_fit,np.arange(len(training_run))[:,None]))
        core[:,:core_width]=np.concatenate(list(map(lambda value:value["core_coefficient"],view_values)),axis=0)
        prior_coefficient=np.concatenate(list(map(lambda value:value["role_coefficient"],view_values)),axis=0)
        prior_bias=np.concatenate(list(map(lambda value:value["mean_bias"],view_values)))
        view_status=view_values
    source_base=_source_base_variant(bundle,variant)

    def native_fit(selected):
        role,native_fixed=_source_fields(bundle,source_base,raw_base,core[selected],identity_core[selected],bg_coefficient[selected],covariance[selected],foreground_prior[selected],bg_prior[selected],variant,center,geometry_scale)
        if variant==4:
            mass=softmax(native_fixed+_role_score(role,source_base,prior_coefficient[selected]),axis=-1)
            source=mass@(period/scale)
            mean=_curve_numpy(source@bundle["average"].T,cal_x[selected],cal_y[selected],cal_n[selected])+prior_bias[selected,None]
            return {"role_coefficient":prior_coefficient[selected],"posterior":mass,"source_period":source,"group_mean":mean,"mean_bias":prior_bias[selected],"success":True,"iterations":0,"gradient":0.,
                "case_success":np.ones(len(selected),dtype=bool),"case_message":np.repeat("Native mean adaptation omitted",len(selected)),
                "case_gradient":np.zeros(len(selected)),"case_iterations":np.zeros(len(selected),dtype=int),"case_solver":np.repeat("not_fitted",len(selected))}
        return _fit_mean_role(bundle,source_base,role,native_fixed,prior_coefficient[selected],group_reference,training_run[selected],
            (cal_x[selected],cal_y[selected],cal_n[selected]),sd[selected],effective_units[selected],maxiter,device,
            period_reference=(target/scale,qualified))

    native_values=list(map(native_fit,case_blocks))
    keys=["role_coefficient","posterior","source_period","group_mean","mean_bias"]
    result=dict(zip(keys,map(lambda key:np.concatenate(list(map(lambda value:value[key],native_values)),axis=0),keys)))
    alpha,original=_within_scale(bundle,anchor,foreground_prior,target,qualified,training_run,variant)
    mean=np.clip(result["group_mean"]*scale,period.min(),period.max())
    source=result["source_period"]*scale
    original=original*scale
    point_anchor=source if variant==5 else original
    deviation=point_anchor-(point_anchor@bundle["average"].T)[:,bundle["run_index"]]
    estimate=mean[:,bundle["run_index"]]+alpha[:,None]*deviation
    estimate=project_periods_with_mean(estimate,mean,bundle["membership"],period.min(),period.max())
    statuses=core_status+[bg_status,cov_status,parent_status]+raw_role_status+view_status+native_values
    reference_success=np.full(len(training_run),all(map(lambda status:status["success"],core_status+[bg_status,cov_status,parent_status])))
    reference_success&=np.concatenate(list(map(lambda pair:np.repeat(pair[1]["success"],len(pair[0])),zip(case_blocks,raw_role_status))))
    view_success=np.concatenate(list(map(lambda value:value["case_success"],view_status))) if view_status else np.ones(len(training_run),dtype=bool)
    native_success=np.concatenate(list(map(lambda value:value["case_success"],native_values)))
    native_message=np.concatenate(list(map(lambda value:value["case_message"],native_values)))
    view_message=np.concatenate(list(map(lambda value:value["case_message"],view_status))) if view_status else np.repeat("not_fitted",len(training_run))
    case_success=reference_success&view_success&native_success
    case_message=np.where(~reference_success,"Reference-stage optimizer did not converge",
        np.where(~view_success,np.char.add("Source-reference prior: ",view_message),np.char.add("Native adaptation: ",native_message)))
    result.update({"core_coefficient":core,"parent_core_coefficient":parent_core,"identity_core_coefficient":identity_core,"anchor_coefficient":anchor,"background_coefficient":bg_coefficient,"role_covariance":covariance,
        "foreground_log_prior":foreground_prior,"background_log_prior":bg_prior,"calibration_prediction":cal_x,"calibration_mean":cal_y,"calibration_experiments":cal_n,
        "raw_role_coefficient":raw_role_coefficient,"prior_role_coefficient":prior_coefficient,"prior_mean_bias":prior_bias,"within_scale":alpha,
        "source_period":source,"original_period":original,"group_mean":mean,"estimate":estimate,
        "geometry_center":np.asarray(center),"geometry_scale":np.asarray(geometry_scale),"effective_training_units":effective_units,
        "success":bool(case_success.all()),"iterations":sum(map(lambda value:value["iterations"],statuses)),
        "case_success":case_success,"case_message":case_message,"reference_success_by_case":reference_success,"view_success_by_case":view_success,
        "native_iterations_by_case":np.concatenate(list(map(lambda value:value["case_iterations"],native_values))),
        "native_gradient_by_case":np.concatenate(list(map(lambda value:value["case_gradient"],native_values))),
        "native_solver_by_case":np.concatenate(list(map(lambda value:value["case_solver"],native_values))),
        "gradient":max(map(lambda value:value["gradient"],statuses))})
    return result




def reference_source_operator_statistics(frame,components,foreground_period,background_period,
        foreground_qualified,background_qualified,period,source_balancing=False,foreground_source_id=None,background_source_id=None,source_hierarchy=False):
    """Known-source reference moments under the same No-frozen observation map.

    components are physical FG/BG/remainder contributions in observations order.
    Only No determines the balanced operator, smoothing, gains and RMS. The
    peer component always uses the retained adjusted-raw reference.
    """
    from source_assignment_core import source_population_moments,source_noise_population_moments
    data=observations(frame);period=np.asarray(period,dtype=float);scale=float(np.sqrt(period.min()*period.max()))
    components=np.asarray(components,dtype=float);n,t,c=components.shape
    assert c==3 and components.shape[:2]==data["signal"].shape
    if source_balancing:
        _,adjusted=shared_background(data,scale)
        adjusted_components=np.stack(list(map(lambda j:shared_background(dict(data,signal=components[:,:,j]),scale)[1],np.arange(3))),axis=-1)
        observed,diagnostic=canonical_source_signal(adjusted,data,period,scale,share_order=False,components=adjusted_components)
        own_physical=diagnostic["applied_components"]*diagnostic["operator_input_rms"][:,None,None]*diagnostic["operator_output_rms"][:,None,None]
        own_observed=observed*diagnostic["operator_input_rms"][:,None]*diagnostic["operator_output_rms"][:,None]
        reference=adjusted;reference_components=adjusted_components
    else:
        reference=detrend(data["signal"],data["time"],data["mask"])
        reference_components=np.stack(list(map(lambda j:detrend(components[:,:,j],data["time"],data["mask"]),np.arange(3))),axis=-1)
        own_physical=reference_components;own_observed=reference
    population_rms=np.sqrt(data["average"]@(np.sum(reference**2,axis=1)/data["count"]))[data["run_index"]]
    own=own_physical/population_rms[:,None,None]
    peer_base=reference_components/population_rms[:,None,None]
    observed_pair=np.stack([own_observed,reference],axis=-1)/population_rms[:,None,None]
    mask=data["mask"];count=data["count"];time_axis=data["time"]
    centered=(time_axis[None]-(mask*time_axis[None]).sum(1)[:,None]/count[:,None])*mask
    trend=np.stack([mask/np.sqrt(count[:,None]),centered/np.sqrt((centered**2).sum(1))[:,None]],axis=-1)
    def projection(index):
        target=np.asarray(foreground_period if index==0 else background_period,dtype=float)
        target=np.where(np.isfinite(target)&(target>0.),target,scale)
        angle=2.*np.pi*time_axis[None]/target[:,None]
        design=np.stack([np.cos(angle),np.sin(angle)],axis=-1)*mask[:,:,None]
        design-=trend@np.einsum("nti,ntj->nij",trend,design,optimize=True)
        gram=np.einsum("nti,ntj->nij",design,design,optimize=True)
        value=np.stack([own[:,:,index],peer_base[:,:,index]],axis=-1)
        rhs=np.einsum("nti,ntv->niv",design,value,optimize=True)
        coefficient=np.linalg.solve(gram,rhs)
        fitted=np.einsum("nti,niv->ntv",design,coefficient,optimize=True)
        return coefficient.transpose(0,2,1).reshape(n,4),fitted
    fg,fg_clock=projection(0);bg,bg_clock=projection(1)
    groups=len(data["runs"])
    source_ids=data["metadata"].index.to_numpy(dtype=str) if foreground_source_id is None else np.asarray(foreground_source_id)
    shared_source=np.asarray([len(np.unique(source_ids[data["run_index"]==group]))==1 for group in range(groups)])
    restored=np.zeros((n,t))
    if source_hierarchy:
        independent=~shared_source[data["run_index"]]
        restored[independent]=fg_clock[independent,:,1]
        fg_clock[independent,:,1]=0.
    # The retained raw-reference residual now includes the complete other-FG
    # waveform in independent-source states. Its empirical E law is fixed
    # before evaluating any query Pi; only the own FG clock is subtracted.
    residual=observed_pair-fg_clock-bg_clock
    output={"source_shared":np.zeros((groups,2,4,4)),"source_private":np.zeros((groups,2,4,4)),
        "source_finite":np.zeros((groups,2,4,4)),
        "noise_shared":np.zeros((groups,2,2)),"noise_private":np.zeros((groups,2,2)),
        "source_measurements":np.zeros((groups,2),dtype=int),"source_valid":np.zeros((groups,2),dtype=bool),
        "noise_valid":np.zeros(groups,dtype=bool),"run_names":data["runs"],
        "trace_id":data["metadata"].index.to_numpy(dtype=str)}
    def cohort(group):
        own_rows=np.flatnonzero(data["run_index"]==group)
        def source(index):
            qualified=foreground_qualified if index==0 else background_qualified
            selected=own_rows[np.asarray(qualified,dtype=bool)[own_rows]]
            output["source_measurements"][group,index]=len(selected)
            if len(selected)<2:
                return None
            output["source_valid"][group,index]=True
            beta=fg[selected] if index==0 else bg[selected]
            identifiers=foreground_source_id if index==0 else background_source_id
            shared_source=identifiers is not None and len(np.unique(np.asarray(identifiers)[selected]))==1
            moment=source_population_moments(beta,np.ones(len(selected)),shared_source=shared_source)
            output["source_shared"][group,index]=moment["shared"]
            output["source_private"][group,index]=moment["private"]
            output["source_finite"][group,index]=moment["finite_loading"]
            output["source_measurements"][group,index]=len(selected)
            return None
        list(map(source,np.arange(2)))
        selected=own_rows[np.asarray(foreground_qualified,dtype=bool)[own_rows]&np.asarray(background_qualified,dtype=bool)[own_rows]]
        if len(selected)<2:
            return None
        output["noise_valid"][group]=True
        noise=source_noise_population_moments(residual[selected],mask[selected],np.ones(len(selected)))
        output["noise_shared"][group]=noise["shared"];output["noise_private"][group]=noise["private"]
        return None
    list(map(cohort,np.arange(groups)))
    output["component_reconstruction_error"]=np.asarray(float(np.max(abs(own.sum(-1)-own_observed/population_rms[:,None]))))
    output["foreground_shared_source"]=shared_source
    output["model_reconstruction_error"]=np.asarray(float(np.max(abs(residual+fg_clock+bg_clock-observed_pair))))
    output["restored_reference_FG_H1_energy"]=data["average"]@(np.sum(restored**2,axis=1)/count)
    return output


def fit_calibrated_reference_roles(bundle,parent,reference,qualified,training_run,independent_unit,
        reference_source_id=None,maxiter=900,device="cuda",case_callback=None):
    """New source measurements receive a newly fitted role center, never the old 62."""
    assert str(parent["source_emission_kind"])=="calibrated_shared_private_joint_gaussian"
    period=bundle["period"];scale=float(bundle["period_scale"]);axis=period/scale
    context={"qualified":qualified,"origin_run":bundle["run"],"origin_trace_id":bundle["trace_id"],
        "independent_unit":np.asarray(independent_unit),"view_weight":np.ones(len(reference)),
        "is_original":np.ones(len(reference),dtype=bool)}
    weights=source_reference_weights(bundle,context,training_run)
    identifiers=bundle["trace_id"] if reference_source_id is None else np.asarray(reference_source_id)
    code,names=pd.factorize(identifiers,sort=True)
    member=(np.arange(len(names))[:,None]==code[None]).astype(float)
    count=((weights["train"].astype(float)@member.T)>0.).sum(1)
    np.testing.assert_array_equal(count,parent["reference_source_count"])
    target=np.where(qualified,reference/scale,0.)
    number=bundle["membership"]@qualified.astype(float)
    group_reference=np.divide(bundle["membership"]@target,number,out=np.zeros(len(number)),where=number>0.)
    coherent=bundle["source_base"].copy();coherent[:,:,11]=bundle["source_coherent_power"]
    design=quality_design(_polynomial(coherent),bundle["mixing_quality"])
    foreground=foreground_reference_log_prior(bundle,parent,np.arange(len(training_run)))
    fitted=[]
    def case(index):
        import time
        started=time.time()
        print("CALIBRATED_REFERENCE_START",index,int(count[index]),len(bundle["trace_id"]),int(np.sum(parent["emission_state_weight"][index]>0.)),flush=True)
        role=source_reference_joint_measurement(bundle,parent,index,device)
        feature_s=time.time()-started
        print("CALIBRATED_REFERENCE_FIELD",index,"seconds",feature_s,flush=True)
        fixed=foreground[index:index+1,None]+np.einsum("npd,d->np",design,parent["core_coefficient"][index],optimize=True)[None]
        coefficient,status=_fit_period_role(bundle["source_base"],role[None],fixed,target,qualified,weights["train"][index:index+1],
            axis,maxiter,device,cohort=(bundle["membership"],bundle["average"],group_reference,training_run[index:index+1]),
            quality=bundle["mixing_quality"],interaction_units=weights["effective_training_units"][index:index+1],
            reference_count=count[index:index+1],training_weight=weights["training_weight"][index:index+1],
            cohort_weight=weights["group_weight"][index:index+1])
        result={"coefficient":coefficient[0],"success":status["success"],"iterations":status["iterations"],
            "gradient":status["gradient"],"feature_s":feature_s,"elapsed_s":time.time()-started}
        fitted.append(result)
        print("CALIBRATED_REFERENCE_ROLE_CASE",index,int(count[index]),result["success"],result["iterations"],result["gradient"],"feature_s",feature_s,"total_s",result["elapsed_s"],flush=True)
        if case_callback is not None:
            case_callback(index,assemble(fitted))
        return result
    def assemble(values):
        count_completed=len(values)
        result=dict(parent)
        result["role_coefficient"]=np.stack(list(map(lambda value:value["coefficient"],values)))
        result["role_feature_kind"]=np.asarray("calibrated_shared_private_joint_gaussian")
        result["role_center_kind"]=np.asarray("new_reference_fit_on_calibrated_joint_measurements")
        result["reference_role_success"]=np.asarray(list(map(lambda value:value["success"],values)))
        result["reference_role_iterations"]=np.asarray(list(map(lambda value:value["iterations"],values)))
        result["reference_role_gradient"]=np.asarray(list(map(lambda value:value["gradient"],values)))
        result["reference_role_feature_s"]=np.asarray(list(map(lambda value:value["feature_s"],values)))
        return result
    list(map(case,np.arange(len(training_run))))
    return assemble(fitted)



def calibrate_reference_source_emission(frame,components,context,background_reference,background_qualified,
        original_run_names,original_training_run,case_indices,period,source_balancing=False,
        weak_background_log_prior=None,source_hierarchy=False):
    """Reference-only source covariance distribution in the actual No operator.

    Each state is an allowed biological unit and physical view action. View
    labels identify calibration rows only; query evaluation uses No densities.
    Population matrices are pooled per measurement cohort before unit pooling,
    so a window's repeated leads do not become independent source references.
    """
    import time
    assert weak_background_log_prior is not None
    component=dict(components);context=dict(context)
    cases=np.asarray(case_indices,dtype=int);period=np.asarray(period,dtype=float)
    data=observations(frame);names=data["metadata"].index.to_numpy(dtype=str)
    order=pd.Index(context["trace_id"]).get_indexer(names);assert np.all(order>=0)
    row_keys=["target","qualified","origin_run","origin_trace_id","independent_unit","view_weight","is_original","view_id","reference_source_id"]
    ctx={key:context[key][order] for key in row_keys}
    group_origin=pd.Index(original_run_names).get_indexer(ctx["origin_run"])
    group_origin=np.rint(data["average"]@group_origin).astype(int)
    view_training=original_training_run[cases][:,group_origin]
    bundle={"trace_id":names,"run":data["metadata"].run.to_numpy(dtype=str),"run_index":data["run_index"],
        "run_names":data["runs"],"membership":data["membership"],"average":data["average"]}
    fg_weight=source_reference_weights(bundle,ctx,view_training)
    bqual=np.asarray(background_qualified,dtype=bool)[order]
    bperiod=np.asarray(background_reference,dtype=float)[order]
    original_source=pd.Index(component["trace_id"]).get_indexer(ctx["origin_trace_id"]);assert np.all(original_source>=0)
    background_ids=np.char.add("BG:",component["reference_source_id"][original_source].astype(str))
    bg_ctx=dict(ctx,qualified=bqual,reference_source_id=background_ids)
    bg_weight=source_reference_weights(bundle,bg_ctx,view_training)
    row_units=ctx["independent_unit"];row_views=ctx["view_id"]
    group_units=pd.Series(row_units,index=bundle["run"]).groupby(level=0).first().loc[data["runs"]].to_numpy(dtype=str)
    group_views=pd.Series(row_views,index=bundle["run"]).groupby(level=0).first().loc[data["runs"]].to_numpy(dtype=str)
    shared=np.zeros((len(data["runs"]),2,4,4));private=np.zeros_like(shared);finite=np.zeros_like(shared)
    noise_shared=np.zeros((len(data["runs"]),2,2));noise_private=np.zeros_like(noise_shared)
    source_valid=np.zeros((len(data["runs"]),2),dtype=bool);noise_valid=np.zeros(len(data["runs"]),dtype=bool)
    foreground_shared_source=np.zeros(len(data["runs"]),dtype=bool)
    source_rows=pd.Index(component["trace_id"])
    def one_view(name):
        started=time.time();rows=np.flatnonzero(row_views==name)
        ids=names[rows];view_frame=frame.loc[frame.trace_id.isin(ids)]
        vd=observations(view_frame);positions=pd.Index(names).get_indexer(vd["metadata"].index)
        origin=source_rows.get_indexer(ctx["origin_trace_id"][positions]);assert np.all(origin>=0)
        vi=int(np.flatnonzero(component["candidate_view_id"]==name)[0])
        fg=component["foreground_source_waveform"][component["view_foreground_index"][vi],component["source_index"][origin]]
        fg=fg*component["foreground_loading"][origin,None]*component["foreground_gain"][vi,origin,None]
        bg=component["background"][origin]*component["background_gain"][vi,origin,None]
        bg+=component["common_background"][origin]*component["common_background_gain"][vi,origin,None]
        remainder=vd["signal"]-fg-bg
        values=np.stack([fg,bg,remainder],axis=-1)
        result=reference_source_operator_statistics(view_frame,values,component["foreground_period"][vi,origin],
            bperiod[positions],ctx["qualified"][positions]&component["foreground_period_qualified"][vi,origin],
            bqual[positions],period,source_balancing=source_balancing,
            foreground_source_id=ctx["reference_source_id"][positions],background_source_id=background_ids[positions],source_hierarchy=source_hierarchy)
        dest=pd.Index(data["runs"]).get_indexer(result["run_names"]);assert np.all(dest>=0)
        shared[dest]=result["source_shared"];private[dest]=result["source_private"];finite[dest]=result["source_finite"]
        noise_shared[dest]=result["noise_shared"];noise_private[dest]=result["noise_private"]
        source_valid[dest]=result["source_valid"];noise_valid[dest]=result["noise_valid"]
        foreground_shared_source[dest]=result["foreground_shared_source"]
        print("SOURCE_OPERATOR_CALIBRATION_VIEW",str(name),len(ids),"balanced",source_balancing,
            "seconds",time.time()-started,"reconstruction",float(result["component_reconstruction_error"]),flush=True)
        return None
    list(map(one_view,np.unique(row_views)))
    pairs,owner=np.unique(np.column_stack([group_units,group_views]),axis=0,return_inverse=True)
    count=len(pairs);fit_cases=len(cases)
    output={"emission_shared_covariance":np.zeros((fit_cases,count,2,4,4)),
        "emission_private_covariance":np.zeros((fit_cases,count,2,4,4)),
        "emission_finite_covariance":np.zeros((fit_cases,count,2,4,4)),
        "emission_noise_shared_covariance":np.zeros((fit_cases,count,2,2)),
        "emission_noise_private_covariance":np.zeros((fit_cases,count,2,2)),
        "emission_state_weight":np.zeros((fit_cases,count))}
    if source_hierarchy:
        output["emission_foreground_shared_source"]=np.zeros((fit_cases,count),dtype=bool)
    def state(index):
        groups=np.flatnonzero(owner==index)
        shared_kind=foreground_shared_source[groups]
        assert np.all(shared_kind==shared_kind[0])
        if source_hierarchy:
            output["emission_foreground_shared_source"][:,index]=shared_kind[0]
        def one_case(fold):
            raw_weight=fg_weight["group_weight"][fold,groups]
            if raw_weight.sum()==0.:
                return None
            for source in range(2):
                w=(raw_weight if source==0 else bg_weight["group_weight"][fold,groups])*source_valid[groups,source]
                assert w.sum()>0.
                w=w/w.sum()
                output["emission_shared_covariance"][fold,index,source]=np.einsum("g,gij->ij",w,shared[groups,source])
                output["emission_private_covariance"][fold,index,source]=np.einsum("g,gij->ij",w,private[groups,source])
                output["emission_finite_covariance"][fold,index,source]=np.einsum("g,gij->ij",w,finite[groups,source])
            w=raw_weight*noise_valid[groups];assert w.sum()>0.;w=w/w.sum()
            output["emission_noise_shared_covariance"][fold,index]=np.einsum("g,gij->ij",w,noise_shared[groups])
            output["emission_noise_private_covariance"][fold,index]=np.einsum("g,gij->ij",w,noise_private[groups])
            output["emission_state_weight"][fold,index]=raw_weight.sum()
            return None
        list(map(one_case,np.arange(fit_cases)))
        return None
    list(map(state,np.arange(count)))
    output["emission_state_weight"]/=output["emission_state_weight"].sum(1)[:,None]
    original_train=fg_weight["train"]&ctx["is_original"][None]
    code,source_names=pd.factorize(ctx["reference_source_id"],sort=True)
    member=(np.arange(len(source_names))[:,None]==code[None]).astype(float)
    bg_code,bg_names=pd.factorize(background_ids,sort=True)
    bg_member=(np.arange(len(bg_names))[:,None]==bg_code[None]).astype(float)
    bg_original=bg_weight["train"]&ctx["is_original"][None]
    output.update({"reference_source_count":((original_train.astype(float)@member.T)>0.).sum(1),
        "background_reference_source_count":((bg_original.astype(float)@bg_member.T)>0.).sum(1),
        "case_index":cases,"state_unit":pairs[:,0],"state_view":pairs[:,1],
        "source_emission_kind":np.asarray("calibrated_shared_private_joint_gaussian"),
        "shared_clock_probability":np.asarray(.5),"period":period,"period_scale":np.asarray(np.sqrt(period.min()*period.max())),
        "emission_background_log_prior":np.asarray(weak_background_log_prior)[cases],
        "emission_covariance_coordinate":np.asarray("frozen own No operator; retained adjusted-raw peer; inverse endpoint RMS congruence"),
        "source_information":np.asarray(str(component["nuisance_reference_kind"]))})
    if source_hierarchy:
        output["source_hierarchy_kind"]=np.asarray("own_source_peer_foreground_nuisance")
        output["peer_foreground_nuisance_definition"]=np.asarray("complete_other_source_FG_retained_in_reference_residual; empirical_Pi_independent_E2_shared_private_moments")
    return output



def _support_component_design(values,bundle):
    """Linear observation-grid evaluation at immutable candidate centres."""
    period=bundle["period"];center=bundle["source_support_center"]
    index=np.clip(np.searchsorted(period,center,side="right")-1,0,len(period)-2)
    fraction=(center-period[index])/(period[index+1]-period[index])
    rows=np.arange(len(center))[:,None]
    return values[rows,index]*(1.-fraction[:,:,None])+values[rows,index+1]*fraction[:,:,None]


def _support_prior_score(log_prior,bundle):
    values=np.broadcast_to(np.asarray(log_prior)[None,:,None],(len(bundle["trace_id"]),len(bundle["period"]),1))
    return _support_component_design(values,bundle)[:,:,0]


def observed_frequency_support(bundle):
    """Return only support fields for all No H1/FWL peaks and profile widths.

    Kernels are truncated at neighbouring valleys and integrated into linear
    grid hats. Their first moment retains a sub-grid centre at arbitrarily
    high SNR; source scores can select kernels but cannot move their centres.
    No reference labels, source components, or training cases are read here.
    """
    from scipy.special import ndtr
    result={}
    period=np.asarray(bundle["period"],dtype=float);scale=float(bundle["period_scale"])
    axis=period/scale;step=float(axis[1]-axis[0]);n=len(bundle["trace_id"]);p=len(period)
    h1=np.asarray(bundle["source_base"][:,:,0],dtype=float)
    fwl=np.asarray(bundle["source_coherent_power"],dtype=float)
    def peaks(value):
        left=np.pad(value[:,:-1],((0,0),(1,0)),constant_values=-np.inf)
        right=np.pad(value[:,1:],((0,0),(0,1)),constant_values=-np.inf)
        return (value>=left)&(value>=right)
    hpeak=peaks(h1);fpeak=peaks(fwl);peak=hpeak|fpeak
    count=peak.sum(1);dimension=int(count.max())
    rows,column=np.nonzero(peak);slot=np.cumsum(peak,axis=1)[rows,column]-1
    source=np.where(hpeak[rows,column],0,1)
    curves=np.stack([h1,fwl],axis=1)
    current=curves[rows,source,column]
    before=curves[rows,source,np.maximum(column-1,0)]
    after=curves[rows,source,np.minimum(column+1,p-1)]
    curvature=(2.*current-before-after)/step**2
    slope=(after-before)/(2.*step)
    interior=(column>0)&(column<p-1)
    displacement=np.divide(slope,curvature,out=np.zeros_like(slope),where=interior&(curvature>0.))
    displacement=np.clip(displacement,-step/2.,step/2.)
    centre=axis[column]+displacement
    height=current+slope*displacement-.5*curvature*displacement**2
    # The H1 unknown-variance profile uses RSS/count=1-explained energy.
    # Cross-half FWL is a signed power estimate, so its conservative width
    # uses the observed No unexplained energy rather than treating it as RSS.
    unexplained=np.where(source==0,1.-height,1.-h1.max(1)[rows])
    unexplained=np.maximum(unexplained,np.finfo(float).eps)
    degrees=np.maximum(np.asarray(bundle["count"],dtype=float)[rows]-4.,1.)
    variance=np.divide(2.*unexplained,degrees*curvature,
        out=np.full_like(curvature,np.inf),where=curvature>0.)
    width=np.sqrt(variance)
    all_index=np.arange(p)[None,None]
    left_value=np.pad(curves[:,:,:-1],((0,0),(0,0),(1,0)),constant_values=np.inf)
    right_value=np.pad(curves[:,:,1:],((0,0),(0,0),(0,1)),constant_values=np.inf)
    valley=(curves<=left_value)&(curves<=right_value)
    lower_index=np.maximum.accumulate(np.where(valley,all_index,0),axis=-1)
    upper_index=np.minimum.accumulate(np.where(valley,all_index,p-1)[:,:,::-1],axis=-1)[:,:,::-1]
    low=axis[lower_index[rows,source,column]];high=axis[upper_index[rows,source,column]]
    # A flat observed profile is one uniform bounded component at each
    # plateau coordinate. Zero-height basins span the configured interval.
    degenerate=high<=low
    low=np.where(degenerate,axis[0],low);high=np.where(degenerate,axis[-1],high)
    a=np.maximum(axis[:-1][None],low[:,None]);b=np.minimum(axis[1:][None],high[:,None])
    active=b>a
    uniform=(~np.isfinite(width))|(width>(high-low)/np.sqrt(np.finfo(float).eps))
    safe_width=np.where(uniform,1.,width)
    za=(a-centre[:,None])/safe_width[:,None];zb=(b-centre[:,None])/safe_width[:,None]
    probability=np.maximum(ndtr(zb)-ndtr(za),0.)
    first=centre[:,None]*probability+safe_width[:,None]*(np.exp(-.5*za**2)-np.exp(-.5*zb**2))/np.sqrt(2.*np.pi)
    probability=np.where(uniform[:,None],b-a,probability)*active
    first=np.where(uniform[:,None],(b*b-a*a)/2.,first)*active
    left_mass=np.maximum((axis[1:][None]*probability-first)/step,0.)
    right_mass=np.maximum((first-axis[:-1][None]*probability)/step,0.)
    mass=np.pad(left_mass,((0,0),(0,1)))+np.pad(right_mass,((0,0),(1,0)))
    total=mass.sum(1)
    # If a machine-width peak has no resolved Gaussian integral, retain
    # its deterministic barycentric point mass without changing its centre.
    missing=total<=np.finfo(float).tiny
    point_index=np.clip(np.searchsorted(axis,centre,side="right")-1,0,p-2)
    fraction=(centre-axis[point_index])/step
    point=np.zeros_like(mass)
    point[np.arange(len(point)),point_index]=1.-fraction
    point[np.arange(len(point)),point_index+1]=fraction
    mass=np.where(missing[:,None],point,mass/np.maximum(total,np.finfo(float).tiny)[:,None])
    kernel=np.zeros((n,dimension,p));centres=np.broadcast_to(period[0],(n,dimension)).copy()
    widths=np.full((n,dimension),np.inf);valid=np.zeros((n,dimension),dtype=bool)
    kernel[rows,slot]=mass;centres[rows,slot]=centre*scale;widths[rows,slot]=width*scale;valid[rows,slot]=True
    result.update({"source_support_kernel":kernel,"source_support_center":centres,
        "source_support_width":widths,"source_support_valid":valid,
        "source_support_kind":np.asarray("observed_H1_FWL_profile_components"),
        "source_support_definition":np.asarray("all_No_H1_FWL_local_maxima; quadratic_subgrid_centre; observed_curvature_width; valley_basins; mean_preserving_grid_hats")})
    return result


def _support_bundle_pair(raw,balanced,kind):
    if kind!="observed_H1_FWL_profile_components":
        return raw,balanced
    if "source_support_kernel" not in raw:
        raw=dict(raw,**observed_frequency_support(raw))
    keys=["source_support_kernel","source_support_center","source_support_width","source_support_valid",
        "source_support_kind","source_support_definition"]
    return raw,dict(balanced,**{key:raw[key] for key in keys})


def _support_torch_log_density(score,kernel,valid):
    score=score.masked_fill(~valid,-torch.inf)
    weights=torch.log_softmax(score,dim=-1)
    log=torch.logsumexp(weights[:,:,None]+kernel,dim=1)
    return log-torch.logsumexp(log,dim=-1,keepdim=True)


def fit_observed_support_reference(bundle,parent,reference,qualified,training_run,independent_unit,
        reference_source_id=None,maxiter=900,device="cuda",case_callback=None):
    """Refit the complete source-weight score in the fixed observed support.

    The old grid-score coefficients are neither an initialization nor a
    regularization centre. Source/unit weights and proper/cohort losses retain
    their original definition; only the observable density representation changes.
    """
    bundle=dict(bundle,**observed_frequency_support(bundle)) if "source_support_kernel" not in bundle else bundle
    period=bundle["period"];scale=float(bundle["period_scale"]);axis=period/scale
    identifiers=bundle["trace_id"] if reference_source_id is None else np.asarray(reference_source_id)
    context={"qualified":qualified,"origin_run":bundle["run"],"origin_trace_id":bundle["trace_id"],
        "independent_unit":np.asarray(independent_unit),"view_weight":np.ones(len(reference)),
        "is_original":np.ones(len(reference),dtype=bool),"reference_source_id":identifiers}
    weights=source_reference_weights(bundle,context,training_run)
    code,names=pd.factorize(identifiers,sort=True)
    member=(np.arange(len(names))[:,None]==code[None]).astype(float)
    number=((weights["train"].astype(float)@member.T)>0.).sum(1)
    np.testing.assert_array_equal(number,parent["reference_source_count"])
    target=np.where(qualified,reference/scale,0.)
    teacher=period_reference_mass(axis,target,qualified)
    reference_number=bundle["membership"]@qualified.astype(float)
    reference_mean=np.divide(bundle["membership"]@target,reference_number,
        out=np.zeros(len(reference_number)),where=reference_number>0.)
    weight=weights["training_weight"]
    sd=np.sqrt(np.maximum(weight@(target**2)-(weight@target)**2,(axis[1]-axis[0])**2/12.))
    foreground=foreground_reference_log_prior(bundle,parent,np.arange(len(training_run)))
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float64,device=device)
    kernel=tensor(np.log(np.maximum(bundle["source_support_kernel"],np.finfo(float).tiny)))
    valid=torch.as_tensor(bundle["source_support_valid"],dtype=torch.bool,device=device)
    ty=tensor(teacher);average=tensor(bundle["average"]);taxis=tensor(axis)
    completed=[]
    def assemble(values):
        result=dict(parent)
        coefficient=np.stack([value["coefficient"] for value in values])
        nc=parent["core_coefficient"].shape[1]//2;nr=parent["role_coefficient"].shape[1]//2
        if bool(parent.get("role_disabled",False)):
            result["core_coefficient"]=coefficient
            result["role_coefficient"]=np.zeros((len(values),2*nr))
        else:
            result["core_coefficient"]=np.concatenate([coefficient[:,:nc],coefficient[:,nc+nr:2*nc+nr]],axis=-1)
            result["role_coefficient"]=np.concatenate([coefficient[:,nc:nc+nr],coefficient[:,2*nc+nr:]],axis=-1)
        result.update({"source_support_kind":np.asarray("observed_H1_FWL_profile_components"),
            "support_reference_center":np.asarray("observed_support_reference_proper_mean"),
            "reference_support_success":np.asarray([value["success"] for value in values]),
            "reference_support_iterations":np.asarray([value["iterations"] for value in values]),
            "reference_support_gradient":np.asarray([value["gradient"] for value in values]),
            "reference_support_initial_loss":np.asarray([value["initial_loss"] for value in values]),
            "reference_support_final_loss":np.asarray([value["final_loss"] for value in values]),
            "reference_support_elapsed_s":np.asarray([value["elapsed_s"] for value in values])})
        if "source_hierarchy_kind" in parent:
            result["role_hierarchy_kind"]=parent["source_hierarchy_kind"]
        return result
    def case(index):
        started=time.time()
        grid=_latent_representation_features(bundle,parent,index,device)
        native=grid[None];nw=weight[index:index+1];units=weights["effective_training_units"][index:index+1]
        xs,transform=_role_conditioning(native,nw,number[index:index+1])
        prior=conditional_prior_transform(native,bundle["mixing_quality"],nw,xs,transform,number[index:index+1],units)
        design=tensor(_support_component_design(grid,bundle));ts=tensor(xs[0]);tt=tensor(transform[0]);tp=tensor(prior[0])
        fixed=tensor(_support_prior_score(foreground[index],bundle))
        tw=tensor(weight[index]);tg=tensor(weights["group_weight"][index]);tr=tensor(reference_mean)
        def objective(parameter,readout=False):
            theta=tensor(parameter).requires_grad_(not readout)
            coefficient=(tt@theta)/ts
            log=_support_torch_log_density(fixed+design@coefficient,kernel,valid)
            mass=log.exp();mean=average@(mass@taxis)
            ce=-(log*ty*tw[:,None]).sum()
            crps=((mass.cumsum(-1)[:,:-1]-ty.cumsum(-1)[:,:-1]).square()*tw[:,None]*tensor(np.diff(axis))[None]).sum()/sd[index]
            moment=.5*((mean-tr).square()*tg).sum()/sd[index]**2
            loss=ce+crps+moment+.5*(tp@theta).square().sum()
            if readout:
                return coefficient.detach().cpu().numpy()
            return float(loss.detach().cpu()),torch.autograd.grad(loss,theta)[0].detach().cpu().numpy()
        initial=np.zeros(grid.shape[-1]);initial_loss=objective(initial)[0]
        print("SUPPORT_REFERENCE_START",index,int(number[index]),int(units[0]),initial_loss,flush=True)
        fit=minimize(objective,initial,jac=True,method="L-BFGS-B",
            options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-5,"maxcor":25,"maxls":40})
        iterations=fit.nit
        while not fit.success and fit.status==1:
            fit=minimize(objective,fit.x,jac=True,method="L-BFGS-B",
                options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-5,"maxcor":25,"maxls":40})
            iterations+=fit.nit
        value={"coefficient":objective(fit.x,True),"success":bool(fit.success),"iterations":iterations,
            "gradient":float(np.max(abs(fit.jac))),"initial_loss":initial_loss,"final_loss":float(fit.fun),
            "elapsed_s":time.time()-started}
        completed.append(value)
        print("SUPPORT_REFERENCE_CASE",index,value["success"],iterations,value["final_loss"],value["gradient"],value["elapsed_s"],flush=True)
        if case_callback is not None:
            case_callback(index,assemble(completed))
        return value
    list(map(case,np.arange(len(training_run))))
    return assemble(completed)


def raw_no_anchor_density(frame,period,anchor_coefficient,period_scale=None):
    """Frozen supervised raw-No anchor scores on actual time/mask geometry.

    The 13 original No/mean-unit-peer statistics and their 91 quadratic terms
    precede common-background removal or amplitude balancing. No period prior,
    target labels, PAV targets, source-role fields or point reconstruction enter
    this base density. The original anchor least-squares fits have zero ridge.
    """
    data=observations(frame);period=np.asarray(period,dtype=float)
    scale=float(np.sqrt(period[0]*period[-1])) if period_scale is None else float(period_scale)
    base=profile_features(data["signal"],data,period,scale,ridge_fraction=0.)[0]
    score=np.einsum("npd,fd->fnp",_polynomial(base),np.atleast_2d(anchor_coefficient),optimize=True)
    log=score-logsumexp(score,axis=-1,keepdims=True)
    return {"log_probability":log,"anchor_score":score,"period":period,"period_scale":np.asarray(scale),
        "trace_id":data["metadata"].index.to_numpy(dtype=str),"run_index":data["run_index"],
        "run_names":data["runs"],"membership":data["membership"],"average":data["average"],
        "density_base":np.asarray("raw_No_supervised_anchor_full_period_score")}


def reference_source_observations(frame,period,representation_options,fundamental_support=True,device="cpu"):
    """One raw-No source representation for every actual mask and duration."""
    from source_assignment_core import legacy_source_role_observations
    data=observations(frame);period=np.asarray(period,dtype=float);scale=float(np.sqrt(period[0]*period[-1]))
    options=dict(representation_options,device=device)
    peer_original=bool(options.pop("peer_original",True))
    source=legacy_source_role_observations(data["time"],data["mask"],data["signal"],
        run_index=data["run_index"],period=period,prepare_source=True,
        representation_options=options,fundamental_support=fundamental_support,reference_peer_original=peer_original)
    identity=profile_features(data["signal"],data,period,scale,ridge_fraction=0.)[0]
    source.update({"trace_id":data["metadata"].index.to_numpy(dtype=str),
        "run":data["metadata"].run.to_numpy(dtype=str),"condition":data["metadata"].condition.to_numpy(dtype=str),
        "run_names":data["runs"],"identity_base_features":identity,
        "source_base":source["base_features"],"source_rhs":source["rhs"],"source_peer":source["physical_peer"],
        "source_coherent_power":source["coherent_power"],"original_coherent_power":source["coherent_power"],
        "original_unit":source["source_unit"],"original_peer":source["physical_peer"],"original_rhs":source["rhs"],
        "harmonic_design":harmonic_profile(source["source_unit"],data,period,scale,harmonics=1)["design"],
        "mixing_quality":np.zeros(len(data["count"]))})
    return source


def predict_reference_source_w2(bundle,model,cases=None,device="cpu",include_diagnostics=False):
    """One transported raw-anchor distribution supplies every final readout."""
    from source_assignment_core import legacy_source_role_readout
    cases=np.arange(len(model["core_coefficient"])) if cases is None else np.atleast_1d(cases)
    fields=["background_coefficient","role_covariance","role_coefficient","calibration_prediction",
        "calibration_mean","calibration_experiments"]
    scoring={key:model[key][cases] for key in fields}
    scoring["calibration_prediction"]=scoring["calibration_prediction"]*float(bundle["period_scale"])
    scoring["calibration_mean"]=scoring["calibration_mean"]*float(bundle["period_scale"])
    scoring.update({"coefficient":model["core_coefficient"][cases],
        "core_coherent_peer":np.asarray(model.get("core_coherent_peer",np.ones(len(model["core_coefficient"]),dtype=bool)))[cases],
        "core_partial_peer":np.asarray(model.get("core_partial_peer",np.zeros(len(model["core_coefficient"]),dtype=bool)))[cases],
        "mean_identity_tails":np.asarray(model.get("mean_identity_tails",np.ones(len(model["core_coefficient"]),dtype=bool)))[cases]})
    if str(model.get("conditioning_coordinate",""))=="source_information_coordinate":
        information=observed_source_information(bundle)
        def conditioned_case(position):
            case=int(cases[position])
            low=float(model["source_information_min"][case]);high=float(model["source_information_max"][case])
            q=np.divide(information-low,high-low,out=np.zeros_like(information),where=high>low).clip(0.,1.)
            background=softmax(np.einsum("npd,d->np",quality_design(bundle["raw_features"],q),model["background_coefficient"][case],optimize=True),axis=-1)[None]
            local=dict(bundle,source_rhs=bundle["rhs"],source_information_coordinate=q)
            role,_=source_role_evidence(local,model["role_covariance"][case:case+1],background,
                rhs=bundle["rhs"],correlated=True,quality=q)
            score=np.einsum("npd,d->np",quality_design(bundle["core_features"],q),model["core_coefficient"][case],optimize=True)
            score+=_role_score(role,bundle["base_features"],model["role_coefficient"][case:case+1],q)[0]
            probability=softmax(score,axis=-1);point=probability@bundle["period"];mean=point@bundle["average"].T
            calibrated=interpolate_mean(mean[None],scoring["calibration_prediction"][position:position+1],
                scoring["calibration_mean"][position:position+1],scoring["calibration_experiments"][position:position+1],True)[0]
            return {"posterior":probability,"source_period":point,"source_mean":mean,"group_mean":calibrated}
        values=list(map(conditioned_case,np.arange(len(cases))))
        source={key:np.stack([value[key] for value in values]) for key in values[0]}
    elif str(model.get("source_features","assignment"))=="density":
        background=np.einsum("npd,fd->fnp",bundle["raw_features"],scoring["background_coefficient"],optimize=True)
        log_background=background-logsumexp(background,axis=-1,keepdims=True)
        raw_core=np.einsum("npd,fd->fnp",bundle["raw_features"],scoring["coefficient"],optimize=True)
        coherent_core=np.einsum("npd,fd->fnp",bundle["core_features"],scoring["coefficient"],optimize=True)
        fixed=np.where(scoring["core_coherent_peer"][:,None,None],coherent_core,raw_core)
        evidence=np.stack([fixed-logsumexp(fixed,axis=-1,keepdims=True),log_background],axis=-1)
        probability=softmax(fixed+_role_score(evidence,bundle["base_features"],scoring["role_coefficient"]),axis=-1)
        point=probability@bundle["period"];mean=point@bundle["average"].T
        calibrated=interpolate_mean(mean,scoring["calibration_prediction"],scoring["calibration_mean"],
            scoring["calibration_experiments"],scoring["mean_identity_tails"])
        source={"posterior":probability,"source_period":point,"source_mean":mean,"group_mean":calibrated}
    else:
        source=legacy_source_role_readout(bundle,scoring,device=device)
    score=np.einsum("npd,fd->fnp",_polynomial(bundle["identity_base_features"]),model["anchor_coefficient"][cases],optimize=True)
    log=score-logsumexp(score,axis=-1,keepdims=True)
    source["group_mean"]=np.clip(source["group_mean"],bundle["period"][0],bundle["period"][-1])
    result=calibrate_source_density(log,bundle["period"],bundle["run_index"],source["group_mean"],
        calibration_kind="wasserstein_affine",within_scale=model["within_scale"][cases,None])
    result.update({"original_period":softmax(log,axis=-1)@bundle["period"],"cohort_source_period":source["source_period"],"cohort_source_mean":source["source_mean"],
        "calibrated_mean_target":source["group_mean"]})
    if include_diagnostics:
        result.update({"source_assignment_probability":source["posterior"],
            "raw_individual_probability":softmax(log,axis=-1),"raw_individual_log_probability":log})
        prepared=bundle.get("source_representation")
        if isinstance(prepared,dict):
            result["source_representation_diagnostics"]={key:value for key,value in prepared.items() if isinstance(value,np.ndarray)}
    return result


def predict_reference_calibration_mixture(frame,model,*,cases=None,device="cpu",include_diagnostics=False,source_bundles=None):
    """Mix complete reference-calibrated densities by actual sampling information.

    Each reference law retains its explicit representation/source-role flags.
    The mixture has one period distribution for source, point, mean and deviation.
    """
    policy=str(model.get("ablation_policy",""))
    cases=np.arange(len(model["case_index"])) if cases is None else np.atleast_1d(cases)
    options=model["calibration_representation_options_json"]
    parameters=["core_coefficient","anchor_coefficient","background_coefficient","role_covariance","role_coefficient",
        "calibration_prediction","calibration_mean","calibration_experiments","within_scale",
        "core_coherent_peer","core_partial_peer","mean_identity_tails"]
    values=[];bundles=[]
    for expert in range(len(options)):
        bundle=reference_source_observations(frame,model["period"],json.loads(str(options[expert])),
            fundamental_support=bool(model["calibration_fundamental_support"][expert]),device=device) if source_bundles is None else source_bundles[expert]
        parent={key:model["calibration_"+key][:,expert] for key in parameters}
        parent["source_features"]=str(model.get("calibration_source_features",np.full(len(options),"assignment"))[expert])
        value=predict_reference_source_w2(bundle,parent,cases=cases,device=device,include_diagnostics=include_diagnostics or policy=="no_W2")
        values.append(value);bundles.append(bundle)
    bundle=bundles[0]
    for current in bundles[1:]:
        np.testing.assert_array_equal(current["trace_id"],bundle["trace_id"])
        np.testing.assert_array_equal(current["mask"],bundle["mask"])
    information=observed_source_information(bundle)
    knots=np.asarray(model["reference_information_knots"])[cases]
    assert np.all(np.diff(knots,axis=1)>0.)
    def weights_at(row):
        return np.stack([np.interp(information,row,np.eye(len(row))[expert]) for expert in range(len(row))],axis=0)
    weight=np.stack(list(map(weights_at,knots)))
    if policy=="base_only":
        weight=np.zeros_like(weight);weight[:,0]=1.
    elif policy=="refined_only":
        weight=np.zeros_like(weight);weight[:,-1]=1.
    elif policy=="equal_reference_weights":
        weight=np.full_like(weight,1./len(options))
    component=np.stack([value["raw_individual_probability"] if policy=="no_W2" else value["posterior"] for value in values],axis=1)
    probability=np.einsum("frn,frnp->fnp",weight,component,optimize=True)
    estimate=probability@model["period"];mean=estimate@bundle["average"].T
    result={"posterior":probability,"source_period":estimate,"estimate":estimate,"group_mean":mean,
        "deviation":estimate-mean[:,bundle["run_index"]],
        "cohort_probability":np.einsum("gn,fnp->fgp",bundle["average"],probability,optimize=True),
        "source_information":information,"reference_calibration_weight":weight}
    if include_diagnostics:
        result.update({"reference_calibration_probability":component,
            "reference_source_assignment_probability":np.stack([value["source_assignment_probability"] for value in values],axis=1),
            "reference_raw_individual_probability":np.stack([value["raw_individual_probability"] for value in values],axis=1),
            "reference_calibration_density_shift":np.stack([value["cohort_density_shift"] for value in values],axis=1),
            "reference_calibration_density_scale":np.stack([value["cohort_density_scale"] for value in values],axis=1),
            "reference_cohort_source_mean":np.stack([value["cohort_source_mean"] for value in values],axis=1),
            "reference_calibration_mean":(component@model["period"])@bundle["average"].T,
            "source_assignment_probability":np.einsum("frn,frnp->fnp",weight,np.stack([value["source_assignment_probability"] for value in values],axis=1),optimize=True),
            "raw_individual_probability":np.einsum("frn,frnp->fnp",weight,np.stack([value["raw_individual_probability"] for value in values],axis=1),optimize=True),
            "calibrated_mean_target":mean})
    return result


def fit_reference_calibration_density(frame,reference,qualified,training_run,independent_unit,period,*,
        reference_source_id=None,background_reference=None,calibration_recipe=None,case_indices=None,
        maxiter=900,device="cpu",time_unit="h"):
    """Fit data-supported reference calibrations; no acquisition-size routing.

    Labels, qualification, unit and source IDs follow observations(frame) rows.
    Each source contributes once to reference risk. Information only defines
    calibration support and density mixing; it never enters a period-score head.
    """
    data=observations(frame);period=np.asarray(period,dtype=float);scale=float(np.sqrt(period[0]*period[-1]))
    reference=np.asarray(reference,dtype=float);qualified=np.asarray(qualified,dtype=bool)
    training_run=np.asarray(training_run,dtype=bool);independent_unit=np.asarray(independent_unit,dtype=str)
    source_id=data["metadata"].index.to_numpy(dtype=str) if reference_source_id is None else np.asarray(reference_source_id,dtype=str)
    source_code,source_names=pd.factorize(source_id,sort=True)
    source_member=(np.arange(len(source_names))[:,None]==source_code[None]).astype(float)
    raw_base=profile_features(data["signal"],data,period,scale,ridge_fraction=0.)[0]
    geometry=_role_geometry(data["signal"],data,period,scale,normalize_reference_peer=True)
    information=observed_source_information({"period":period,"joint_gram":geometry["joint"],"layout":geometry["layout"]})
    options={"common_baseline":False,"common_baseline_strength":.05,"common_baseline_period":scale,
        "share_order":True,"posterior_energy":False,"frequency_marginal":False,"frequency_correlated":False,
        "colored_order":False,"cohort_center":False,"peer_original":False}
    if calibration_recipe is None:
        calibration_recipe=[{"name":"general_reference","representation_options":options,"fundamental_support":False,
            "core_coherent_peer":False,"core_partial_peer":False,"mean_identity_tails":False,
            "training_stage":"raw_reference","supervision":"all"}]
        eligible_information=np.where(training_run[:,data["run_index"]]&qualified[None],information[None],np.nan)
        context_has_information_range=np.nanmax(eligible_information,axis=1)>np.nanmin(eligible_information,axis=1)
        if np.all(context_has_information_range):
            calibration_recipe.append({"name":"refined_reference","representation_options":dict(options,common_baseline=True,
                share_order=False,posterior_energy=True,frequency_marginal=True,frequency_correlated=True,
                colored_order=True,peer_original=True),"fundamental_support":True,"core_coherent_peer":True,
                "core_partial_peer":True,"mean_identity_tails":True,"training_stage":"reference_coherent_amplitude",
                "supervision":"information_coordinate"})
    recipe=list(calibration_recipe);assert 1<=len(recipe)<=2
    assert recipe[0]["training_stage"]=="raw_reference"
    bundles=[reference_source_observations(frame,period,value["representation_options"],
        fundamental_support=bool(value["fundamental_support"]),device=device) for value in recipe]
    raw_bundle=dict(bundles[0],base_features=raw_base,identity_base_features=raw_base,source_base=raw_base,
        original_unit=geometry["unit"],original_peer=geometry["peer"],source_unit=geometry["unit"],
        source_peer=geometry["peer"],physical_peer=geometry["peer"],original_rhs=geometry["rhs"],
        source_rhs=geometry["rhs"],rhs=geometry["rhs"],harmonic_design=geometry["design"],
        joint_gram=geometry["joint"],layout=geometry["layout"])
    raw_features=_polynomial(raw_base)
    coherent_base=raw_base.copy()
    coherent_base[:,:,11]=generic_coherent_fwl_power(data["time"],data["mask"],geometry["unit"],geometry["peer"],period,half_gram=False)
    coherent_features=_polynomial(coherent_base)
    background_label=period[raw_base[:,:,4].argmax(1)] if background_reference is None else np.asarray(background_reference,dtype=float)
    assert np.isfinite(background_label).all()
    bg_index=abs(period[None]-background_label[:,None]).argmin(1)
    left,right=np.triu_indices(13);bg_columns=np.flatnonzero(np.r_[np.isin(np.arange(13),[4,5,6,7,8]),
        np.isin(left,[4,5,6,7,8])&np.isin(right,[4,5,6,7,8])])
    member=data["membership"];average=data["average"];count=member@qualified.astype(float)
    target=np.where(qualified,reference,0.);group_reference=(member@(target/scale))/np.maximum(count,1.)
    raw_rivals=np.stack([raw_base[:,:,0].argmax(1),raw_base[:,:,4].argmax(1)],axis=-1)
    cases=np.arange(len(training_run)) if case_indices is None else np.asarray(case_indices,dtype=int)
    records=[];masks=[];weights=[];knots=[];source_counts=[];unit_counts=[]
    for case in cases:
        start=time.time();allowed=training_run[case,data["run_index"]]&qualified
        low=float(information[allowed].min());high=float(information[allowed].max())
        coordinate=np.divide(information-low,high-low,out=np.zeros_like(information),where=high>low).clip(0.,1.)
        knots.append(np.asarray([low] if len(recipe)==1 else [low,high]))
        local=[];local_masks=[];local_weights=[];local_n=[];local_g=[]
        for expert,definition in enumerate(recipe):
            strength=np.ones(len(reference)) if definition["supervision"]=="all" else coordinate
            active=allowed&(strength>0.)
            source_number=source_member@active.astype(float)
            n=np.asarray([int((source_number>0.).sum())]);g=int(np.unique(independent_unit[active]).size)
            assert n[0]>0, "Each requested calibration needs allowed foreground references in this fold"
            weight=np.divide(active*strength,np.maximum(source_number[source_code],1.));weight/=weight.sum()
            train=active[None];w=weight[None];zero=np.zeros((1,len(period)))
            anchor,core,status=_fit_core(raw_base,target,qualified,train,period,maxiter,zero,
                reference_count=n,training_weight=w,sequential_reference=True)
            background,bg_status=_fit_score(raw_features,background_label,qualified,train,period,columns=bg_columns,
                maxiter=maxiter,effective_sample_size=n,training_weight=w)
            covariance,cov_status=_fit_covariance(raw_bundle,target,qualified,train,bg_index,device,maxiter,
                reference_count=n,training_weight=w)
            bg=softmax(np.einsum("npd,fd->fnp",raw_features,background,optimize=True),axis=-1)
            role_field,_=source_role_evidence(raw_bundle,covariance,bg,rhs=geometry["rhs"],correlated=True)
            groups=training_run[case:case+1]&(member@active.astype(float)>0.)[None]
            role,role_status=_fit_period_role(raw_base,role_field,np.einsum("npd,fd->fnp",raw_features,core,optimize=True),
                target/scale,qualified,train,period/scale,maxiter,device,
                cohort=(member,average,group_reference,groups),reference_count=n,training_weight=w)
            statuses=status+[bg_status,cov_status,role_status]
            raw_score=np.einsum("npd,fd->fnp",raw_features,core,optimize=True)+_role_score(role_field,raw_base,role)
            score=raw_score
            if definition["training_stage"]=="reference_coherent_amplitude":
                delta,fit=_fit_score(coherent_features,target,qualified,train,period,
                    parent=np.einsum("npd,fd->fnp",coherent_features,core,optimize=True),
                    ordered=True,ranking=True,rivals=raw_rivals,maxiter=maxiter,effective_sample_size=n,training_weight=w)
                core=core+delta;statuses.append(fit);bundle=bundles[expert]
                bg=softmax(np.einsum("npd,fd->fnp",bundle["raw_features"],background,optimize=True),axis=-1)
                source_role,_=source_role_evidence(bundle,covariance,bg,rhs=bundle["rhs"],correlated=True)
                features=bundle["core_features"] if definition["core_coherent_peer"] else bundle["raw_features"]
                fixed=_role_score(source_role,bundle["base_features"],role)
                rivals=np.stack([bundle["base_features"][:,:,0].argmax(1),bundle["base_features"][:,:,4].argmax(1)],axis=-1)
                delta,fit=_fit_score(features,target,qualified,train,period,
                    parent=np.einsum("npd,fd->fnp",features,core,optimize=True)+fixed,
                    ordered=True,ranking=True,rivals=rivals,maxiter=maxiter,effective_sample_size=n,training_weight=w)
                core=core+delta;statuses.append(fit);score=np.einsum("npd,fd->fnp",features,core,optimize=True)+fixed
            probability=softmax(score,axis=-1);mean=(probability@(period/scale))@average.T
            _,x,y,number=fit_monotone_mean(mean,group_reference,groups)
            raw_point=softmax(np.einsum("npd,fd->fnp",raw_features,anchor,optimize=True),axis=-1)@(period/scale)
            deviation=raw_point-(raw_point@average.T)[:,data["run_index"]]
            residual=np.where(qualified,target/scale-group_reference[data["run_index"]],0.)
            variance=float(weight@(residual**2))
            alpha=np.maximum((np.sum(w*deviation*residual[None],axis=1)+variance/n)/
                (np.sum(w*deviation**2,axis=1)+variance/n),0.)
            local.append({"core_coefficient":core[0],"anchor_coefficient":anchor[0],"background_coefficient":background[0],
                "role_covariance":covariance[0],"role_coefficient":role[0],"calibration_prediction":x[0],
                "calibration_mean":y[0],"calibration_experiments":number[0],"within_scale":alpha[0],
                "core_coherent_peer":np.asarray(definition["core_coherent_peer"]),
                "core_partial_peer":np.asarray(definition["core_partial_peer"]),
                "mean_identity_tails":np.asarray(definition["mean_identity_tails"]),
                "success":np.asarray(all(value["success"] for value in statuses))})
            local_masks.append(active);local_weights.append(weight);local_n.append(int(n[0]));local_g.append(g)
        records.append({key:np.stack([value[key] for value in local]) for key in local[0]})
        masks.append(np.stack(local_masks));weights.append(np.stack(local_weights));source_counts.append(local_n);unit_counts.append(local_g)
        print("REFERENCE_CALIBRATION_FIT",int(case),"N",local_n,"G",local_g,"success",[bool(value["success"]) for value in local],
            "seconds",time.time()-start,flush=True)
    model={"calibration_"+key:np.stack([value[key] for value in records]) for key in records[0]}
    mask=np.stack(masks);union=mask.any(1)
    model.update({"schema":np.asarray("source_assignment_unified_0.3.1"),"measurement_kind":np.asarray("reference_calibration_mixture"),
        "readout_kind":np.asarray("information_conditioned_reference_W2"),"period":period,"period_scale":np.asarray(scale),
        "time_unit":np.asarray(time_unit),"case_index":cases,"success":model.pop("calibration_success").all(1),
        "reference_information_knots":np.stack(knots),"calibration_recipe_json":np.asarray(json.dumps(recipe,sort_keys=True)),
        "calibration_name":np.asarray([value["name"] for value in recipe]),
        "calibration_representation_options_json":np.asarray([json.dumps(value["representation_options"],sort_keys=True) for value in recipe]),
        "calibration_fundamental_support":np.asarray([value["fundamental_support"] for value in recipe]),
        "calibration_reference_source_count":np.asarray(source_counts),"calibration_reference_unit_count":np.asarray(unit_counts),
        "reference_supervision_mask":mask,"reference_supervision_weight":np.stack(weights),
        "calibration_reference_source_id":np.where(mask,np.broadcast_to(source_id,mask.shape),""),
        "reference_source_id":source_id,"reference_source_count":np.asarray([len(np.unique(source_id[row])) for row in union]),
        "effective_training_units":np.asarray([len(np.unique(independent_unit[row])) for row in union]),
        "training_run":training_run[cases],"source_refs_qualified":qualified,"source_refs_reference":reference,
        "source_refs_reference_source_id":source_id,"source_refs_independent_unit":independent_unit,
        "qualified":qualified,"reference":reference,"trace_id":data["metadata"].index.to_numpy(dtype=str),
        "run_names":data["runs"],"foreground_prior_kind":np.asarray("uniform"),
        "reference_information_definition":np.asarray("exp(mean_period(0.5*logdet(actual_H1_trend_residual_Gram)))"),
        "reference_weight_definition":np.asarray("continuous_piecewise_linear_barycentric_weights_clipped_at_reference_information_knots"),
        "parameter_policy":np.asarray("reference_calibration_recipe_learned_only_from_allowed_sources; each_delta_centered_at_its_own_reference_parent"),
        "model_kind":np.asarray("information_conditioned_reference_W2"),"source_operator":np.asarray("same_actual_time_mask_source_formula_per_declared_calibration"),
        "source_core_policy":np.asarray("reference_stage_raw_identity; optional_reference_coherent_amplitude_deltas"),
        "background_reference_policy":np.asarray("No_peer_H1_proxy" if background_reference is None else "supplied_training_background_period"),
        "methods":np.asarray(["unified"]),"method_indices":np.asarray([0]),"preferred_method":np.asarray(0)})
    model.update(predict_reference_calibration_mixture(frame,model,device=device,source_bundles=bundles,include_diagnostics=True))
    return model


def fit_reference_source_w2(bundle,reference,qualified,training_run,background_reference,independent_unit,
        reference_source_id=None,initial_model=None,maxiter=900,device="cuda",information_conditioned=False,view_context=None,raw_frame=None,faithful_adaptation=False):
    """Original shared source-role fit, uniform base, and one W2 density readout."""
    if faithful_adaptation:
        model=fit_reference_source_adaptation(bundle,raw_frame,reference,qualified,training_run,background_reference,independent_unit,
            reference_source_id=reference_source_id,maxiter=maxiter,device=device)
    elif information_conditioned:
        information=observed_source_information(bundle);bounds=[]
        def fit_case(index):
            allowed=training_run[index,bundle["run_index"]]
            low=float(information[allowed].min());high=float(information[allowed].max());bounds.append([low,high])
            q=np.divide(information-low,high-low,out=np.zeros_like(information),where=high>low).clip(0.,1.)
            current=dict(bundle,source_information_coordinate=q)
            initial=None if initial_model is None else {key:value[index:index+1] for key,value in initial_model.items()}
            return fit_general_source_roles(current,reference,qualified,training_run[index:index+1],background_reference,
                maxiter=maxiter,device=device,source_consistent=True,independent_unit=independent_unit,conditioned=True,
                reference_source_id=reference_source_id,foreground_prior_kind="uniform",initial_model=initial,
                legacy_correlated_roles=True,conditioning_field="source_information_coordinate",view_context=view_context)
        values=list(map(fit_case,np.arange(len(training_run))))
        model={key:np.concatenate([value[key] for value in values],axis=0) if np.ndim(values[0][key])>0 else values[0][key] for key in values[0]}
        model.update({"source_information_min":np.asarray(bounds)[:,0],"source_information_max":np.asarray(bounds)[:,1],
            "conditioning_coordinate":np.asarray("source_information_coordinate"),
            "source_information_definition":np.asarray("exp(mean_period(0.5*logdet(H1_trend_residual_Gram))); training_fold_minmax_clip")})
    else:
        model=fit_general_source_roles(bundle,reference,qualified,training_run,background_reference,
            maxiter=maxiter,device=device,source_consistent=True,independent_unit=independent_unit,
            reference_source_id=reference_source_id,foreground_prior_kind="uniform",initial_model=initial_model,
            legacy_correlated_roles=True,view_context=view_context)
    model.update({"schema":np.asarray("source_assignment_unified_0.3.1"),
        "readout_kind":np.asarray("reference_source_bounded_W2"),
        "model_kind":np.asarray("reference_source_bounded_W2"),
        "source_operator":np.asarray("actual_time_mask_shared_reference_source"),
        "source_core_policy":np.asarray("original104_coherent_core31_opposite_roles_uniform_prior"),
        "period":bundle["period"],"period_scale":bundle["period_scale"],
        "calibration_coordinate":np.asarray("period_divided_by_period_scale"),
        "mean_identity_tails":np.ones(len(training_run),dtype=bool)})
    result=predict_reference_source_w2(bundle,model,device=device)
    model.update({key:value for key,value in result.items() if key not in ["calibration_kind","calibration_discretization","calibration_interval_semantics"]})
    model["calibration_kind"]=np.asarray("cohort_bounded_W2_affine_pushforward")
    return model


def observed_source_information(bundle):
    """Phase-coverage/observation information from unit-invariant H1 geometry."""
    period=np.arange(len(bundle["period"]))
    gram=bundle["joint_gram"][bundle["layout"][:,None],period[None],period[None],:2,:2]
    sign,logdet=np.linalg.slogdet(gram)
    assert np.all(sign>0.)
    return np.exp(.5*logdet.mean(axis=-1))


def fit_reference_source_adaptation(source_bundle,raw_frame,reference,qualified,training_run,background_reference,
        independent_unit,reference_source_id=None,maxiter=900,device="cuda"):
    """Fit reference -> coherent -> amplitude delta priors on legal references."""
    data=observations(raw_frame);period=source_bundle["period"];scale=float(source_bundle["period_scale"])
    order=pd.Index(data["metadata"].index).get_indexer(source_bundle["trace_id"])
    assert np.array_equal(order,np.arange(len(order)))
    raw_base=profile_features(data["signal"],data,period,scale,ridge_fraction=0.)[0]
    raw_geometry=_role_geometry(data["signal"],data,period,scale,normalize_reference_peer=True)
    raw_coherent=generic_coherent_fwl_power(data["time"],data["mask"],raw_geometry["unit"],raw_geometry["peer"],period,half_gram=False)
    reference_bundle=dict(source_bundle,base_features=raw_base,source_base=raw_base,
        original_unit=raw_geometry["unit"],original_peer=raw_geometry["peer"],
        source_unit=raw_geometry["unit"],source_peer=raw_geometry["peer"],
        original_rhs=raw_geometry["rhs"],source_rhs=raw_geometry["rhs"],
        harmonic_design=raw_geometry["design"],joint_gram=raw_geometry["joint"],layout=raw_geometry["layout"],
        identity_base_features=raw_base)
    identities=source_bundle["trace_id"] if reference_source_id is None else np.asarray(reference_source_id)
    source_code,source_names=pd.factorize(identities,sort=True)
    source_member=(np.arange(len(source_names))[:,None]==source_code[None]).astype(float)
    unit_code,unit_names=pd.factorize(independent_unit,sort=True)
    unit_member=(np.arange(len(unit_names))[:,None]==unit_code[None]).astype(float)
    target=np.where(qualified,reference,0.)
    train=training_run[:,source_bundle["run_index"]]&qualified[None]
    number=((train.astype(float)@source_member.T)>0.).sum(1)
    unit_group=(source_bundle["membership"]@unit_member.T)>0.
    count=source_bundle["membership"]@qualified.astype(float)
    group_reference=np.divide(source_bundle["membership"]@(target/scale),count,out=np.zeros(len(count)),where=count>0.)
    bg_index=abs(period[None]-background_reference[:,None]).argmin(1)
    bg_target=period[bg_index]
    left,right=np.triu_indices(13)
    bg_columns=np.flatnonzero(np.r_[np.isin(np.arange(13),[4,5,6,7,8]),np.isin(left,[4,5,6,7,8])&np.isin(right,[4,5,6,7,8])])
    raw_features=_polynomial(raw_base)
    coherent_base=raw_base.copy();coherent_base[:,:,11]=raw_coherent
    coherent_features=_polynomial(coherent_base)
    source_features=source_bundle["core_features"]
    zero=np.zeros((1,len(period)));values=[]
    def case(index):
        started=time.time();allowed=train[index:index+1];groups=training_run[index:index+1];n=number[index:index+1]
        g=((groups.astype(float)@unit_group)>0.).sum(1)
        print("REFERENCE_ADAPTATION_START",index,int(n[0]),int(g[0]),flush=True)
        anchor,reference_core,reference_status=_fit_core(raw_base,target,qualified,allowed,period,maxiter,zero,
            reference_count=n,sequential_reference=True)
        background,bg_status=_fit_score(raw_features,bg_target,qualified,allowed,period,columns=bg_columns,
            maxiter=maxiter,effective_sample_size=n)
        covariance,cov_status=_fit_covariance(reference_bundle,target,qualified,allowed,bg_index,device,maxiter,reference_count=n)
        bg_probability=softmax(np.einsum("npd,fd->fnp",raw_features,background,optimize=True),axis=-1)
        role_field,_=source_role_evidence(reference_bundle,covariance,bg_probability,rhs=raw_geometry["rhs"],correlated=True)
        role,role_status=_fit_period_role(raw_base,role_field,
            np.einsum("npd,fd->fnp",raw_features,reference_core,optimize=True),
            target/scale,qualified,allowed,period/scale,maxiter,device,
            cohort=(source_bundle["membership"],source_bundle["average"],group_reference,groups),reference_count=n)
        rivals=np.stack([raw_base[:,:,0].argmax(1),raw_base[:,:,4].argmax(1)],axis=-1)
        coherent_delta,coherent_status=_fit_score(coherent_features,target,qualified,allowed,period,
            parent=np.einsum("npd,fd->fnp",coherent_features,reference_core,optimize=True),
            ordered=True,ranking=True,rivals=rivals,maxiter=maxiter,effective_sample_size=n)
        coherent_core=reference_core+coherent_delta
        source_bg=softmax(np.einsum("npd,fd->fnp",source_bundle["raw_features"],background,optimize=True),axis=-1)
        source_role,_=source_role_evidence(source_bundle,covariance,source_bg,rhs=source_bundle["rhs"],correlated=True)
        fixed_role=_role_score(source_role,source_bundle["base_features"],role)
        source_rivals=np.stack([source_bundle["base_features"][:,:,0].argmax(1),source_bundle["base_features"][:,:,4].argmax(1)],axis=-1)
        source_delta,source_status=_fit_score(source_features,target,qualified,allowed,period,
            parent=np.einsum("npd,fd->fnp",source_features,coherent_core,optimize=True)+fixed_role,
            ordered=True,ranking=True,rivals=source_rivals,maxiter=maxiter,effective_sample_size=n)
        source_core=coherent_core+source_delta
        score=np.einsum("npd,fd->fnp",source_features,source_core,optimize=True)+fixed_role
        probability=softmax(score,axis=-1);source_mean=(probability@(period/scale))@source_bundle["average"].T
        _,x,y,cn=fit_monotone_mean(source_mean,group_reference,groups&(count>0.)[None])
        alpha,raw_point=_within_scale(reference_bundle,anchor,zero,target,qualified,groups,0)
        within_reference=np.where(qualified,target/scale-group_reference[source_bundle["run_index"]],0.)
        alpha=np.where(np.sum(allowed*within_reference[None]**2,axis=1)==0.,0.,alpha)
        statuses=reference_status+[bg_status,cov_status,role_status,coherent_status,source_status]
        value={"core_coefficient":source_core[0],"anchor_coefficient":anchor[0],
            "background_coefficient":background[0],"role_covariance":covariance[0],"role_coefficient":role[0],
            "reference_core_coefficient":reference_core[0],"coherent_core_coefficient":coherent_core[0],
            "coherent_delta_coefficient":coherent_delta[0],"source_delta_coefficient":source_delta[0],
            "calibration_prediction":x[0],"calibration_mean":y[0],"calibration_experiments":cn[0],
            "within_scale":alpha[0],"mean_bias":np.asarray(0.),"reference_source_count":n[0],
            "effective_training_units":g[0],"success":np.asarray(all(item["success"] for item in statuses)),
            "fit_elapsed_s":np.asarray(time.time()-started)}
        values.append(value)
        print("REFERENCE_ADAPTATION_CASE",index,bool(value["success"]),float(value["fit_elapsed_s"]),
            "ref_coh_amp_L2",float(np.linalg.norm(reference_core)),float(np.linalg.norm(coherent_delta)),float(np.linalg.norm(source_delta)),flush=True)
        return value
    list(map(case,np.arange(len(training_run))))
    model={key:np.stack([value[key] for value in values]) for key in values[0]}
    model.update({"foreground_prior_kind":np.asarray("uniform"),
        "fitting_policy":np.asarray("same_allowed_raw_reference_then_coherent_delta_then_amplitude_delta; penalize_only_each_delta"),
        "raw_orthogonal_and_coherent_policy":np.asarray("raw_reference_retains_orthogonal11; coherent_delta_replaces11_with_raw_cross_half_FWL; amplitude_delta_uses_source_coherent11")})
    return model


def refit_source_amplitude_views(bundle,context,parent,original_bundle,maxiter=900,device="cuda",cohort_response=False,pair_context=None):
    """Only amplitude delta sees source-calibrated waveform views; parents freeze."""
    context=dict(context)
    order=pd.Index(context["trace_id"]).get_indexer(bundle["trace_id"]);assert np.all(order>=0)
    row_keys=["target","qualified","origin_run","origin_trace_id","independent_unit","view_weight","is_original","reference_source_id"]
    ctx={key:context[key][order] for key in row_keys}
    origin_index=pd.Index(parent["run_names"]).get_indexer(ctx["origin_run"]);assert np.all(origin_index>=0)
    group_origin=np.rint(bundle["average"]@origin_index).astype(int)
    training=parent["training_run"][:,group_origin]
    weight=source_reference_weights(bundle,ctx,training)
    np.testing.assert_array_equal(weight["effective_sample_size"],parent["reference_source_count"])
    np.testing.assert_array_equal(weight["effective_training_units"],parent["effective_training_units"])
    target=np.where(ctx["qualified"],ctx["target"],0.)
    original_allowed=weight["train"]&ctx["is_original"][None]
    prior_weight=original_allowed/original_allowed.sum(1)[:,None]
    period=bundle["period"];scale=float(bundle["period_scale"]);values=[]
    source_original=np.flatnonzero(ctx["is_original"])
    source_order=pd.Index(ctx["origin_trace_id"][source_original]).get_indexer(parent["trace_id"]);assert np.all(source_order>=0)
    source_original=source_original[source_order]
    group_count=bundle["membership"]@ctx["qualified"].astype(float)
    view_group_reference=np.divide(bundle["membership"]@target,group_count,out=np.zeros(len(group_count)),where=group_count>0.)
    prior_mean=prior_weight@target
    original_variance=np.maximum(prior_weight@(target**2)-prior_mean**2,(period[1]-period[0])**2/12.)
    if cohort_response:
        names=pd.Index(bundle["run_names"])
        pleft=pair_context["pair_left_run"].ravel();pright=pair_context["pair_right_run"].ravel()
        left=names.get_indexer(pair_context["run_name"][np.maximum(pleft,0)])
        right=names.get_indexer(pair_context["run_name"][np.maximum(pright,0)])
        pair_eligible=pair_context["pair_eligible"].ravel()&(pleft>=0)&(pright>=0)&(left>=0)&(right>=0)
        left=np.maximum(left,0);right=np.maximum(right,0)
        pair_effect=pair_context["pair_effect"].ravel()
    parent_run=pd.Index(parent["run_names"]).get_indexer(ctx["origin_run"][source_original])
    member=(np.arange(len(parent["run_names"]))[:,None]==parent_run[None]).astype(float)
    average=member/member.sum(1)[:,None]
    number=member@parent["qualified"].astype(float)
    truth=(member@np.where(parent["qualified"],parent["reference"]/scale,0.))/number
    rivals=np.stack([bundle["base_features"][:,:,0].argmax(1),bundle["base_features"][:,:,4].argmax(1)],axis=-1)
    for index in range(len(parent["core_coefficient"])):
        started=time.time();n=parent["reference_source_count"][index:index+1]
        background=softmax(np.einsum("npd,d->np",bundle["raw_features"],parent["background_coefficient"][index],optimize=True),axis=-1)[None]
        role,_=source_role_evidence(bundle,parent["role_covariance"][index:index+1],background,rhs=bundle["rhs"],correlated=True)
        fixed_role=_role_score(role,bundle["base_features"],parent["role_coefficient"][index:index+1])
        fixed=np.einsum("npd,d->np",bundle["core_features"],parent["coherent_core_coefficient"][index],optimize=True)[None]+fixed_role
        print("SOURCE_VIEW_DELTA_START",index,int(n[0]),int(weight["effective_training_units"][index]),int(weight["train"][index].sum()),flush=True)
        moment=None;paired=None
        if cohort_response:
            moment={"average":bundle["average"],"reference":view_group_reference,
                "weight":weight["group_weight"][index:index+1],"variance":np.full(len(bundle["run_names"]),original_variance[index])}
            eligible=pair_eligible&training[index,left]&training[index,right]
            pair_weight=.5*(weight["group_weight"][index,left]+weight["group_weight"][index,right])*eligible
            pair_weight/=pair_weight.sum()
            paired={"average":bundle["average"],"left":left,"right":right,"effect":pair_effect,
                "weight":pair_weight[None],"variance":original_variance[index]}
            print("SOURCE_VIEW_MOMENT_CF",index,"original_sd",float(np.sqrt(original_variance[index])),
                "group_weight",float(moment["weight"].sum()),"pair_weight",float(pair_weight.sum()),"pairs",int(eligible.sum()),flush=True)
        delta,status=_fit_score(bundle["core_features"],target,ctx["qualified"],weight["train"][index:index+1],period,
            parent=fixed,ordered=True,ranking=True,rivals=rivals,maxiter=maxiter,
            training_weight=weight["training_weight"][index:index+1],effective_sample_size=n,
            prior_scale_weight=prior_weight[index:index+1],initial_coefficient=parent["source_delta_coefficient"][index:index+1],cohort_moments=moment,paired_moments=paired)
        coefficient=parent["coherent_core_coefficient"][index:index+1]+delta
        # Mean calibration sees original No only and the newly adapted density.
        ob=softmax(np.einsum("npd,d->np",original_bundle["raw_features"],parent["background_coefficient"][index],optimize=True),axis=-1)[None]
        orole,_=source_role_evidence(original_bundle,parent["role_covariance"][index:index+1],ob,rhs=original_bundle["rhs"],correlated=True)
        score=np.einsum("npd,fd->fnp",original_bundle["core_features"],coefficient,optimize=True)+_role_score(orole,original_bundle["base_features"],parent["role_coefficient"][index:index+1])
        source_mean=(softmax(score,axis=-1)@(period/scale))@original_bundle["average"].T
        _,x,y,count=fit_monotone_mean(source_mean,truth,parent["training_run"][index:index+1])
        values.append({"core_coefficient":coefficient[0],"source_delta_coefficient":delta[0],
            "calibration_prediction":x[0],"calibration_mean":y[0],"calibration_experiments":count[0],
            "success":np.asarray(status["success"]),"view_delta_iterations":np.asarray(status["iterations"]),
            "view_delta_gradient":np.asarray(status["gradient"]),"view_delta_elapsed_s":np.asarray(time.time()-started)})
        print("SOURCE_VIEW_DELTA_CASE",index,status["success"],status["iterations"],time.time()-started,flush=True)
    model=dict(parent)
    model.update({key:np.stack([value[key] for value in values]) for key in values[0]})
    model.update(predict_reference_source_w2(original_bundle,model,device=device))
    model["fitting_policy"]=np.asarray("original_reference_coherent_BG_cov_role_anchor_within_frozen; amplitude_delta_real_source_views_only")
    model["source_view_weight_policy"]=np.asarray("unit_origin_viewshare_distinct_FG_source; original_source_N_G; query_origin_and_all_views_excluded")
    model["source_delta_prior_scale_policy"]=np.asarray("original_allowed_No_only; prior_zero_delta_about_frozen_coherent_parent")
    if cohort_response:
        model["fitting_policy"]=np.asarray("frozen_original_parents; amplitude_delta_real9_proper_cohort_and_paired_source_response")
        model["source_response_loss"]=np.asarray("CE+CRPS+rival+0.5_group_mean_MSE/original_SD2+0.5_paired_effect_MSE/original_SD2; normalized_group_pair_weights")
    return model


def reference_phase_log_likelihood(phase_waveform,data,period):
    """Actual-time single-H1 unknown-variance working likelihood."""
    scale=float(np.sqrt(period[0]*period[-1]))
    value=detrend(np.asarray(phase_waveform,dtype=float),data["time"],data["mask"])
    profile=harmonic_profile(value,data,period,scale,harmonics=1,ridge_fraction=0.)
    energy=np.sum(value**2,axis=1)
    rss=np.maximum(energy[:,None]-profile["energy_one"],np.finfo(float).eps*np.maximum(energy[:,None],1.))
    score=-.5*(data["count"]-2.)[:,None]*np.log(rss/np.maximum(energy[:,None],np.finfo(float).tiny))
    return score-score.max(1)[:,None]


def fit_reference_phase_source(data,source_bundle,teacher,context,parent,original_bundle,*,learner_kind="hgb",
        learner_path,prepared=None,maxiter=900,initial_model=None,initial_learners=None):
    """Dense reference phase -> one likelihood temperature -> original PAV/W2."""
    import joblib
    from source_assignment_distribution import reference_phase_temporal_features,fit_reference_phase_learner,predict_reference_phase_learner
    period=source_bundle["period"];scale=float(source_bundle["period_scale"])
    order=pd.Index(context["trace_id"]).get_indexer(source_bundle["trace_id"]);assert np.all(order>=0)
    keys=["target","qualified","origin_run","origin_trace_id","independent_unit","view_weight","is_original","reference_source_id"]
    ctx={key:context[key][order] for key in keys}
    teacher=np.asarray(teacher)[order]
    group_origin=np.rint(source_bundle["average"]@pd.Index(parent["run_names"]).get_indexer(ctx["origin_run"])).astype(int)
    training=parent["training_run"][:,group_origin]
    weights=source_reference_weights(source_bundle,ctx,training)
    np.testing.assert_array_equal(weights["effective_sample_size"],parent["reference_source_count"])
    if prepared is None:
        prepared=reference_phase_temporal_features(data,source_bundle,scale)
    original=np.flatnonzero(ctx["is_original"])
    original=original[pd.Index(ctx["origin_trace_id"][original]).get_indexer(parent["trace_id"])]
    raw_score=np.einsum("npd,fd->fnp",_polynomial(original_bundle["identity_base_features"]),parent["anchor_coefficient"],optimize=True)
    raw_log=raw_score-logsumexp(raw_score,axis=-1,keepdims=True)
    member=original_bundle["membership"];average=original_bundle["average"]
    number=member@parent["qualified"].astype(float)
    reference_mean=(member@np.where(parent["qualified"],parent["reference"]/scale,0.))/number
    target=np.where(ctx["qualified"],ctx["target"],0.);teacher_mass=period_reference_mass(period,target,ctx["qualified"])
    models=[];values=[]
    for index,case in enumerate(parent["case_index"]):
        reuse=-1 if initial_model is None else int(pd.Index(initial_model["case_index"]).get_indexer([case])[0])
        if reuse>=0:
            np.testing.assert_array_equal(parent["training_run"][index],initial_model["training_run"][reuse])
            np.testing.assert_array_equal(parent["anchor_coefficient"][index],initial_model["anchor_coefficient"][reuse])
            np.testing.assert_array_equal(parent["within_scale"][index],initial_model["within_scale"][reuse])
            fields=["posterior","source_period","estimate","group_mean","deviation","cohort_density_tilt","cohort_density_shift",
                "cohort_density_scale","cell_clipped_probability","calibration_prediction","calibration_mean","calibration_experiments",
                "source_temperature","temperature_success","fit_elapsed_s","source_assignment_probability","phase_waveform","source_log_likelihood"]
            values.append({key:initial_model[key][reuse] for key in fields})
            row=int(pd.Index(initial_learners["case_index"]).get_indexer([case])[0])
            models.append(initial_learners["models"][row])
            joblib.dump({"models":models,"case_index":parent["case_index"][:index+1],"measurement_kind":"reference_phase_waveform"},learner_path)
            print("REFERENCE_PHASE_CASE_REUSED",int(case),learner_kind,flush=True)
            continue
        started=time.time();n=int(parent["reference_source_count"][index])
        learner=fit_reference_phase_learner(prepared,teacher,weights["training_weight"][index],n,learner_kind=learner_kind)
        waveform=predict_reference_phase_learner(prepared,learner)
        score=reference_phase_log_likelihood(waveform,data,period)
        weight=weights["training_weight"][index]
        def temperature_objective(parameter):
            eta=float(parameter[0]);temperature=np.exp(eta)
            energy=temperature*score;log=energy-logsumexp(energy,axis=-1,keepdims=True);mass=np.exp(log)
            loss=-np.sum(weight[:,None]*teacher_mass*log)+eta*eta/(2.*n)
            gradient=np.sum(weight[:,None]*(mass-teacher_mass)*score)*temperature+eta/n
            return float(loss),np.asarray([gradient])
        fitted=minimize(temperature_objective,np.zeros(1),jac=True,method="BFGS",options={"maxiter":maxiter,"gtol":1e-6})
        temperature=float(np.exp(fitted.x[0]));probability=softmax(temperature*score[original],axis=-1)
        source_mean=(probability@(period/scale))@average.T
        _,x,y,count=fit_monotone_mean(source_mean[None],reference_mean,parent["training_run"][index:index+1])
        target_mean=interpolate_mean(source_mean[None],x,y,count,identity_tails=True)*scale
        target_mean=np.clip(target_mean,period[0],period[-1])
        final=calibrate_source_density(raw_log[index:index+1],period,original_bundle["run_index"],target_mean,
            calibration_kind="wasserstein_affine",within_scale=parent["within_scale"][index])
        models.append(learner)
        value={key:array[0] for key,array in final.items() if isinstance(array,np.ndarray) and array.ndim>0}
        value.update({"calibration_prediction":x[0],"calibration_mean":y[0],"calibration_experiments":count[0],
            "source_temperature":np.asarray(temperature),"temperature_success":np.asarray(fitted.success),
            "fit_elapsed_s":np.asarray(time.time()-started),"source_assignment_probability":probability,
            "phase_waveform":waveform[original],"source_log_likelihood":score[original]})
        values.append(value)
        joblib.dump({"models":models,"case_index":parent["case_index"][:index+1],"measurement_kind":"reference_phase_waveform"},learner_path)
        print("REFERENCE_PHASE_CASE",int(case),learner_kind,"N",n,"G",int(weights["effective_training_units"][index]),
            "timestamps",learner["training_timestamps"],"features",learner["feature_count"],
            "temperature",temperature,"temperature_success",bool(fitted.success),"seconds",time.time()-started,flush=True)
    model={key:np.stack([value[key] for value in values]) for key in values[0]}
    for key in ["anchor_coefficient","within_scale","period","period_scale","reference_source_count","effective_training_units",
        "trace_id","run_names","training_run","case_index","qualified","reference","time_unit","representation_options_json","fundamental_support"]:
        model[key]=parent[key]
    model.update({"schema":np.asarray("source_assignment_unified_0.3.1"),
        "measurement_kind":np.asarray("reference_phase_waveform"),"readout_kind":np.asarray("reference_phase_temporal_bounded_W2"),
        "learner_kind":np.asarray(learner_kind),"phase_learner_path":np.asarray(str(learner_path)),
        "foreground_prior_kind":np.asarray("uniform"),"calibration_coordinate":np.asarray("period_divided_by_period_scale"),
        "shape_sigma":prepared["shape_sigma"],"temporal_lags":prepared["temporal_lags"],
        "feature_definition":prepared["feature_definition"],"virtual_coordinate_count":np.asarray(4),
        "coordinate_topology":np.asarray("virtual_transform_basis"),
        "training_supervision":np.asarray("measured_FG_full_waveform_H1_normalized_signed_phase; reference_period_teacher_only"),
        "source_period":model["estimate"].copy()})
    return model


def predict_reference_phase_source(frame,model,*,source_bundle=None,learner_path=None,cases=None,device="cpu",include_diagnostics=False):
    """No-only temporal source phase and the same final raw-density W2 readout."""
    import joblib
    from source_assignment_distribution import reference_phase_temporal_features,predict_reference_phase_learner
    data=observations(frame)
    if source_bundle is None:
        options=json.loads(str(model["representation_options_json"]))
        source_bundle=reference_source_observations(frame,model["period"],options,
            fundamental_support=bool(model["fundamental_support"]),device=device)
    prepared=reference_phase_temporal_features(data,source_bundle,float(model["period_scale"]))
    learned=joblib.load(str(model["phase_learner_path"]) if learner_path is None else learner_path)
    cases=np.arange(len(model["case_index"])) if cases is None else np.atleast_1d(cases)
    raw_score=np.einsum("npd,fd->fnp",_polynomial(source_bundle["identity_base_features"]),model["anchor_coefficient"][cases],optimize=True)
    raw_log=raw_score-logsumexp(raw_score,axis=-1,keepdims=True)
    values=[]
    stacked=bool(model.get("phase_reference_stacking",False))
    reference_score=raw_reference_period_score(data,model["period"],model,cases=cases,device=device) if stacked else None
    for position,index in enumerate(cases):
        row=int(pd.Index(learned["case_index"]).get_indexer([model["case_index"][index]])[0]);assert row>=0
        waveform=predict_reference_phase_learner(prepared,learned["models"][row])
        score=reference_phase_log_likelihood(waveform,data,model["period"])
        energy=reference_score[position]+float(model["source_phase_weight"][index])*score if stacked else float(model["source_temperature"][index])*score
        probability=softmax(energy,axis=-1)
        source_mean=(probability@(model["period"]/float(model["period_scale"])))@source_bundle["average"].T
        target=interpolate_mean(source_mean[None],model["calibration_prediction"][index:index+1],
            model["calibration_mean"][index:index+1],model["calibration_experiments"][index:index+1],True)*float(model["period_scale"])
        target=np.clip(target,model["period"][0],model["period"][-1])
        result=calibrate_source_density(raw_log[position:position+1],model["period"],data["run_index"],target,
            calibration_kind="wasserstein_affine",within_scale=model["within_scale"][index])
        result={key:value[0] for key,value in result.items() if isinstance(value,np.ndarray) and value.ndim>0}
        if include_diagnostics:
            result.update({"source_assignment_probability":probability,"raw_individual_probability":softmax(raw_log[position],-1),
                "raw_individual_log_probability":raw_log[position],"predicted_source_phase":waveform,
                "source_log_likelihood":score,"calibrated_mean_target":target[0]})
        values.append(result)
    return {key:np.stack([value[key] for value in values]) for key in values[0]}


def raw_reference_period_score(data,period,parent,cases=None,device="cpu"):
    """Fixed original-No orthogonal13/core104 and same-reference role31 score."""
    cases=np.arange(len(parent["reference_core_coefficient"])) if cases is None else np.atleast_1d(cases)
    scale=float(np.sqrt(period[0]*period[-1]))
    base=profile_features(data["signal"],data,period,scale,ridge_fraction=0.)[0]
    features=_polynomial(base);geometry=_role_geometry(data["signal"],data,period,scale,normalize_reference_peer=True)
    bundle={"period":period,"source_rhs":geometry["rhs"],"rhs":geometry["rhs"],"joint_gram":geometry["joint"],"layout":geometry["layout"]}
    output=[]
    for case in cases:
        background=softmax(np.einsum("npd,d->np",features,parent["background_coefficient"][case],optimize=True),axis=-1)[None]
        role,_=source_role_evidence(bundle,parent["role_covariance"][case:case+1],background,rhs=geometry["rhs"],correlated=True)
        score=np.einsum("npd,d->np",features,parent["reference_core_coefficient"][case],optimize=True)
        score+=_role_score(role,base,parent["role_coefficient"][case:case+1])[0]
        output.append(score)
    return np.stack(output)


def fit_reference_phase_stacking(data,source_bundle,context,parent,phase_model,original_bundle,*,prepared=None,maxiter=900):
    """One nonnegative conditional phase-score weight; fixed raw source identity."""
    import joblib
    from source_assignment_distribution import reference_phase_temporal_features,predict_reference_phase_learner
    period=source_bundle["period"];scale=float(source_bundle["period_scale"])
    order=pd.Index(context["trace_id"]).get_indexer(source_bundle["trace_id"])
    keys=["target","qualified","origin_run","origin_trace_id","independent_unit","view_weight","is_original","reference_source_id"]
    ctx={key:context[key][order] for key in keys}
    origin=pd.Index(parent["run_names"]).get_indexer(ctx["origin_run"]);group_origin=np.rint(source_bundle["average"]@origin).astype(int)
    training=parent["training_run"][:,group_origin];weights=source_reference_weights(source_bundle,ctx,training)
    np.testing.assert_array_equal(weights["effective_sample_size"],parent["reference_source_count"])
    if prepared is None:
        prepared=reference_phase_temporal_features(data,source_bundle,scale)
    learned=joblib.load(str(phase_model["phase_learner_path"]))
    reference_score=raw_reference_period_score(data,period,parent,device="cuda")
    target=np.where(ctx["qualified"],ctx["target"],0.);teacher=period_reference_mass(period,target,ctx["qualified"]);cdf=teacher.cumsum(-1)[:,:-1]
    original=np.flatnonzero(ctx["is_original"])
    original=original[pd.Index(ctx["origin_trace_id"][original]).get_indexer(parent["trace_id"])]
    member=original_bundle["membership"];number=member@parent["qualified"].astype(float)
    reference_mean=(member@np.where(parent["qualified"],parent["reference"]/scale,0.))/number
    values=[]
    for index,case in enumerate(parent["case_index"]):
        started=time.time();learner=int(pd.Index(learned["case_index"]).get_indexer([case])[0])
        waveform=predict_reference_phase_learner(prepared,learned["models"][learner])
        phase_score=reference_phase_log_likelihood(waveform,data,period)
        raw=reference_score[index];w=weights["training_weight"][index];g=float(weights["effective_training_units"][index])
        sd=np.sqrt(max(float(w@(target**2)-(w@target)**2),(period[1]-period[0])**2/12.))
        def objective(parameter):
            beta=float(parameter[0]);score=raw+beta*phase_score
            log=score-logsumexp(score,axis=-1,keepdims=True);mass=np.exp(log)
            error=mass.cumsum(-1)[:,:-1]-cdf
            loss=-np.sum(w[:,None]*teacher*log)+np.sum(w[:,None]*error**2)*(period[1]-period[0])/sd+beta*beta/(2.*g)
            partial=np.pad(2.*(period[1]-period[0])*np.cumsum(error[:,::-1],axis=-1)[:,::-1]/sd,((0,0),(0,1)))
            derivative=mass-teacher+mass*(partial-(mass*partial).sum(-1)[:,None])
            gradient=float(np.sum(w[:,None]*derivative*phase_score)+beta/g)
            return float(loss),np.asarray([gradient])
        fit=minimize(objective,np.zeros(1),jac=True,method="L-BFGS-B",bounds=[(0.,None)],
            options={"maxiter":maxiter,"ftol":1e-10,"gtol":1e-6})
        beta=float(fit.x[0]);probability=softmax(raw[original]+beta*phase_score[original],axis=-1)
        source_mean=(probability@(period/scale))@original_bundle["average"].T
        _,x,y,n=fit_monotone_mean(source_mean[None],reference_mean,parent["training_run"][index:index+1])
        values.append({"source_phase_weight":np.asarray(beta),"calibration_prediction":x[0],"calibration_mean":y[0],
            "calibration_experiments":n[0],"success":np.asarray(fit.success),"stack_gradient":np.asarray(float(abs(fit.jac[0]))),
            "source_assignment_probability":probability,"phase_waveform":waveform[original],
            "source_log_likelihood":phase_score[original],"fit_elapsed_s":np.asarray(time.time()-started)})
        print("REFERENCE_PHASE_STACK_CASE",int(case),str(phase_model["learner_kind"]),"N",int(parent["reference_source_count"][index]),
            "G",int(g),"beta",beta,"success",bool(fit.success),"seconds",time.time()-started,flush=True)
    model=dict(parent);model.update({key:np.stack([value[key] for value in values]) for key in values[0]})
    for key in ["phase_learner_path","learner_kind","shape_sigma","temporal_lags","feature_definition","virtual_coordinate_count","coordinate_topology"]:
        model[key]=phase_model[key]
    model.update({"measurement_kind":np.asarray("reference_phase_waveform"),"readout_kind":np.asarray("reference_phase_temporal_bounded_W2"),
        "phase_reference_stacking":np.asarray(True),
        "source_score_definition":np.asarray("fixed_legal_raw_reference_orthogonal13_plus_role31_score + beta*untempered_phase_H1_working_score"),
        "source_weight_prior":np.asarray("0.5_beta_squared_over_independent_unit_G"),
        "source_density_semantics":np.asarray("reference_supervised_conditional_score_stacking; same_No_inputs_not_independent_Bayesian_likelihoods")})
    result=predict_reference_phase_source_from_cached(model,original_bundle,reference_score[:,original])
    model.update(result)
    return model


def predict_reference_phase_source_from_cached(model,bundle,raw_reference_score):
    """Assemble the final source/calibrated/raw/W2 layers from original scores."""
    period=model["period"];scale=float(model["period_scale"])
    raw_anchor=np.einsum("npd,fd->fnp",_polynomial(bundle["identity_base_features"]),model["anchor_coefficient"],optimize=True)
    raw_log=raw_anchor-logsumexp(raw_anchor,axis=-1,keepdims=True)
    target=(interpolate_mean((model["source_assignment_probability"]@(period/scale))@bundle["average"].T,
        model["calibration_prediction"],model["calibration_mean"],model["calibration_experiments"],True)*scale).clip(period[0],period[-1])
    result=calibrate_source_density(raw_log,period,bundle["run_index"],target,
        calibration_kind="wasserstein_affine",within_scale=model["within_scale"][:,None])
    return {key:value for key,value in result.items() if isinstance(value,np.ndarray) and value.ndim>0}


def fit_raw_individual_reference_folds(bundle,reference,qualified,training_run,independent_unit,reference_source_id=None,
        *,case_indices=None,existing_parent=None,maxiter=900):
    """Complete legal raw-No CE/CRPS anchors and within scales for every fold."""
    cases=np.arange(len(training_run)) if case_indices is None else np.asarray(case_indices)
    identities=bundle["trace_id"] if reference_source_id is None else np.asarray(reference_source_id)
    code,names=pd.factorize(identities,sort=True);source_member=(np.arange(len(names))[:,None]==code[None]).astype(float)
    target=np.where(qualified,reference,0.);train=training_run[:,bundle["run_index"]]&qualified[None]
    count=((train.astype(float)@source_member.T)>0.).sum(1)
    unit_code,unit_names=pd.factorize(independent_unit,sort=True)
    unit_member=(np.arange(len(unit_names))[:,None]==unit_code[None]).astype(float)
    allowed_unit=((train.astype(float)@unit_member.T)>0.).sum(1)
    x=_polynomial(bundle["identity_base_features"]);left,right=np.triu_indices(13)
    phase=np.r_[np.arange(13)==12,(left==12)|(right==12)];columns=np.flatnonzero(~phase)
    zero=np.zeros((1,len(bundle["period"])));anchors=[];within=[];success=[]
    for index,case in enumerate(cases):
        reuse=-1 if existing_parent is None else int(pd.Index(existing_parent["case_index"]).get_indexer([case])[0])
        if reuse>=0:
            np.testing.assert_array_equal(training_run[index],existing_parent["training_run"][reuse])
            anchors.append(existing_parent["anchor_coefficient"][reuse]);within.append(existing_parent["within_scale"][reuse]);success.append(bool(existing_parent["success"][reuse]))
            print("RAW_INDIVIDUAL_REFERENCE_REUSED",int(case),int(count[index]),int(allowed_unit[index]),flush=True)
            continue
        likelihood,first=_fit_score(x,target,qualified,train[index:index+1],bundle["period"],columns,parent=zero[:,None],
            ordered=False,maxiter=maxiter,effective_sample_size=count[index:index+1])
        anchor,second=_fit_score(x,target,qualified,train[index:index+1],bundle["period"],columns,parent=zero[:,None],
            ordered=True,maxiter=maxiter,effective_sample_size=count[index:index+1],initial_coefficient=likelihood)
        alpha,_=_within_scale(bundle,anchor,zero,target,qualified,training_run[index:index+1],0)
        number=bundle["membership"]@qualified.astype(float)
        reference_mean=np.divide(bundle["membership"]@(target/float(bundle["period_scale"])),number,
            out=np.zeros(len(number)),where=number>0.)
        deviation=np.where(qualified,target/float(bundle["period_scale"])-reference_mean[bundle["run_index"]],0.)
        alpha=np.where(np.sum(train[index:index+1]*deviation[None]**2,axis=1)==0.,0.,alpha)
        anchors.append(anchor[0]);within.append(alpha[0]);success.append(bool(first["success"] and second["success"]))
        print("RAW_INDIVIDUAL_REFERENCE_FIT",int(case),int(count[index]),int(allowed_unit[index]),bool(success[-1]),flush=True)
    parent={"anchor_coefficient":np.stack(anchors),"within_scale":np.asarray(within),"raw_anchor_success":np.asarray(success),
        "reference_source_count":count,"effective_training_units":allowed_unit,"trace_id":bundle["trace_id"],
        "run_names":bundle["run_names"],"training_run":training_run,"case_index":cases,"qualified":qualified,"reference":reference,
        "period":bundle["period"],"period_scale":bundle["period_scale"]}
    return parent
