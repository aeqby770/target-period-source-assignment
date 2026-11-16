"""Source-period distribution geometry for the current cohort correction."""
import numpy as np
from scipy.special import softmax

SOURCE_METHOD="task_role_assignment_source_cohort_mean_total_monotone"
DISTRIBUTION_METHOD=SOURCE_METHOD+"_gp_context_CDF_product"
NATIVE_TRANSFER_METHOD=DISTRIBUTION_METHOD+"_short_native_transfer"
SELECTED_METHOD="task_source_assignment_common_baseline"
SELECTED_MODEL_STEM="lemna_task_period_source_assignment_common_baseline"
SELECTED_ABLATIONS=("no_common_baseline","no_frequency_average","no_half_gram","no_source_representation","balanced_anchor","no_shared_order")
DENSITY_NATIVE_TRANSFER_METHOD="task_density_competition_source_cohort_mean_total_monotone_gp_context_CDF_product_short_native_transfer"


def period_distribution_features(source_probability, background_probability, average):
    """Embed target and background laws by their ordered period CDFs."""
    source=np.einsum("...np,rn->...rp",source_probability,average,optimize=True)
    background=np.einsum("...np,rn->...rp",background_probability,average,optimize=True)
    return np.concatenate([source.cumsum(axis=-1),background.cumsum(axis=-1)],axis=-1)


def period_distribution_kernel(query, training, bandwidth):
    """A Gaussian kernel in the CDF Hilbert space."""
    distance=np.maximum((query**2).sum(axis=-1)[..., :,None]+(training**2).sum(axis=-1)[...,None,:]
        -2*np.matmul(query,training.swapaxes(-1,-2)),0.)
    return np.exp(-distance/(2*bandwidth[...,None,None]))


def fit_distribution_profile(context_kernel,training_cdf,query_cdf,training_run,residual):
    """Fit the fixed context-times-CDF kernel on the allowed fold-out errors."""
    distance=np.maximum((training_cdf**2).sum(axis=-1)[..., :,None]
        +(training_cdf**2).sum(axis=-1)[...,None,:]
        -2*np.matmul(training_cdf,training_cdf.swapaxes(-1,-2)),0.)
    pairs=training_run[:,:,None]&training_run[:,None,:]&~np.eye(training_run.shape[-1],dtype=bool)
    bandwidth=np.nanmedian(np.where(pairs,distance,np.nan),axis=(-1,-2))
    assert np.all(bandwidth>0)
    kernel=context_kernel*period_distribution_kernel(training_cdf,training_cdf,bandwidth)
    cross_kernel=context_kernel*period_distribution_kernel(query_cdf,training_cdf,bandwidth)
    weight=training_run/training_run.sum(axis=-1)[:,None]
    centered=np.where(training_run,residual-(residual*weight).sum(axis=-1)[...,None],0.)
    systems=kernel*training_run[:,:,None]*training_run[:,None,:]+np.eye(training_run.shape[-1])
    dual=np.linalg.solve(systems,centered[...,None])[...,0]
    delta=np.sum(cross_kernel*training_run[:,None,:]*dual[...,None,:],axis=-1)
    return {"training_cdf":training_cdf,"query_cdf":query_cdf,"bandwidth":bandwidth,
        "kernel":kernel,"cross_kernel":cross_kernel,"dual":dual,"delta":delta}


def crossfitted_distribution_features(data,inner,inner_parent,outer_probability,
        outer_background,inner_background,base_features,period):
    """Construct original 228-fold laws with every training batch held out."""
    membership=data["run_membership"]
    average=membership/membership.sum(axis=-1)[:,None]
    left,right=np.triu_indices(base_features.shape[-1])
    raw_features=np.concatenate([base_features,base_features[:,:,left]*base_features[:,:,right]],axis=-1)
    source_features=inner["source_features"][np.tile(np.arange(2),2)]
    coefficient=inner["coefficient"].reshape(4,len(inner["inner_owner"]),-1)
    score=np.einsum("...npk,...k->...np",source_features,coefficient[...,:2],optimize=True)
    score+=np.einsum("...npk,npd,...kd->...np",source_features,base_features,
        coefficient[...,2:-3].reshape(*coefficient.shape[:-1],2,base_features.shape[-1]),optimize=True)
    role_left,role_right=np.triu_indices(2)
    score+=np.einsum("...npk,...k->...np",
        source_features[...,role_left]*source_features[...,role_right],coefficient[...,-3:],optimize=True)
    score+=np.einsum("npd,gd->gnp",raw_features,inner_parent["source_coefficient"],optimize=True)[None]
    probability=softmax(score,axis=-1)
    inner_source_bags=np.einsum("...np,rn->...rp",probability,average,optimize=True)
    mean_check=np.max(abs(inner_source_bags@period-inner["inner_run_prediction"][:4]))
    assert mean_check<1e-9
    del source_features,score,probability
    inner_bg_bags=np.einsum("...np,rn->...rp",inner_background,average,optimize=True)[np.tile(np.arange(2),2)]
    owner=np.eye(len(data["outer_protocol"]))[inner["inner_owner"]]
    query=(inner["inner_query"].astype(float)@membership.T)>0
    counts=owner.T@query.astype(float)
    shape=(4,len(counts),len(membership),len(period))
    training_source=np.divide(np.einsum("kg,kr,mkrp->mgrp",owner,query,inner_source_bags,optimize=True),
        counts[None,:,:,None],out=np.zeros(shape),where=counts[None,:,:,None]>0)
    training_bg=np.divide(np.einsum("kg,kr,mkrp->mgrp",owner,query,inner_bg_bags,optimize=True),
        counts[None,:,:,None],out=np.zeros(shape),where=counts[None,:,:,None]>0)
    training_cdf=np.concatenate([training_source.cumsum(axis=-1),training_bg.cumsum(axis=-1)],axis=-1)
    query_cdf=period_distribution_features(outer_probability,outer_background,average)
    assert np.all(counts[data["outer_train_run"]]>0)
    return np.concatenate([training_cdf,training_cdf]),np.concatenate([query_cdf,query_cdf]),mean_check


SHARED_LANDMARK_MEASUREMENT="shared_landmark_renewal"


def shared_landmark_interval_readouts(event_time,event_count,event_score,time,training_source,reference,independent_unit):
    """Fixed same-information interval controls; no intervals are discarded.

    Source axes are (case,source).  Padding NaNs represent absent event slots,
    never rejected intervals. Ridge uses six standardized No-only statistics,
    source/unit-balanced weights and lambda=1/qualified training source count.
    """
    event_time=np.asarray(event_time,dtype=float)
    event_count=np.asarray(event_count,dtype=int)
    if event_time.ndim==4:
        assert event_time.shape[2]==1
        event_time=event_time[:,:,0];event_count=event_count[:,:,0]
    intervals=np.diff(event_time,axis=-1)
    mean=np.nanmean(intervals,axis=-1);median=np.nanmedian(intervals,axis=-1)
    spread=np.nanstd(intervals,axis=-1)
    valid=np.isfinite(event_time)
    position=np.searchsorted(time,np.where(valid,event_time,time[0]));position=np.clip(position,0,len(time)-1)
    peak=np.take_along_axis(np.asarray(event_score),position,axis=-1)
    peak_mean=np.divide(np.sum(np.where(valid,peak,0.),axis=-1),valid.sum(-1),out=np.full(valid.shape[:2],np.nan),where=valid.sum(-1)>0)
    features=np.stack([np.maximum(event_count-1,0),mean,median,spread,peak_mean,np.nanmean(event_score,axis=-1)],axis=-1)
    training=np.asarray(training_source,dtype=bool);reference=np.asarray(reference,dtype=float)
    units,code=np.unique(np.asarray(independent_unit,dtype=str),return_inverse=True)
    membership=(np.arange(len(units))[:,None]==code[None]).astype(float)
    number=training.astype(float)@membership.T
    weight=np.divide(training.astype(float),number[:,code],out=np.zeros_like(training,dtype=float),where=number[:,code]>0.)
    weight/=weight.sum(1)[:,None]
    count=training.sum(1);center=np.einsum("fn,fnk->fk",weight,np.where(training[:,:,None],features,0.))
    variance=np.einsum("fn,fnk->fk",weight,np.where(training[:,:,None],(features-center[:,None])**2,0.))
    scale=np.sqrt(variance);scale=np.where(scale>0.,scale,1.)
    x=(features-center[:,None])/scale[:,None]
    target=np.sum(weight*np.where(training,reference[None],0.),axis=1)
    def fit_case(case):
        take=training[case];current=x[case,take];w=weight[case,take]
        beta=np.linalg.solve((current.T*w)@current+np.eye(features.shape[-1])/count[case],
            (current.T*w)@(reference[take]-target[case]))
        return beta
    beta=np.stack(list(map(fit_case,np.arange(len(training)))))
    ridge=target[:,None]+np.einsum("fnk,fk->fn",x,beta)
    return {"interval_mean":mean,"interval_median":median,"interval_ridge":ridge,
        "interval_statistics":features,"interval_feature_mean":center,"interval_feature_scale":scale,
        "interval_ridge_coefficient":beta,"interval_ridge_intercept":target,"interval_ridge_lambda":1./count,
        "interval_feature_layout":np.asarray(["interval_count","interval_mean","interval_median","interval_sd","mean_peak_event_score","mean_time_event_score"]),
        "interval_estimate_available":np.isfinite(mean)}


def _shared_landmark_preprocess(data,period_max,measurement_count=None,feature_options=None):
    """Observed synchronous source vectors, preserving source/measurement order."""
    from source_assignment_unified import shared_background
    options={} if feature_options is None else feature_options
    time=np.asarray(data["time"],dtype=float);mask=np.asarray(data["mask"],dtype=bool)
    signal=np.where(mask,np.asarray(data["signal"],dtype=float),0.)
    owner=np.asarray(data.get("source_index",data["run_index"]),dtype=int)
    groups=np.arange(int(owner.max())+1);number=np.bincount(owner,minlength=len(groups))
    channels=int(number[0]) if measurement_count is None else int(measurement_count)
    assert np.all(number==channels), "Query measurement layout differs from the trained source topology."
    physical={"time":time,"signal":signal,"mask":mask,"count":mask.sum(1)}
    _,periodic=shared_background(physical,float(period_max),strength=(2.*np.pi)**-4,per_record=True)
    def source(index):
        rows=np.flatnonzero(owner==index);observed=mask[rows].all(0)
        value=periodic[rows][:,observed];actual=time[observed]
        if len(actual)<2:
            return value,actual,observed
        if not options.get("cross_sensor_whitening",True):
            value=value-value.mean(1)[:,None]
            rms=np.sqrt(np.mean(value**2,axis=1))
            white=np.divide(value,rms[:,None],out=np.zeros_like(value),where=rms[:,None]>0.)
            return white,actual,observed
        covariance=value@value.T/len(actual)
        eigen,vector=np.linalg.eigh(covariance)
        keep=eigen>np.finfo(float).eps*channels*eigen[-1]
        inverse=np.divide(1.,np.sqrt(np.maximum(eigen,0.)),out=np.zeros_like(eigen),where=keep)
        white=((vector*inverse[None])@vector.T)@value
        return white,actual,observed
    prepared=list(map(source,groups))
    names=np.asarray(data.get("source_names",data.get("runs",groups.astype(str))),dtype=str)
    return {"values":list(map(lambda item:item[0],prepared)),"times":list(map(lambda item:item[1],prepared)),
        "source_mask":np.stack(list(map(lambda item:item[2],prepared))),"source_index":owner,
        "source_names":names,"time":time,"measurement_count":channels,"coordinate_topology":np.asarray("synchronous_physical_sensors")}


def _shared_landmark_feature_source(values,time,shape_sigma,temporal_lags,signed=False,feature_options=None):
    """Polarity/scale-invariant source features at actual observed timestamps.

    The exact regular-grid branch is the fitted scipy Gaussian operator.
    Irregular observations use its reflected Gaussian quadrature; no missing
    timestamp is added as a training sample or as a detected event.
    """
    from scipy.ndimage import gaussian_filter1d
    options={} if feature_options is None else feature_options
    channels,count=values.shape
    energy_dimensions=3 if options.get("derivative_curvature_energy",True) else 1
    if count<2:
        return np.empty((count,len(temporal_lags)*((5 if signed else 2)*channels+energy_dimensions)),dtype=np.float32)
    step=np.diff(time);cadence=float(np.median(step));sigma=float(shape_sigma)/cadence
    regular=np.allclose(step,cadence,rtol=1e-9,atol=cadence*1e-9)
    if regular:
        derivative=gaussian_filter1d(values,sigma=sigma,order=1,axis=-1,mode="reflect")*sigma
        curvature=gaussian_filter1d(values,sigma=sigma,order=2,axis=-1,mode="reflect")*sigma**2
    else:
        left=time[0]-step[0]/2.;right=time[-1]+step[-1]/2.
        extended=np.r_[2.*left-time[::-1],time,2.*right-time[::-1]]
        y=np.concatenate([values[:,::-1],values,values[:,::-1]],axis=1)
        delta=np.diff(extended);quadrature=np.r_[delta[0],(delta[:-1]+delta[1:])/2.,delta[-1]]
        first=np.searchsorted(extended,time-4.*shape_sigma);last=np.searchsorted(extended,time+4.*shape_sigma,side="right")
        index=first[:,None]+np.arange(int(np.max(last-first)))[None]
        valid=index<last[:,None];index=np.minimum(index,len(extended)-1)
        x=(time[:,None]-extended[index])/shape_sigma
        weight=np.exp(-.5*x*x)*quadrature[index]*valid
        denominator=weight.sum(1)
        derivative=np.einsum("tk,ctk->ct",-x*weight,y[:,index],optimize=True)/denominator[None]
        curvature=np.einsum("tk,ctk->ct",(x*x-1.)*weight,y[:,index],optimize=True)/denominator[None]
    energy=np.stack([np.sum(values**2,axis=0),np.sum(derivative**2,axis=0),np.sum(curvature**2,axis=0)],axis=-1)
    if not options.get("derivative_curvature_energy",True):
        energy=energy[:,:1]
    base=np.concatenate([values.T,abs(values).T,(values**2).T,derivative.T,curvature.T,energy],axis=-1).astype(np.float32) if signed else np.concatenate([abs(values).T,(values**2).T,energy],axis=-1).astype(np.float32)
    samples=np.rint(np.asarray(temporal_lags)/cadence).astype(int)
    exact=regular and np.allclose(np.asarray(temporal_lags)/cadence,samples,rtol=0.,atol=1e-8)
    def lagged(item):
        lag,offset=item
        if exact:
            return base[np.clip(np.arange(count)+offset,0,count-1)]
        return np.stack(list(map(lambda column:np.interp(time+lag,time,base[:,column]),np.arange(base.shape[1]))),axis=-1).astype(np.float32)
    return np.concatenate(list(map(lagged,zip(temporal_lags,samples))),axis=-1)


def shared_landmark_event_features(prepared,shape_sigma,temporal_lags,feature_options=None):
    """Learned physical lags of absolute/square and spatial energies.

    Normalization is already defined by prepared/model. Absent options retain
    all temporal and derivative/curvature energy fields.
    """
    return list(map(lambda pair:_shared_landmark_feature_source(pair[0],pair[1],shape_sigma,temporal_lags,feature_options=feature_options),
        zip(prepared["values"],prepared["times"])))


def _shared_landmark_events(score,time,minimum_period):
    from scipy.signal import find_peaks
    if len(time)<2:
        return np.asarray([],dtype=int)
    gap=np.diff(time);cadence=float(np.median(gap))
    if np.allclose(gap,cadence,rtol=1e-9,atol=cadence*1e-9):
        return find_peaks(score,distance=max(1,int(np.ceil(minimum_period/cadence))),height=.5)[0]
    candidates=find_peaks(score,height=.5)[0]
    active=np.ones(len(candidates),dtype=bool)
    def select(index):
        if active[index]:
            active[(abs(time[candidates]-time[candidates[index]])<minimum_period)&(np.arange(len(candidates))!=index)]=False
        return None
    list(map(select,np.argsort(score[candidates],kind="stable")[::-1]))
    return candidates[active]


def shared_landmark_renewal_log_likelihood(intervals,period,rr_variance,timing_variance,precision,recall,duration,missed_clutter=True):
    """One mean-period parameter from every interval, with fitted missed/clutter rates."""
    from scipy.special import logsumexp
    intervals=np.asarray(intervals,dtype=float);period=np.asarray(period,dtype=float)
    if not len(intervals):
        return np.zeros(len(period))
    if not missed_clutter:
        variance=float(rr_variance)+2.*float(timing_variance)
        return (-.5*((intervals[:,None]-period[None])**2/variance+np.log(2.*np.pi*variance))).sum(0)
    cycles=np.arange(1,int(np.floor(duration/period.min()))+1,dtype=float)
    recall=float(np.clip(recall,0.,1.));precision=float(np.clip(precision,0.,1.))
    geometric=recall*np.power(1.-recall,cycles-1.)
    geometric=np.divide(geometric,geometric.sum(),out=np.full(len(cycles),1./len(cycles)),where=geometric.sum()>0.)
    variance=cycles*float(rr_variance)+2.*float(timing_variance)
    log_weight=np.log(np.maximum(geometric,np.finfo(float).tiny))
    log_normal=-.5*((intervals[:,None,None]-cycles[None,:,None]*period[None,None])**2/variance[None,:,None]+np.log(2.*np.pi*variance)[None,:,None])
    real=logsumexp(log_normal+log_weight[None,:,None],axis=1)
    pair=precision**2
    log_real=np.log(pair) if pair>0. else -np.inf
    log_clutter=np.log1p(-pair)-np.log(duration) if pair<1. else -np.inf
    return np.logaddexp(log_real+real,log_clutter).sum(0)


def _shared_landmark_unit_seconds(unit):
    return {"s":1.,"second":1.,"seconds":1.,"ms":.001,"millisecond":.001,"h":3600.,"hour":3600.,"hours":3600.,"min":60.,"minute":60.}[str(unit)]


def predict_shared_landmark_renewal(data,model,*,detector_path=None,case_indices=None,include_event_score=False):
    """No-only prediction: no reference waveform, event landmarks or query labels.

    Nc is the saved source measurement layout. Every measurement of the same
    source receives the same complete period law. Input time_unit may differ
    physically from the fitted unit; output periods use the input unit.
    """
    import joblib,json
    feature_options=json.loads(str(model.get("feature_options_json","{}")))
    measurement_options=json.loads(str(model.get("measurement_options_json","{}")))
    assert str(model["measurement_kind"])==SHARED_LANDMARK_MEASUREMENT
    input_unit=str(data.get("time_unit",model["time_unit"]))
    factor=_shared_landmark_unit_seconds(input_unit)/_shared_landmark_unit_seconds(model["time_unit"])
    physical=dict(data,time=np.asarray(data["time"],dtype=float)*factor)
    period=np.asarray(model["period"],dtype=float)
    prepared=_shared_landmark_preprocess(physical,period.max(),int(model["measurement_channel_count"]),feature_options=feature_options)
    saved=joblib.load(str(detector_path or model["detector_joblib_path"]))
    detectors=saved["models"] if isinstance(saved,dict) else saved
    model_cases=np.asarray(model.get("case_index",np.arange(len(detectors))),dtype=int)
    selected=model_cases if case_indices is None else np.asarray(case_indices,dtype=int)
    positions=np.asarray([np.flatnonzero(model_cases==case).item() for case in selected])
    sources=len(prepared["times"]);total_time=len(prepared["time"])
    max_events=max(1,int(np.ceil((prepared["time"][-1]-prepared["time"][0])/period.min()))+2)
    likelihood=np.empty((len(selected),sources,len(period)))
    event_time=np.full((len(selected),sources,max_events),np.nan);event_count=np.zeros((len(selected),sources),dtype=int)
    event_score=np.full((len(selected),sources,total_time),np.nan,dtype=np.float32)
    def one_case(position):
        current=int(positions[position]);case=int(selected[position]);detector=detectors[current]
        features=shared_landmark_event_features(prepared,float(model["learned_QRS_width"][current]),model["learned_temporal_lags"][current],feature_options=feature_options)
        number=np.asarray(list(map(len,features)));joined=np.concatenate(features,axis=0)
        predicted=detector.predict_proba(joined)[:,1].astype(np.float32)
        scores=np.split(predicted,np.cumsum(number)[:-1])
        def source(index):
            time=prepared["times"][index];score=scores[index]
            event_score[position,index,prepared["source_mask"][index]]=score
            peaks=_shared_landmark_events(score,time,float(period.min()))
            moments=time[peaks];event_count[position,index]=len(moments);event_time[position,index,:len(moments)]=moments
            duration=(time[-1]-time[0]+float(np.median(np.diff(time)))) if len(time)>1 else float(model["training_window_duration"])
            likelihood[position,index]=shared_landmark_renewal_log_likelihood(np.diff(moments),period,
                float(model["renewal_RR_variance"][current]),float(model["renewal_timing_variance"][current]),
                float(model["renewal_detector_precision"][current]),float(model["renewal_detector_recall"][current]),duration,
                missed_clutter=measurement_options.get("missed_clutter",True))
            return None
        list(map(source,np.arange(sources)))
        return None
    list(map(one_case,np.arange(len(selected))))
    prior=np.asarray(model["foreground_log_prior"])[positions]
    temperature=np.asarray(model["renewal_temperature"])[positions]
    probability=softmax(temperature[:,None,None]*likelihood+prior[:,None],axis=-1)
    output_period=period/factor;point=probability@output_period;owner=prepared["source_index"]
    cohort=np.asarray(data.get("cohort_index",data["run_index"]),dtype=int)
    member=(np.arange(int(cohort.max())+1)[:,None]==cohort[None]).astype(float)
    average=member/member.sum(1)[:,None]
    mean=point[:,owner]@average.T
    cohort_probability=np.einsum("fnp,gn->fgp",probability[:,owner],average,optimize=True)
    result={"posterior":probability[:,owner],"source_probability":probability,"source_period":point,
        "estimate":point[:,owner],"group_mean":mean,"cohort_probability":cohort_probability,
        "deviation":point[:,owner]-mean[:,cohort],
        "period":output_period,"case_index":selected,"source_id":prepared["source_names"],"source_index":owner,"run_index":cohort,
        "source_log_likelihood":likelihood,"source_event_time":event_time/factor,"source_event_count":event_count,
        "source_likelihood_available":event_count>=2,"time_unit":np.asarray(input_unit)}
    if "interval_ridge_coefficient" in model:
        # Control readouts use the same event stream; they do not alter its law.
        interval=np.diff(event_time,axis=-1)
        valid=np.isfinite(event_time);idx=np.searchsorted(prepared["time"],np.where(valid,event_time,prepared["time"][0]));idx=np.clip(idx,0,total_time-1)
        peak=np.take_along_axis(event_score,idx,axis=-1)
        peak_mean=np.divide(np.sum(np.where(valid,peak,0.),axis=-1),valid.sum(-1),out=np.full(valid.shape[:2],np.nan),where=valid.sum(-1)>0)
        features=np.stack([np.maximum(event_count-1,0),np.nanmean(interval,axis=-1),np.nanmedian(interval,axis=-1),np.nanstd(interval,axis=-1),peak_mean,np.nanmean(event_score,axis=-1)],axis=-1)
        standardized=(features-np.asarray(model["interval_feature_mean"])[positions,None])/np.asarray(model["interval_feature_scale"])[positions,None]
        ridge=np.asarray(model["interval_ridge_intercept"])[positions,None]+np.einsum("fnk,fk->fn",standardized,np.asarray(model["interval_ridge_coefficient"])[positions])
        result.update(interval_mean=np.nanmean(interval,axis=-1)/factor,interval_median=np.nanmedian(interval,axis=-1)/factor,interval_ridge=ridge/factor)
    if include_event_score:
        result["event_score"]=event_score
    return result


def fit_shared_landmark_renewal(data,reference,qualified,training_source,independent_unit,
        reference_waveform,reference_landmarks,period,*,reference_intervals=None,time_unit="s",
        detector_path,case_indices=None,case_callback=None,feature_options=None,measurement_options=None):
    """Fit the fixed shared-source landmark detector and renewal period law.

    reference/qualified/independent_unit are source-level arrays (S,).
    training_source is (F,S). reference_waveform is (S,T), with NaN outside
    observations; landmark lists have source-specific physical event times.
    reference_intervals may retain boundary-owned intervals from the original
    mean-period definition. Only allowed training sources enter supervision.
    All observed training timestamp vectors are used, including full binning.
    feature_options: cross_sensor_whitening, temporal_neighborhood and
    derivative_curvature_energy. The default retains sensor coordinates via
    independent center/RMS (cross_sensor_whitening=False); both temporal and
    derivative/curvature features remain enabled. measurement_options:
    missed_clutter=True and foreground_prior="reference". Saved options are
    used by the same predictor; these ablations have shared-source scope.
    """
    from pathlib import Path
    import joblib,json
    feature_options={"cross_sensor_whitening":False,"temporal_neighborhood":True,"derivative_curvature_energy":True}|({} if feature_options is None else dict(feature_options))
    measurement_options={"missed_clutter":True,"foreground_prior":"reference"}|({} if measurement_options is None else dict(measurement_options))
    lag_factors=np.arange(-2,3) if feature_options.get("temporal_neighborhood",True) else np.asarray([0])
    from scipy.fft import rfft,irfft,next_fast_len
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from sklearn.ensemble import HistGradientBoostingClassifier
    from source_assignment_unified import shared_background,period_reference_mass
    import time as clock
    period=np.asarray(period,dtype=float);reference=np.asarray(reference,dtype=float)
    qualified=np.asarray(qualified,dtype=bool);training=np.asarray(training_source,dtype=bool)
    if training.ndim==1:
        training=training[None]
    cases=np.arange(len(training)) if case_indices is None else np.asarray(case_indices,dtype=int)
    train=training[cases]&qualified[None]
    units,code=np.unique(np.asarray(independent_unit,dtype=str),return_inverse=True)
    member=(np.arange(len(units))[:,None]==code[None]).astype(float)
    number=train.astype(float)@member.T
    weights=np.divide(train.astype(float),number[:,code],out=np.zeros_like(train,dtype=float),where=number[:,code]>0.)
    weights/=weights.sum(1)[:,None]
    prepared=_shared_landmark_preprocess(data,period.max(),feature_options=feature_options);sources=len(prepared["times"])
    channels=int(prepared["measurement_count"]);time=np.asarray(data["time"],dtype=float)
    cadence=float(np.median(np.diff(time)));duration=float(time[-1]-time[0]+cadence)
    foreground=np.asarray(reference_waveform,dtype=float);fgmask=np.isfinite(foreground)
    _,foreground=shared_background({"time":time,"signal":np.where(fgmask,foreground,0.),"mask":fgmask,"count":fgmask.sum(1)},period.max(),strength=(2.*np.pi)**-4,per_record=True)
    half=max(1,int(np.floor(period.min()/(2.*cadence))));offset=np.arange(-half,half+1)
    count=train.sum(1);models=[];width=np.empty(len(cases));lags=np.empty((len(cases),len(lag_factors)))
    rrvar=np.empty(len(cases));timingvar=np.empty(len(cases));precision=np.empty(len(cases));recall=np.empty(len(cases))
    positive_fraction=np.empty(len(cases));timestamps=np.empty(len(cases),dtype=int)
    positive_tolerance=np.empty(len(cases));template=np.empty((len(cases),len(offset)))
    curves=np.empty((len(cases),sources,len(period)))
    max_events=max(1,int(np.ceil(duration/period.min()))+2)
    times=np.full((len(cases),sources,max_events),np.nan);event_count=np.zeros((len(cases),sources),dtype=int)
    event_score=np.full((len(cases),sources,len(time)),np.nan,dtype=np.float32)
    fit_seconds=np.empty(len(cases));path=Path(detector_path).resolve()
    energy_layout=["space_energy","derivative_energy","curvature_energy"] if feature_options.get("derivative_curvature_energy",True) else ["space_energy"]
    layout=np.asarray([part+str(channel) for part in ["abs_lead_","square_lead_"] for channel in range(channels)]+energy_layout)
    def one_case(position):
        started=clock.perf_counter();take=np.flatnonzero(train[position]);shapes=[];scatter=[]
        def teacher(source):
            landmarks=np.asarray(reference_landmarks[source],dtype=float)
            samples=np.rint((landmarks-time[0])/cadence).astype(int)
            inside=samples[(samples>=half)&(samples<len(time)-half)]
            current=foreground[source,inside[:,None]+offset[None]];current-=current.mean(1)[:,None]
            peak=np.argmax(abs(current),axis=1);current*=np.sign(current[np.arange(len(current)),peak])[:,None]
            norm=np.sqrt(np.sum(current**2,axis=1))
            shapes.append(np.divide(current,norm[:,None],out=np.zeros_like(current),where=norm[:,None]>0.))
            intervals=np.diff(landmarks) if reference_intervals is None else np.asarray(reference_intervals[source],dtype=float)
            scatter.append(intervals-reference[source])
            return None
        list(map(teacher,take))
        kernel=np.median(np.concatenate(shapes,axis=0),axis=0);kernel-=kernel.mean();kernel/=np.sqrt(np.sum(kernel**2))
        spectrum=rfft(kernel,n=next_fast_len(2*len(kernel)-1));ac=irfft(spectrum*spectrum.conj(),n=next_fast_len(2*len(kernel)-1))[:half+1];ac/=ac[0]
        crossing=np.flatnonzero(ac<=0.);end=int(crossing[0]) if len(crossing) else len(ac)
        sigma=max(1.,float(np.sum(ac[:end])/np.sqrt(np.pi)))
        width[position]=sigma*cadence;lags[position]=np.rint(lag_factors*sigma)*cadence;template[position]=kernel
        match_width=min(half,max(1,end))*cadence;positive_tolerance[position]=np.ceil(sigma)*cadence
        rrvar[position]=float(np.mean(np.concatenate(scatter)**2))
        features=shared_landmark_event_features(prepared,width[position],lags[position],feature_options=feature_options)
        def source_labels(source):
            actual=prepared["times"][source];landmarks=np.asarray(reference_landmarks[source],dtype=float)
            # Regular data recover the exact original integer label neighborhood.
            if np.allclose(np.diff(actual),cadence,rtol=1e-9,atol=cadence*1e-9):
                indices=np.rint((landmarks-actual[0])/cadence).astype(int)[:,None]+np.arange(-int(np.ceil(sigma)),int(np.ceil(sigma))+1)[None]
                valid=(indices>=0)&(indices<len(actual));label=np.zeros(len(actual),dtype=bool);label[indices[valid]]=True
                return label
            distance=np.min(abs(actual[:,None]-landmarks[None]),axis=1)
            return distance<=positive_tolerance[position]
        labels=list(map(source_labels,take));length=np.asarray(list(map(len,labels)))
        source_weight=weights[position,take]
        timestamps[position]=int(length.sum())
        if np.all(length==length[0]):
            dense=np.stack(labels)
            positive_fraction[position]=float(np.sum(source_weight[:,None]*dense)/length[0])
            sample_weight=(source_weight[:,None]*np.where(dense,.5/positive_fraction[position],.5/(1.-positive_fraction[position]))*len(take)).ravel()
        else:
            positive_fraction[position]=sum(map(lambda pair:pair[0]*np.mean(pair[1]),zip(source_weight,labels)))
            def point_weight(item):
                source,index=item;y=labels[index];weight=source_weight[index]/len(y)
                return weight*np.where(y,.5/positive_fraction[position],.5/(1.-positive_fraction[position]))
            sample_weight=np.concatenate(list(map(point_weight,zip(take,np.arange(len(take))))))*timestamps[position]
        x=np.concatenate(list(map(lambda source:features[source],take)),axis=0);y=np.concatenate(labels)
        estimator=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,early_stopping=False,random_state=12345)
        original=estimator._bin_data
        def bin_all(values,is_training_data):
            if is_training_data:
                estimator._bin_mapper.subsample=None
            return original(values,is_training_data)
        estimator._bin_data=bin_all
        estimator.fit(x,y,sample_weight=sample_weight);del estimator._bin_data
        assert estimator._bin_mapper.subsample is None
        models.append(estimator)
        joined=np.concatenate(features,axis=0);size=np.asarray(list(map(len,features)))
        probability=estimator.predict_proba(joined)[:,1].astype(np.float32)
        scores=np.split(probability,np.cumsum(size)[:-1]);errors=[];matched=0;detected=0;true=0
        moments=[]
        def detect_source(source):
            actual=prepared["times"][source];score=scores[source]
            event_score[position,source,prepared["source_mask"][source]]=score
            peaks=_shared_landmark_events(score,actual,float(period.min()));current=actual[peaks]
            times[position,source,:len(current)]=current;event_count[position,source]=len(current);moments.append(current)
            if train[position,source]:
                teacher=np.asarray(reference_landmarks[source],dtype=float);right=np.searchsorted(teacher,current)
                first=current-teacher[np.clip(right-1,0,len(teacher)-1)];second=current-teacher[np.clip(right,0,len(teacher)-1)]
                error=np.where(abs(first)<=abs(second),first,second);valid=abs(error)<=match_width
                errors.append(error[valid])
                return int(valid.sum()),len(current),len(teacher)
            return 0,0,0
        numbers=np.asarray(list(map(detect_source,np.arange(sources))));matched,detected,true=numbers.sum(0)
        error=np.concatenate(errors)
        timingvar[position]=float(np.mean((error-error.mean())**2)) if len(error)>1 else float((period.max()-period.min())**2/12.)
        precision[position]=matched/detected if detected else 0.;recall[position]=matched/true if true else 0.
        def source_likelihood(source):
            actual=prepared["times"][source]
            exposure=float(actual[-1]-actual[0]+np.median(np.diff(actual))) if len(actual)>1 else duration
            return shared_landmark_renewal_log_likelihood(np.diff(moments[source]),period,rrvar[position],timingvar[position],precision[position],recall[position],exposure,missed_clutter=measurement_options.get("missed_clutter",True))
        curves[position]=np.stack(list(map(source_likelihood,np.arange(sources))))
        fit_seconds[position]=clock.perf_counter()-started
        joblib.dump({"models":models,"case_index":cases[:position+1],"measurement_kind":SHARED_LANDMARK_MEASUREMENT,
            "binning_all_timestamps":True,"reference_source_count":count[:position+1],"shape_sigma_s":width[:position+1]},path)
        print("SHARED_LANDMARK_FIT",int(cases[position]),"N",int(count[position]),"G",int((number[position]>0).sum()),"timestamps",int(timestamps[position]),"precision",precision[position],"recall",recall[position],"seconds",fit_seconds[position],flush=True)
        if case_callback is not None:
            case_callback(int(cases[position]),{"case_index":cases[:position+1],"train_reference_source_count":count[:position+1],"detector_joblib_path":np.asarray(str(path)),"fit_stage":np.asarray("detector_and_rates")})
        return None
    list(map(one_case,np.arange(len(cases))))
    mean=weights@np.where(qualified,reference,0.)
    sd=np.sqrt(np.maximum(weights@np.where(qualified,reference**2,0.)-mean**2,(period[1]-period[0])**2/12.))
    prior=-.5*((period[None]-mean[:,None])/sd[:,None])**2;prior-=logsumexp(prior,axis=1,keepdims=True)
    if measurement_options.get("foreground_prior","reference")=="uniform":
        prior=np.full_like(prior,-np.log(len(period)))
    teacher=period_reference_mass(period,reference,qualified)
    objective=lambda theta:(beta:=np.exp(theta),logits:=beta[:,None,None]*curves+prior[:,None],logp:=logits-logsumexp(logits,axis=-1,keepdims=True),
        (-(weights[:,:,None]*teacher[None]*logp).sum()+.5*np.sum(theta**2/count),beta*np.einsum("fn,fnp,fnp->f",weights,np.exp(logp)-teacher[None],curves)+theta/count))[-1]
    optimized=minimize(objective,np.zeros(len(cases)),jac=True,method="L-BFGS-B",options={"maxiter":300,"ftol":1e-10,"gtol":1e-7})
    source_probability=softmax(np.exp(optimized.x)[:,None,None]*curves+prior[:,None],axis=-1)
    model={"schema":np.asarray("source_assignment_unified_0.3.1"),"measurement_kind":np.asarray(SHARED_LANDMARK_MEASUREMENT),
        "readout_kind":np.asarray("shared_source_renewal_period_posterior"),"case_index":cases,"period":period,
        "period_bounds":np.asarray([period.min(),period.max()]),"time_unit":np.asarray(time_unit),
        "measurement_channel_count":np.asarray(channels),"source_grouping":np.asarray(data.get("source_grouping","run")),
        "coordinate_topology":np.asarray("synchronous_physical_sensors"),
        "No_source_topology":np.asarray("synchronous_measurements_share_one_landmark_stream"),
        "measurement_layout":np.asarray(["measurement_"+str(index) for index in range(channels)]),
        "base_feature_layout":layout,"feature_layout":np.asarray(["lag_"+str(lag)+":"+name for lag in range(len(lag_factors)) for name in layout]),
        "learned_temporal_lags":lags,"learned_QRS_width":width,"foreground_event_template":template,
        "positive_event_tolerance":positive_tolerance,"class_positive_prior":positive_fraction,
        "class_weight_positive":.5/positive_fraction,"class_weight_negative":.5/(1.-positive_fraction),
        "training_timestamp_count":timestamps,"train_reference_source_count":count,"train_reference_unit_count":(number>0).sum(1),
        "training_source":training[cases],"training_reference_access_mask":train,"source_id":prepared["source_names"],
        "training_independent_unit":np.asarray(independent_unit,dtype=str),"training_source_weight":weights,
        "detector_class_weight_scope":np.asarray("unit_and_source_balanced_then_positive_negative_balanced"),
        "renewal_RR_variance":rrvar,"renewal_timing_variance":timingvar,"renewal_detector_precision":precision,
        "renewal_detector_recall":recall,"foreground_log_prior":prior,"foreground_reference_mean":mean,"foreground_reference_sd":sd,
        "renewal_temperature":np.exp(optimized.x),"training_window_duration":np.asarray(duration),"training_time_step":np.asarray(cadence),
        "detector_joblib_path":np.asarray(str(path)),"binning_all_timestamps":np.asarray(True),"seed":np.asarray(12345),
        "training_supervision":np.asarray("No_timestamp_event_labels_from_training_landmarks_and_reference_waveform_shape"),
        "detector_configuration":np.asarray("HistGradientBoostingClassifier max_iter100 max_leaf_nodes15 early_stoppingFalse full_timestamp_binning"),
        "detector_rate_calibration":np.asarray("in_sample_allowed_training_sources"),
        "feature_options_json":np.asarray(json.dumps(feature_options,sort_keys=True)),"measurement_options_json":np.asarray(json.dumps(measurement_options,sort_keys=True)),
        "ablation_field_scope":np.asarray("shared-source"),"fit_seconds":fit_seconds,"success":np.asarray(optimized.success),
        "source_probability":source_probability,"source_period":source_probability@period,"source_log_likelihood":curves}
    controls=shared_landmark_interval_readouts(times,event_count,event_score,time,train,reference,independent_unit);model.update(controls)
    return model


def shared_landmark_observations(frame,*,source_grouping="run",time_unit="s"):
    """No-only long table to actual-time/mask source measurements, including Nc=1."""
    import pandas as pd
    wide=frame.pivot(index="trace_id",columns="x",values="signal_y").sort_index(axis=1)
    columns=["trace_id","run","condition"]
    metadata=frame[columns].drop_duplicates().set_index("trace_id").loc[wide.index]
    cohort,runs=pd.factorize(metadata.run,sort=True)
    source_key=metadata.index.to_numpy(dtype=str) if source_grouping=="trace_id" else metadata[source_grouping].to_numpy(dtype=str)
    owner,names=pd.factorize(source_key,sort=True)
    membership=(np.arange(len(runs))[:,None]==cohort[None]).astype(float)
    mask=wide.notna().to_numpy()
    return {"time":wide.columns.to_numpy(dtype=float),"signal":wide.fillna(0.).to_numpy(dtype=float),"mask":mask,
        "count":mask.sum(1),"metadata":metadata,"run_index":cohort,"runs":np.asarray(runs,dtype=str),
        "membership":membership,"average":membership/membership.sum(1)[:,None],
        "source_index":owner,"source_names":np.asarray(names,dtype=str),"source_grouping":source_grouping,"time_unit":time_unit}


def load_shared_landmark_renewal_model(model_path):
    """Load the portable NPZ and resolve its detector path beside the model."""
    from pathlib import Path
    path=Path(model_path).resolve();model=dict(np.load(path,allow_pickle=False))
    detector=Path(str(model["detector_joblib_path"]))
    model["detector_joblib_path"]=np.asarray(str(detector if detector.is_absolute() else path.parent/detector))
    return model


def save_shared_landmark_renewal_model(model,model_path):
    """Save array metadata with a relative detector artifact path."""
    from pathlib import Path
    import os
    path=Path(model_path).resolve();saved=dict(model);detector=Path(str(saved["detector_joblib_path"]))
    detector=detector if detector.is_absolute() else path.parent/detector
    saved["detector_joblib_path"]=np.asarray(Path(os.path.relpath(detector,path.parent)).as_posix())
    np.savez_compressed(path,**saved)
    return path


def reference_phase_temporal_features(data,source_bundle,period_scale):
    """Four No-only virtual coordinates; no additional physical source."""
    from scipy.signal import hilbert
    from source_assignment_unified import detrend
    time=np.asarray(data["time"],dtype=float);mask=np.asarray(data["mask"],dtype=bool)
    raw=detrend(data["signal"],time,mask)
    rms=np.sqrt(np.sum(raw**2,axis=1)/mask.sum(1));raw/=rms[:,None]
    peer=np.asarray(source_bundle["physical_peer"],dtype=float)
    balanced=np.asarray(source_bundle["source_unit"],dtype=float)
    sigma=float(period_scale)/16.;lags=np.arange(-2,3)*sigma
    def record(index):
        observed=mask[index];actual=time[observed];p=peer[index,observed]
        step=np.diff(actual)
        if np.allclose(step,np.median(step),rtol=1e-9,atol=np.median(step)*1e-9):
            quadrature=hilbert(p).imag
        else:
            t=(actual-actual[0])/(actual[-1]-actual[0])
            modes=min((len(actual)-1)//2,max(1,int(np.ceil(2.*(actual[-1]-actual[0])/float(period_scale)))))
            angle=2.*np.pi*t[:,None]*np.arange(1,modes+1)[None]
            cosine,sine=np.cos(angle),np.sin(angle);design=np.column_stack([cosine,sine]);design-=design.mean(0)
            coefficient=np.linalg.lstsq(design,p,rcond=None)[0]
            quadrature=coefficient[:modes]@sine.T-coefficient[modes:]@cosine.T
        values=np.stack([raw[index,observed],p,quadrature,balanced[index,observed]])
        return _shared_landmark_feature_source(values,actual,sigma,lags,signed=True)
    features=list(map(record,np.arange(len(mask))))
    return {"features":features,"mask":mask,"time":time,"shape_sigma":np.asarray(sigma),"temporal_lags":lags,
        "feature_definition":np.asarray("rawNo_physicalPeer_peerQuadrature_balancedNo; signed_abs_square_derivative_curvature_energy; five_relative_lags"),
        "virtual_coordinate_count":np.asarray(4),"coordinate_topology":np.asarray("virtual_transform_basis")}


def reference_phase_cosine_teacher(components,origin_trace_id,view_id,period_scale):
    """Signed H1 source phase from measured references and declared time maps."""
    from source_assignment_unified import detrend
    c=dict(components);time=np.asarray(c["time"],dtype=float)
    wave=np.asarray(c["foreground_source_waveform"][0],dtype=float);mask=c["source_mask"]
    period=np.asarray(c["source_reference_period"],dtype=float)
    period=np.where(np.isfinite(period),period,np.asarray(c["extension_period"],dtype=float))
    if len(period)!=len(wave):
        first=np.zeros(len(wave),dtype=int)
        first[c["source_index"]]=np.arange(len(c["source_index"]))
        period=period[first]
    angle=2.*np.pi*time[None]/period[:,None]
    design=np.stack([np.cos(angle),np.sin(angle)],axis=-1)*mask[:,:,None]
    design=np.stack([detrend(design[:,:,i],time,mask) for i in range(2)],axis=-1)
    beta=np.linalg.solve(np.einsum("nti,ntj->nij",design,design),np.einsum("nti,nt->ni",design,wave)[:,:,None])[:,:,0]
    phase=np.arctan2(beta[:,1],beta[:,0])
    import pandas as pd
    origin=pd.Index(c["trace_id"]).get_indexer(origin_trace_id)
    view=pd.Index(c["candidate_view_id"]).get_indexer(view_id)
    assert np.all(origin>=0) and np.all(view>=0)
    source=c["source_index"][origin]
    basis=c["view_foreground_index"][view]
    donor=c["foreground_basis_source_index"][basis,source]
    scale=c["foreground_time_scale"][view,origin];shift=c["foreground_time_shift"][view,origin]
    query=time[0]+scale[:,None]*(time[None]-time[0])+shift[:,None]
    sign=np.sign(c["foreground_loading"][origin]*c["foreground_gain"][view,origin])
    teacher=sign[:,None]*np.cos(2.*np.pi*query/period[donor,None]-phase[donor,None])
    return teacher


def fit_reference_phase_learner(prepared,teacher,training_weight,reference_count,*,learner_kind="hgb"):
    """Fixed HGB or same-feature standardized ridge, using all weighted times."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    features=prepared["features"];mask=prepared["mask"]
    selected=np.flatnonzero(np.asarray(training_weight)>0.)
    x=np.concatenate([features[i] for i in selected])
    y=np.concatenate([teacher[i,mask[i]] for i in selected])
    point_weight=np.concatenate([np.full(len(features[i]),training_weight[i]/len(features[i])) for i in selected])
    point_weight/=point_weight.sum()
    if learner_kind=="hgb":
        estimator=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,early_stopping=False,random_state=12345)
        original=estimator._bin_data
        def all_binning(values,is_training_data):
            if is_training_data:
                estimator._bin_mapper.subsample=None
            return original(values,is_training_data)
        estimator._bin_data=all_binning
        estimator.fit(x,y,sample_weight=point_weight*len(point_weight));del estimator._bin_data
        assert estimator._bin_mapper.subsample is None
        model={"kind":"hgb","estimator":estimator}
    else:
        mean=point_weight@x;variance=point_weight@(x*x)-mean**2
        scale=np.sqrt(np.maximum(variance,0.));scale=np.where(scale>0.,scale,1.)
        z=(x-mean)/scale;center=float(point_weight@y)
        covariance=(z.T*point_weight)@z
        coefficient=np.linalg.solve(covariance+np.eye(x.shape[1])/float(reference_count),z.T@(point_weight*(y-center)))
        model={"kind":"ridge","mean":mean,"scale":scale,"coefficient":coefficient,"intercept":center}
    model.update({"feature_count":x.shape[1],"training_timestamps":len(y),"reference_source_count":int(reference_count),
        "full_binning":True,"shape_sigma":float(prepared["shape_sigma"]),"temporal_lags":prepared["temporal_lags"]})
    return model


def predict_reference_phase_learner(prepared,model):
    """No-only source phase estimates at every actual observed timestamp."""
    joined=np.concatenate(prepared["features"]);size=np.asarray([len(x) for x in prepared["features"]])
    if model["kind"]=="hgb":
        prediction=model["estimator"].predict(joined)
    else:
        prediction=((joined-model["mean"])/model["scale"])@model["coefficient"]+model["intercept"]
    values=np.zeros(prepared["mask"].shape,dtype=float)
    parts=np.split(prediction,np.cumsum(size)[:-1])
    for i in range(len(parts)):
        values[i,prepared["mask"][i]]=parts[i]
    return values


def calibrate_shared_landmark_temperature(source_log_likelihood,period,reference,qualified,training_source,independent_unit,foreground_log_prior):
    """The fixed proper-score temperature for cached same-detector likelihoods."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from source_assignment_unified import period_reference_mass
    curves=np.asarray(source_log_likelihood,dtype=float);period=np.asarray(period,dtype=float)
    reference=np.asarray(reference,dtype=float);qualified=np.asarray(qualified,dtype=bool)
    train=np.asarray(training_source,dtype=bool)&qualified[None]
    units,code=np.unique(np.asarray(independent_unit,dtype=str),return_inverse=True)
    member=(np.arange(len(units))[:,None]==code[None]).astype(float);number=train.astype(float)@member.T
    weight=np.divide(train.astype(float),number[:,code],out=np.zeros_like(train,dtype=float),where=number[:,code]>0.)
    weight/=weight.sum(1)[:,None];count=train.sum(1)
    prior=np.asarray(foreground_log_prior,dtype=float);teacher=period_reference_mass(period,reference,qualified)
    objective=lambda theta:(beta:=np.exp(theta),logits:=beta[:,None,None]*curves+prior[:,None],logp:=logits-logsumexp(logits,axis=-1,keepdims=True),
        (-(weight[:,:,None]*teacher[None]*logp).sum()+.5*np.sum(theta**2/count),beta*np.einsum("fn,fnp,fnp->f",weight,np.exp(logp)-teacher[None],curves)+theta/count))[-1]
    fit=minimize(objective,np.zeros(len(train)),jac=True,method="L-BFGS-B",options={"maxiter":300,"ftol":1e-10,"gtol":1e-7})
    probability=softmax(np.exp(fit.x)[:,None,None]*curves+prior[:,None],axis=-1)
    return {"renewal_temperature":np.exp(fit.x),"source_probability":probability,"source_period":probability@period,
        "source_log_likelihood":curves,"foreground_log_prior":prior,"success":np.asarray(fit.success)}
