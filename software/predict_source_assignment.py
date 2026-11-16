"""Predict target periods from observations using a fitted model archive."""
from pathlib import Path
import os,argparse,sys
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1],"model":Path(__file__).resolve().parents[1]/"results/0.3.1/unified_model.npz"},"threads":os.cpu_count(),"plot":{},"control_condition":"control"}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax,logsumexp,expit
sys.path.insert(0,str(Path(__file__).resolve().parent))
from period_support import project_periods_with_mean
from source_assignment_distribution import period_distribution_features,period_distribution_kernel
from source_assignment_core import role_geometry, role_assignment_evidence, role_score_features, fit_monotone_mean, interpolate_mean, measured_peer_partial_power
from source_amplitude_normalization import balance_source_amplitudes,correlated_frequency_average_representation,AMPLITUDE_SOLVER
from source_correlated_geometry import correlated_clock_order,harmonic_fundamental_support
torch.set_num_threads(CONFIG["threads"])
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model",type=Path,default=CONFIG["paths"]["model"],help="Fitted model archive")
parser.add_argument("--input",required=True,type=Path)
parser.add_argument("--balance-cache",type=Path,
    help="Legacy 0.2.4 amplitude cache; observations and solver must match")
parser.add_argument("--save-balance-cache",type=Path,
    help="Legacy 0.2.4: save amplitude fits")
parser.add_argument("--output",required=True,type=Path)
parser.add_argument("--control-condition",default=CONFIG["control_condition"],help="Condition label for the control mean in the contrast output; it does not enter period prediction")
parser.add_argument("--case",type=int,default=-1)
parser.add_argument("--period-min",type=float)
parser.add_argument("--period-max",type=float)
parser.add_argument("--time-unit",default=None)
parser.add_argument("--variant",type=int,default=None,help="Variant ID; unified models default to their recommended configuration")
parser.add_argument("--device",choices=["cpu","cuda"],default="cpu",help="Device for the unified No-only source representation and role score")
parser.add_argument("--training-length",type=int,default=0,
    help="Legacy 0.2.4 fitted acquisition head; unified models use observed timestamps and masks")
parser.add_argument("--source-features",choices=["assignment","density","core"],default="assignment",
    help="Legacy 0.2.4 source feature definition")
parser.add_argument("--mean-readout",choices=["full","calibrated","source"],default="full",
    help="Legacy 0.2.4 mean-stage readout; unified controls use --variant")
parser.add_argument("--period-composition",choices=["bounded","unbounded"],default="bounded",
    help="Legacy 0.2.4 support projection; unified controls use --variant")
args=parser.parse_args()
output_path=args.output
saved=np.load(args.model,allow_pickle=False)
if str(saved.get("measurement_kind",""))=="shared_landmark_renewal":
    from source_assignment_distribution import shared_landmark_observations,load_shared_landmark_renewal_model,predict_shared_landmark_renewal,_shared_landmark_unit_seconds
    model=load_shared_landmark_renewal_model(args.model)
    unit=str(model["time_unit"]) if args.time_unit is None else args.time_unit
    factor=_shared_landmark_unit_seconds(unit)/_shared_landmark_unit_seconds(model["time_unit"])
    if args.period_min is not None:
        assert np.isclose(args.period_min*factor,model["period"].min(),rtol=1e-10,atol=1e-12), "Period bounds must match the fitted physical grid."
    if args.period_max is not None:
        assert np.isclose(args.period_max*factor,model["period"].max(),rtol=1e-10,atol=1e-12), "Period bounds must match the fitted physical grid."
    query=pd.read_parquet(args.input,columns=["trace_id","run","condition","x","signal_y"])
    data=shared_landmark_observations(query,source_grouping=str(model["source_grouping"]),time_unit=unit)
    case=0 if args.case<0 else args.case
    value=predict_shared_landmark_renewal(data,model,case_indices=[case])
    owner=value["source_index"];cohort=value["run_index"]
    predictions=data["metadata"].reset_index().assign(case_index=case,estimate=value["estimate"][0],
        source_period=value["source_period"][0,owner],group_mean=value["group_mean"][0,cohort],deviation=value["deviation"][0],
        time_unit=unit,source_id=value["source_id"][owner],observed_count=data["mask"].sum(1),
        measurement_kind="shared_landmark_renewal",readout_kind=str(model["readout_kind"]),
        method_status=np.where(value["source_likelihood_available"][0,owner],"ok","prior_no_events"))
    predictions.to_csv(output_path,index=False,encoding="utf-8",lineterminator="\n")
    np.savez_compressed(output_path.with_name(output_path.stem+"_source.npz"),**value,
        trace_id=predictions.trace_id.to_numpy(dtype=str),measurement_kind=np.asarray("shared_landmark_renewal"))
    position=np.flatnonzero(model["case_index"]==case).item()
    print("SHARED_LANDMARK_PREDICTED",len(predictions),"measurements",len(value["source_id"]),"sources",case,"case",
        "training_reference_sources",int(model["train_reference_source_count"][position]),"training_units",int(model["train_reference_unit_count"][position]),output_path,flush=True)
    raise SystemExit(0)
if str(saved.get("schema","")) in {"source_assignment_unified_0.3.0","source_assignment_unified_0.3.1"}:
    from source_assignment_unified import predict_file
    predictions=predict_file(args,saved)
    raise SystemExit(0)
profile_kernel=str(saved.get("profile_kernel","context"))
active=np.flatnonzero((np.arange(len(saved["fit_length"]))==args.case)|(args.case<0))
shared=["method","time","lag","period","profile_bags"]
case_fields=["fit_length","coefficient","anchor_coefficient","within_coefficient","background_coefficient",
    "role_covariance","role_coefficient","calibration_experiments","calibration_prediction",
    "calibration_mean","profile_feature_scale","profile_normalization","profile_bandwidth",
    "profile_dual","profile_train_run"]
case_fields+={"context":[],"context-cdf":["profile_training_cdf","profile_cdf_bandwidth"]}[profile_kernel]
model={k:saved[k] for k in shared}
model.update({k:saved[k][active] for k in case_fields})
model["core_partial_peer"]=saved.get("core_partial_peer",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["core_coherent_peer"]=saved.get("core_coherent_peer",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["mean_identity_tails"]=saved.get("mean_identity_tails",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["source_shared_order"]=saved.get("source_shared_order",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["source_posterior_energy"]=saved.get("source_posterior_energy",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["source_frequency_marginal"]=saved.get("source_frequency_marginal",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["core_half_gram"]=saved.get("core_half_gram",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["source_colored_order"]=saved.get("source_colored_order",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["source_common_baseline"]=saved.get("source_common_baseline",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["source_frequency_correlated"]=saved.get("source_frequency_correlated",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["source_harmonic_support"]=saved.get("source_harmonic_support",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["source_cohort_center"]=saved.get("source_cohort_center",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["source_peer_original"]=saved.get("source_peer_original",np.zeros(len(saved["fit_length"]),dtype=bool))[active]
model["core_contrast_coefficient"]=saved.get("core_contrast_coefficient",np.zeros((len(saved["fit_length"]),15)))[active]
query = pd.read_parquet(args.input,columns=["trace_id","run","condition","x","signal_y"])
metadata = query[["trace_id","run","condition"]].drop_duplicates().set_index("trace_id")
query["time_relative_h"] = query.x-query.groupby("run").x.transform("min")
table = query.pivot(index="trace_id",columns="time_relative_h",values="signal_y")
assert metadata.index.is_unique and np.isfinite(table.columns.to_numpy(dtype=float)).all()
trace_id = table.index.to_numpy(dtype=str)
metadata = metadata.loc[trace_id]
time = table.columns.to_numpy(dtype=float)
cadence = np.median(np.diff(time))
assert cadence>0 and np.allclose(np.diff(time),cadence,rtol=1e-10,atol=1e-10)
period = model["period"]
signal = table.to_numpy(dtype=float)
mask = np.isfinite(signal)
count = mask.sum(axis=1)
requested_head = np.where(args.training_length>0,args.training_length,count)
model_index = pd.Index(model["fit_length"]).get_indexer(requested_head)
assert args.training_length>0 or np.isclose(cadence,1.,rtol=1e-10,atol=1e-10)
assert np.all(model_index>=0) and np.array_equal(mask,np.arange(len(time))[None]<count[:,None])
centered_time = (time[None]-np.sum(mask*time[None],axis=1)[:,None]/count[:,None])*mask
base = np.stack([mask/np.sqrt(count[:,None]),
    centered_time/np.sqrt(np.sum(centered_time**2,axis=1))[:,None]],axis=-1)
signal = np.where(mask,signal,0.)
residual = signal-np.einsum("nti,ni->nt",base,np.einsum("nti,nt->ni",base,signal))
anchor_input_mixed=residual/np.sqrt(np.sum(residual**2,axis=1)/count)[:,None]
groups,first,run_index = np.unique(metadata.run.to_numpy(dtype=str),return_index=True,return_inverse=True)
membership = (groups[:,None]==metadata.run.to_numpy(dtype=str)[None]).astype(float)
run_n = membership.sum(axis=1)
assert np.all(run_n>1) and np.array_equal(count,count[first][run_index])
baseline_selected=model["source_common_baseline"][model_index]
if np.any(baseline_selected):
    baseline_lengths,baseline_layout=np.unique(count,return_inverse=True)
    baseline_difference=np.diff(np.eye(len(time)),n=2,axis=0)
    baseline_rows=np.arange(len(time)-2)[None]<baseline_lengths[:,None]-2
    baseline_lambda=float(saved["common_baseline_scale"])*(float(saved["common_baseline_period_h"])/cadence)**4
    baseline_precision=np.eye(len(time))[None]+(baseline_lambda*baseline_difference.T)[None]@(
        baseline_difference[None]*baseline_rows[:,:,None])
    baseline_highpass=np.eye(len(time))[None]-np.linalg.solve(
        baseline_precision,np.broadcast_to(np.eye(len(time)),baseline_precision.shape))
    baseline_mean=(membership/run_n[:,None])@residual
    baseline_by_layout=baseline_mean[None]@(np.eye(len(time))[None]-baseline_highpass)
    common_baseline=baseline_by_layout[baseline_layout[first],np.arange(len(groups))]
    residual=np.where(baseline_selected[:,None],residual-common_baseline[run_index],residual)
mixed = residual/np.sqrt(np.sum(residual**2,axis=1)/count)[:,None]
input_mixed=mixed.copy()
if np.any(model["source_cohort_center"][model_index]):
    contrast=residual-(membership@residual/run_n[:,None])[run_index]
    contrast/=np.sqrt(np.sum(contrast**2,axis=1)/count)[:,None]
    mixed=np.where(model["source_cohort_center"][model_index,None],contrast,mixed)
source_input=mixed.copy()
if bool(saved.get("source_amplitude_balance",False)):
    balance_arguments = {}
    if np.any(model["source_frequency_marginal"][model_index]):
        balance_arguments["frequency_marginal"] = model["source_frequency_marginal"][model_index]&~model["source_frequency_correlated"][model_index]
    if np.any(model["source_colored_order"][model_index]|model["source_frequency_correlated"][model_index]):
        balance_arguments["return_residuals"]=True
    if args.balance_cache is None:
        balance_arguments["return_residuals"] = True
        amplitude=balance_source_amplitudes(time,count,mixed,period,**balance_arguments)
    else:
        loaded=dict(np.load(args.balance_cache,allow_pickle=False))
        assert str(loaded["amplitude_solver"])==AMPLITUDE_SOLVER
        index=pd.Index(loaded["input_trace_id"]).get_indexer(trace_id)
        assert (index>=0).all()
        record_keys=["signal","periods","component_rms","model_order","bic_one","bic_two",
            "two_clock_probability","two_clock_signal","posterior_energy_signal","posterior_component_rms",
            "two_optimizer_nfev","noise_variance","periodic_variance","one_clock_residual",
            "two_clock_residual","one_clock_period","two_clock_periods","frequency_mean_signal",
            "frequency_order_probability"]
        amplitude={k:(v[index].copy() if k in record_keys else v.copy())
            for k,v in loaded.items() if not k.startswith("input_")}
        time_keys=["signal","two_clock_signal","posterior_energy_signal","frequency_mean_signal",
            "one_clock_residual","two_clock_residual"]
        amplitude={k:(v[:,:len(time)] if k in time_keys else v) for k,v in amplitude.items()}
        np.testing.assert_array_equal(count,loaded["input_count"][index])
        np.testing.assert_array_equal(time,loaded["input_time"][:len(time)])
        np.testing.assert_array_equal(period,loaded["input_period"])
        np.testing.assert_allclose(mixed,loaded["input_mixed"][index,:len(time)],rtol=0.,atol=1e-12)
        marginal=balance_arguments.get("frequency_marginal",np.zeros(len(count),dtype=bool))
        assert np.all(~marginal|loaded["input_frequency_marginal"][index])
    if args.save_balance_cache is not None:
        assert args.balance_cache is None
        args.save_balance_cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.save_balance_cache,**amplitude,input_trace_id=trace_id,
            input_time=time,input_count=count,input_mixed=mixed,input_period=period,
            input_frequency_marginal=balance_arguments.get("frequency_marginal",np.zeros(len(count),dtype=bool)))
    if np.any(model["source_colored_order"][model_index]):
        amplitude,source_noise=correlated_clock_order(amplitude,membership,run_index,count,
            model["source_colored_order"][model_index])
    correlated_selected=np.flatnonzero(model["source_frequency_correlated"][model_index])
    if len(correlated_selected):
        correlated_frequency=correlated_frequency_average_representation(time,count[correlated_selected],
            source_input[correlated_selected],period,source_noise["rho"][correlated_selected],
            amplitude["noise_variance"][correlated_selected],amplitude["periodic_variance"][correlated_selected],
            device="cuda" if torch.cuda.is_available() else "cpu")
        frequency_signal=amplitude.get("frequency_mean_signal",amplitude["posterior_energy_signal"].copy())
        frequency_signal[correlated_selected]=correlated_frequency["signal"]
        amplitude["frequency_mean_signal"]=frequency_signal
    if np.any(model["source_frequency_marginal"][model_index]):
        amplitude["posterior_energy_signal"] = amplitude["frequency_mean_signal"]
        amplitude["two_clock_probability"] = amplitude["frequency_order_probability"]
    mixed=amplitude["signal"]
    if np.any(model["source_shared_order"]|model["source_posterior_energy"]):
        two_signal=np.where(model["source_posterior_energy"][model_index,None],
            amplitude["posterior_energy_signal"],amplitude["two_clock_signal"])
        shared_probability=expit(.5*(membership@(amplitude["bic_one"]-amplitude["bic_two"])))
        source_two_probability=np.where(model["source_shared_order"][model_index],
            shared_probability[run_index],amplitude["two_clock_probability"])
        mixed=source_input+source_two_probability[:,None]*(two_signal-source_input)
    mixed/=np.sqrt(np.sum(mixed**2,axis=1)/count)[:,None]
peer_raw = ((membership@mixed)[run_index]-mixed)/(run_n[run_index,None]-1.)
if np.any(model["source_peer_original"][model_index]):
    reference_scale=np.sqrt((membership@(np.sum(residual**2,axis=1)/count))/run_n)
    reference_mixed=residual/reference_scale[run_index,None]
    reference_peer_raw=((membership@reference_mixed)[run_index]-reference_mixed)/(run_n[run_index,None]-1.)
    peer_raw=np.where(model["source_peer_original"][model_index,None],reference_peer_raw,peer_raw)
peer_rms2 = np.sum(peer_raw**2,axis=1)/count
peer = peer_raw/np.sqrt(peer_rms2)[:,None]
lengths = np.unique(count)
layout_index = pd.Index(lengths).get_indexer(count)
layout_mask = np.arange(len(time))[None]<lengths[:,None]
layout_centered_time = (time[None]-np.sum(layout_mask*time[None],axis=1)[:,None]/lengths[:,None])*layout_mask
layout_base = np.stack([layout_mask/np.sqrt(lengths[:,None]),
    layout_centered_time/np.sqrt(np.sum(layout_centered_time**2,axis=1))[:,None]],axis=-1)
angle = 2*np.pi*time[None]/period[:,None]
harmonic = np.stack([np.cos(angle),np.sin(angle),np.cos(2*angle),np.sin(2*angle)],axis=-1)
base_cross = np.einsum("lti,ptj->lpij",layout_base,harmonic,optimize=True)
gram = np.einsum("lt,pti,ptj->lpij",layout_mask.astype(float),harmonic,harmonic,optimize=True)
gram -= base_cross.swapaxes(-1,-2)@base_cross
rhs_own = np.einsum("pti,nt->npi",harmonic,mixed,optimize=True)
rhs_peer = np.einsum("pti,nt->npi",harmonic,peer,optimize=True)
rhs = np.stack([rhs_own,rhs_peer],axis=-1)
coefficient_h2 = np.linalg.solve(gram[layout_index],rhs)
coefficient_h1 = np.linalg.solve(gram[layout_index,:,:2,:2],rhs[:,:,:2])
power_h2 = np.sum(rhs*coefficient_h2,axis=2)/count[:,None,None]
power_h1 = np.sum(rhs[:,:,:2]*coefficient_h1,axis=2)/count[:,None,None]
cross_power = np.sum(rhs_own[:,:,:2]*coefficient_h1[:,:,:,1],axis=2)/count[:,None]
phase_alignment = np.clip(cross_power/np.sqrt(power_h1[:,:,0]*power_h1[:,:,1]),-1.,1.)
other_power = ((membership@power_h1[:,:,0])[run_index]-power_h1[:,:,0])/(run_n[run_index,None]-1.)
if np.any(model["source_peer_original"][model_index]):
    reference_rhs=np.einsum("pti,nt->npi",harmonic[:,:,:2],reference_mixed,optimize=True)
    reference_h1=np.linalg.solve(gram[layout_index,:,:2,:2],reference_rhs[:,:,:,None])[:,:,:,0]
    reference_power=np.sum(reference_rhs*reference_h1,axis=-1)/count[:,None]
    reference_other=((membership@reference_power)[run_index]-reference_power)/(run_n[run_index,None]-1.)
    other_power=np.where(model["source_peer_original"][model_index,None],reference_other,other_power)
coherence = np.clip(power_h1[:,:,1]*peer_rms2[:,None]/other_power,0.,1.)
power_ratio = power_h1[:,:,0]/other_power
orthogonal = power_h1[:,:,0]*(1.-phase_alignment**2)
lag = np.arange(int(np.ceil(period.max()/cadence))+1)
assert np.all(count>lag[-1])
fft_length = 2*len(time)
own_fft = np.fft.rfft(mixed,n=fft_length,axis=1)
peer_fft = np.fft.rfft(peer,n=fft_length,axis=1)
own_acf = np.fft.irfft(own_fft*own_fft.conj(),n=fft_length,axis=1)[:,lag]/(count[:,None]-lag[None])
peer_acf = np.fft.irfft(peer_fft*peer_fft.conj(),n=fft_length,axis=1)[:,lag]/(count[:,None]-lag[None])
acf = np.stack([own_acf,peer_acf],axis=-1)
period_lag = period/cadence
lower = np.floor(period_lag).astype(int)
upper = np.ceil(period_lag).astype(int)
fraction = period_lag-lower
at_period = acf[:,lower]*(1.-fraction)[None,:,None]+acf[:,upper]*fraction[None,:,None]
half = period_lag/2.
lower_half = np.floor(half).astype(int)
upper_half = np.ceil(half).astype(int)
fraction_half = half-lower_half
at_half = acf[:,lower_half]*(1.-fraction_half)[None,:,None]+acf[:,upper_half]*fraction_half[None,:,None]
harmonic_increment=np.maximum(power_h2-power_h1,0.)
if np.any(model["source_harmonic_support"][model_index]):
    harmonic_support,harmonic_log_bf=harmonic_fundamental_support(time,count,mixed,peer,harmonic,source_noise["rho"])
    harmonic_support=np.where(model["source_harmonic_support"][model_index,None,None],harmonic_support,1.)
    harmonic_increment*=harmonic_support
own_base = np.stack([power_h1[:,:,0],harmonic_increment[:,:,0],
    at_period[:,:,0],at_half[:,:,0]],axis=-1)
peer_base = np.stack([power_h1[:,:,1],harmonic_increment[:,:,1],
    at_period[:,:,1],at_half[:,:,1]],axis=-1)
base_features = np.concatenate([own_base,peer_base,np.stack([coherence,power_ratio,phase_alignment,orthogonal],axis=-1)],axis=-1)
oriented_area = coefficient_h1[:,:,0,0]*coefficient_h1[:,:,1,1]-coefficient_h1[:,:,1,0]*coefficient_h1[:,:,0,1]
power_norm = count[:,None]*np.sqrt(power_h1[:,:,0]*power_h1[:,:,1])
quadrature = np.divide(oriented_area*np.sqrt(np.linalg.det(gram[:,:,:2,:2]))[layout_index],power_norm,
    out=np.zeros_like(oriented_area),where=power_norm>np.finfo(float).eps)
quadrature = np.clip(quadrature,-1.,1.)
base_features = np.concatenate([base_features,quadrature[:,:,None]],axis=-1)
left,right = np.triu_indices(base_features.shape[-1])
features = np.concatenate([base_features,base_features[:,:,left]*base_features[:,:,right]],axis=-1)

raw_features=features.copy()
design,layout,joint_gram,rhs=role_geometry(time,count,mixed,peer,period)
background={"posterior":softmax(np.einsum("npd,gd->gnp",raw_features,model["background_coefficient"],optimize=True),axis=-1)}
shape_phase,diagnostic=role_assignment_evidence(joint_gram,rhs,layout,model["role_covariance"],
    background["posterior"],len(period)//2,len(period)//2)

core_logit_all=np.einsum("npd,gd->gnp",raw_features,model["coefficient"],optimize=True)
if np.any(model["core_partial_peer"]):
    partial_base=base_features.copy()
    partial_base[:,:,11]=measured_peer_partial_power(time,count,mixed,peer,design,layout)
    partial_features=np.concatenate([partial_base,partial_base[:,:,left]*partial_base[:,:,right]],axis=-1)
    core_logit_all[model["core_partial_peer"]]=np.einsum("npd,gd->gnp",partial_features,
        model["coefficient"][model["core_partial_peer"]],optimize=True)
if np.any(model["core_coherent_peer"]):
    coherent_base=base_features.copy()
    coherent_base[:,:,11]=measured_peer_partial_power(time,count,mixed,peer,design,layout,coherent=True,
        half_gram=bool(np.any(model["core_half_gram"])))
    coherent_features=np.concatenate([coherent_base,coherent_base[:,:,left]*coherent_base[:,:,right]],axis=-1)
    core_logit_all[model["core_coherent_peer"]]=np.einsum("npd,gd->gnp",coherent_features,
        model["coefficient"][model["core_coherent_peer"]],optimize=True)
if np.any(model["core_contrast_coefficient"]):
    assert not np.any(np.any(model["core_contrast_coefficient"]!=0.,axis=-1)&~model["core_coherent_peer"])
    raw_contrast=residual-(membership@residual/run_n[:,None])[run_index]
    contrast_rms2=np.sum(raw_contrast**2,axis=-1)/count
    contrast_signal=np.divide(raw_contrast,np.sqrt(contrast_rms2)[:,None],
        out=np.zeros_like(raw_contrast),where=contrast_rms2[:,None]>0.)
    contrast_rhs=np.einsum("pti,nt->npi",harmonic[:,:,:2],contrast_signal,optimize=True)
    contrast_coefficient=np.linalg.solve(gram[layout_index,:,:2,:2],contrast_rhs[:,:,:,None])[:,:,:,0]
    contrast_power=np.sum(contrast_rhs*contrast_coefficient,axis=-1)/count[:,None]
    contrast_features=np.concatenate([contrast_power[:,:,None],contrast_power[:,:,None]*coherent_base,
        contrast_power[:,:,None]**2],axis=-1)
    core_logit_all+=np.einsum("npd,gd->gnp",contrast_features,model["core_contrast_coefficient"],optimize=True)
core_log_mass=core_logit_all-logsumexp(core_logit_all,axis=-1,keepdims=True)
density_shape=np.stack([core_log_mass,np.log(np.maximum(background["posterior"],np.finfo(float).tiny))],axis=-1)
shape_phase={"assignment":shape_phase,"density":density_shape,"core":np.zeros_like(shape_phase)}[args.source_features]
role_features=role_score_features(shape_phase,base_features)
role_score=np.einsum("npd,nd->np",role_features[model_index,np.arange(len(count))],
    model["role_coefficient"][model_index],optimize=True)
core_score=np.einsum("npd,nd->np",raw_features,model["coefficient"][model_index],optimize=True)
partial_record=model["core_partial_peer"][model_index]
if np.any(partial_record):
    core_score[partial_record]=np.einsum("npd,nd->np",partial_features[partial_record],
        model["coefficient"][model_index[partial_record]],optimize=True)
coherent_record=model["core_coherent_peer"][model_index]
if np.any(coherent_record):
    core_score[coherent_record]=np.einsum("npd,nd->np",coherent_features[coherent_record],
        model["coefficient"][model_index[coherent_record]],optimize=True)
if np.any(model["core_contrast_coefficient"]):
    core_score+=np.einsum("npd,nd->np",contrast_features,model["core_contrast_coefficient"][model_index],optimize=True)
source_score=core_score+role_score

source_probability = softmax(source_score,axis=-1)
source_expectation = source_probability@period
anchor_features=raw_features
if bool(saved.get("anchor_original_No",False)):
    anchor_peer_raw=((membership@anchor_input_mixed)[run_index]-anchor_input_mixed)/(run_n[run_index,None]-1.)
    anchor_peer_rms2=np.sum(anchor_peer_raw**2,axis=1)/count
    anchor_peer=anchor_peer_raw/np.sqrt(anchor_peer_rms2)[:,None]
    anchor_rhs=np.stack([np.einsum("pti,nt->npi",harmonic,anchor_input_mixed,optimize=True),
        np.einsum("pti,nt->npi",harmonic,anchor_peer,optimize=True)],axis=-1)
    anchor_h2=np.linalg.solve(gram[layout_index],anchor_rhs)
    anchor_h1=np.linalg.solve(gram[layout_index,:,:2,:2],anchor_rhs[:,:,:2])
    anchor_power_h2=np.sum(anchor_rhs*anchor_h2,axis=2)/count[:,None,None]
    anchor_power_h1=np.sum(anchor_rhs[:,:,:2]*anchor_h1,axis=2)/count[:,None,None]
    anchor_cross=np.sum(anchor_rhs[:,:,:2,0]*anchor_h1[:,:,:,1],axis=-1)/count[:,None]
    anchor_alignment=np.clip(anchor_cross/np.sqrt(anchor_power_h1[:,:,0]*anchor_power_h1[:,:,1]),-1.,1.)
    anchor_other=((membership@anchor_power_h1[:,:,0])[run_index]-anchor_power_h1[:,:,0])/(run_n[run_index,None]-1.)
    anchor_coherence=np.clip(anchor_power_h1[:,:,1]*anchor_peer_rms2[:,None]/anchor_other,0.,1.)
    anchor_fft=np.fft.rfft(np.stack([anchor_input_mixed,anchor_peer],axis=-1),n=fft_length,axis=1)
    anchor_acf=np.fft.irfft(anchor_fft*anchor_fft.conj(),n=fft_length,axis=1)[:,lag]/(count[:,None,None]-lag[None,:,None])
    anchor_at_period=anchor_acf[:,lower]*(1.-fraction)[None,:,None]+anchor_acf[:,upper]*fraction[None,:,None]
    anchor_at_half=anchor_acf[:,lower_half]*(1.-fraction_half)[None,:,None]+anchor_acf[:,upper_half]*fraction_half[None,:,None]
    anchor_area=anchor_h1[:,:,0,0]*anchor_h1[:,:,1,1]-anchor_h1[:,:,1,0]*anchor_h1[:,:,0,1]
    anchor_norm=count[:,None]*np.sqrt(anchor_power_h1[:,:,0]*anchor_power_h1[:,:,1])
    anchor_quadrature=np.divide(anchor_area*np.sqrt(np.linalg.det(gram[:,:,:2,:2]))[layout_index],anchor_norm,
        out=np.zeros_like(anchor_area),where=anchor_norm>np.finfo(float).eps)
    anchor_base=np.stack([anchor_power_h1[:,:,0],np.maximum(anchor_power_h2[:,:,0]-anchor_power_h1[:,:,0],0.),
        anchor_at_period[:,:,0],anchor_at_half[:,:,0],anchor_power_h1[:,:,1],
        np.maximum(anchor_power_h2[:,:,1]-anchor_power_h1[:,:,1],0.),anchor_at_period[:,:,1],anchor_at_half[:,:,1],
        anchor_coherence,anchor_power_h1[:,:,0]/anchor_other,anchor_alignment,
        anchor_power_h1[:,:,0]*(1.-anchor_alignment**2),np.clip(anchor_quadrature,-1.,1.)],axis=-1)
    anchor_features=np.concatenate([anchor_base,anchor_base[:,:,left]*anchor_base[:,:,right]],axis=-1)
anchor_score = np.einsum("npd,nd->np",anchor_features,model["anchor_coefficient"][model_index],optimize=True)
anchor_period = softmax(anchor_score,axis=-1)@period
run_source_mean = (membership@source_expectation)/run_n
run_anchor_mean = (membership@anchor_period)/run_n
within_deviation = model["within_coefficient"][model_index]*(anchor_period-run_anchor_mean[run_index])
calibration_index = model_index[first]
calibration_n = model["calibration_experiments"][calibration_index]
calibration_x = model["calibration_prediction"][calibration_index]
calibration_y = model["calibration_mean"][calibration_index]
calibrated_mean = interpolate_mean(run_source_mean[:,None],calibration_x,calibration_y,calibration_n,
    model["mean_identity_tails"][calibration_index])[:,0]

query_bags=np.einsum("rn,npd->rpd",membership/run_n[:,None],raw_features,optimize=True)
query_z=(query_bags/model["profile_feature_scale"][calibration_index,None]).reshape(len(groups),-1)
train_z=(model["profile_bags"][None]/model["profile_feature_scale"][calibration_index,None,None,:]).reshape(len(groups),len(model["profile_bags"]),-1)
distance=np.maximum(np.sum(query_z**2,axis=-1)[:,None]+np.sum(train_z**2,axis=-1)
    -2*np.einsum("rd,rtd->rt",query_z,train_z,optimize=True),0.)/query_z.shape[-1]/model["profile_normalization"][calibration_index,None]
kernel=np.exp(-distance/(2*model["profile_bandwidth"][calibration_index,None]))
if profile_kernel=="context-cdf":
    background_probability=background["posterior"][model_index,np.arange(len(count))]
    query_cdf=period_distribution_features(source_probability,background_probability,membership/run_n[:,None])
    kernel*=period_distribution_kernel(query_cdf[:,None],model["profile_training_cdf"][calibration_index],
        model["profile_cdf_bandwidth"][calibration_index])[:,0]
profile_shift=np.sum(kernel*model["profile_dual"][calibration_index]*model["profile_train_run"][calibration_index],axis=-1)
unbounded_means=np.stack([calibrated_mean+profile_shift,calibrated_mean,run_source_mean])
unbounded_periods=unbounded_means[:,run_index]+within_deviation[None]
bounded_periods=project_periods_with_mean(unbounded_periods,unbounded_means,membership,period.min(),period.max())
readout_index={"full":0,"calibrated":1,"source":2}[args.mean_readout]
estimate={"bounded":bounded_periods,"unbounded":unbounded_periods}[args.period_composition][readout_index]
experiment_mean={"bounded":np.clip(unbounded_means,period.min(),period.max()),
    "unbounded":unbounded_means}[args.period_composition][readout_index]
unbounded_deviation=within_deviation.copy()
within_deviation=estimate-experiment_mean[run_index]
calibration_shift=np.where(args.mean_readout=="source",0.,calibrated_mean-run_source_mean)
profile_shift=np.where(args.mean_readout=="full",profile_shift,0.)
support_shift=experiment_mean-unbounded_means[readout_index]
assert np.isfinite(estimate).all()

predictions=metadata.reset_index().assign(estimate_h=estimate,experiment_mean_h=experiment_mean[run_index],
    individual_deviation_h=within_deviation,source_mean_h=run_source_mean[run_index],
    calibration_shift_h=calibration_shift[run_index],profile_shift_h=profile_shift[run_index],
    support_shift_h=support_shift[run_index],unbounded_individual_deviation_h=unbounded_deviation,
    estimate_full_h={"bounded":bounded_periods,"unbounded":unbounded_periods}[args.period_composition][0],
    estimate_calibrated_h={"bounded":bounded_periods,"unbounded":unbounded_periods}[args.period_composition][1],
    estimate_source_h={"bounded":bounded_periods,"unbounded":unbounded_periods}[args.period_composition][2],
    period_composition=args.period_composition,
    method=str(model["method"])+{"full":"","calibrated":"__no_group","source":"__raw_mean"}[args.mean_readout],method_status="ok")
predictions.to_csv(output_path,index=False,encoding="utf-8",lineterminator="\n")
experiments=predictions.groupby(["run","condition"]).agg(n_input=("trace_id","size"),
    estimate_mean_h=("estimate_h","mean")).reset_index()
conditions=experiments.groupby("condition").agg(n_experiments=("run","size"),n_input=("n_input","sum"),
    estimate_mean_h=("estimate_mean_h","mean")).reset_index()
control=conditions.loc[conditions.condition.eq(args.control_condition),"estimate_mean_h"].mean()
contrasts=conditions.loc[conditions.condition.ne(args.control_condition)].copy()
contrasts["estimate_effect_h"]=contrasts.estimate_mean_h-control
experiments.to_csv(output_path.with_name(output_path.stem+"_experiments.csv"),index=False,encoding="utf-8",lineterminator="\n")
contrasts.to_csv(output_path.with_name(output_path.stem+"_contrasts.csv"),index=False,encoding="utf-8",lineterminator="\n")
print("SOURCE_ASSIGNMENT_PREDICTED",len(predictions),"records",len(experiments),"experiments",output_path,flush=True)
