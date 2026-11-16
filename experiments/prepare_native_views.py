"""Build source-reference training views with the original independent-unit exclusions."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1],"output":"results/0.3.0"},
    "datasets":{"native":{"input":"data/sota_dev/recomputed_clean/lemna_task_period_acf_input.parquet",
        "context":"data/sota_dev/recomputed_clean/lemna_task_period_supervised_context.npz",
        "reference":"data/external/lemna_watanabe2023/dual_reporter_observations.parquet","unit":"h","bounds":(12.,48.)},
        "adfecg":{"input":"data/external/adfecgdb/inputs.parquet",
        "context":"data/external/adfecgdb/supervised_context.npz",
        "reference":"data/external/adfecgdb/foreground_reference.parquet","unit":"s","bounds":(.25,1.5)}},
    "threads":os.cpu_count(),"plot":{},"background_gains":(.5,2.),
    "candidates":73,"row_group_size":256000}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
import sys,json,argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
plt.rcParams.update(CONFIG["plot"])
root=CONFIG["paths"]["root"]
sys.path.insert(0,str(root/"software"))
from source_assignment_unified import observations,detrend,harmonic_profile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset",choices=list(CONFIG["datasets"]),default="native")
parser.add_argument("--period-min",type=float)
parser.add_argument("--period-max",type=float)
parser.add_argument("--time-unit",choices=["h","min","s","ms"])
parser.add_argument("--candidates",type=int,default=CONFIG["candidates"])
parser.add_argument("--coherent-driver",action="store_true")
parser.add_argument("--positive-shifts",action="store_true",help="Append unit-clock time dilations calibrated by distinct qualified FG sources within each independent unit")
args=parser.parse_args()
settings=CONFIG["datasets"][args.dataset]
unit=args.time_unit or settings["unit"]
seconds={"h":3600.,"min":60.,"s":1.,"ms":.001}
time_factor=seconds[settings["unit"]]/seconds[unit]
period_min=settings["bounds"][0]*time_factor if args.period_min is None else args.period_min
period_max=settings["bounds"][1]*time_factor if args.period_max is None else args.period_max
assert 0<period_min<period_max and args.candidates>=2
period=np.linspace(period_min,period_max,args.candidates)
period_scale=np.sqrt(period_min*period_max)
folder=root/CONFIG["paths"]["output"]
if args.coherent_driver or args.positive_shifts:
    folder=root/"results/0.3.1"
output_tag=args.dataset+("_positive" if args.positive_shifts else "")+("_coherent_driver" if args.coherent_driver else "_training")
folder.mkdir(parents=True,exist_ok=True)
raw=pd.read_parquet(root/settings["input"],columns=["trace_id","run","condition","x","signal_y"],dtype_backend="pyarrow")
old=np.load(root/settings["context"],allow_pickle=False)
raw["x"]=raw.x*time_factor
data=observations(raw)
time=data["time"];mask=data["mask"];count=data["count"]
membership=data["membership"];run=data["run_index"]
origin=data["metadata"].reset_index()
origin_index=pd.Index(old["trace_id"]).get_indexer(origin.trace_id)
assert np.all(origin_index>=0)
target=old["target"][origin_index]*time_factor
qualified=old["qualified"][origin_index]
assert np.all((target[qualified]>=period_min)&(target[qualified]<=period_max))
independent=old["subject"][origin_index] if args.dataset=="adfecg" else origin.run.to_numpy(dtype=str)
no=detrend(data["signal"],time,mask)
no_rms=np.sqrt(np.sum(no**2,axis=1)/count)
tiny=np.sqrt(np.finfo(float).eps)

if args.dataset=="native":
    optical=pd.read_parquet(root/settings["reference"],columns=["trace_id","time_h","green","red"])
    optical["x"]=optical.time_h*time_factor
    optical["foreground_y"]=2.11*optical.green-.374*optical.red
    optical["background_y"]=-.741*optical.green+4.04*optical.red
    fg_wide=optical.pivot(index="trace_id",columns="x",values="foreground_y").reindex(index=origin.trace_id,columns=time)
    bg_wide=optical.pivot(index="trace_id",columns="x",values="background_y").reindex(index=origin.trace_id,columns=time)
    fg_present=np.isfinite(fg_wide.to_numpy())&mask
    nuisance_present=np.isfinite(bg_wide.to_numpy())&mask
    fg=detrend(fg_wide.fillna(0.).to_numpy(),time,fg_present)
    nuisance=detrend(bg_wide.fillna(0.).to_numpy(),time,nuisance_present)
    # Original run-level optical-to-No gain; the RGB separation is unchanged.
    no_scale=np.maximum(no_rms,np.finfo(float).tiny)
    sources=np.stack([fg/no_scale[:,None],nuisance/no_scale[:,None]],axis=-1)
    sources*=(fg_present&nuisance_present)[:,:,None]
    group_gram=np.einsum("gn,nij->gij",membership,np.einsum("nti,ntj->nij",sources,sources,optimize=True),optimize=True)
    group_rhs=np.einsum("gn,ni->gi",membership,np.einsum("nti,nt->ni",sources,no/no_scale[:,None],optimize=True),optimize=True)
    gain=(np.linalg.pinv(group_gram)@group_rhs[:,:,None])[:,:,0]
    foreground_component=fg*gain[run,0,None]
    background_component=nuisance*gain[run,1,None]
    nuisance_kind="No_calibrated_measured_optical_background"
else:
    foreground=pd.read_parquet(root/settings["reference"],columns=["run","x","foreground_y"])
    foreground["x"]=foreground.x*time_factor
    fg_wide=foreground.pivot(index="run",columns="x",values="foreground_y").reindex(index=data["runs"],columns=time)
    fg_raw=fg_wide.to_numpy()[run]
    fg_present=np.isfinite(fg_raw)&mask
    fg=detrend(np.nan_to_num(fg_raw),time,fg_present)
    # Estimated nuisance only: no measured ECG background is available.
    fg_energy=np.sum(fg**2,axis=1)
    projection=np.divide(np.sum(no*fg,axis=1),fg_energy,out=np.zeros(len(origin)),where=fg_energy>0.)
    foreground_component=projection[:,None]*fg
    nuisance=(no-foreground_component)*fg_present
    nuisance_present=fg_present
    # One actual FG waveform per window, observed through four lead gains.
    # The remaining shared temporal source is estimated from the four No
    # residuals. Its rank-one part and the remaining residual are kept apart.
    number=membership.sum(1).astype(int)
    assert np.all(number==4)
    residual_group=nuisance.reshape(len(data["runs"]),4,len(time))
    residual_covariance=residual_group@residual_group.transpose(0,2,1)
    _,residual_vector=np.linalg.eigh(residual_covariance)
    loading=residual_vector[:,:,-1]
    shared_wave=np.einsum("gc,gct->gt",loading,residual_group,optimize=True)
    background_component=(loading[:,:,None]*shared_wave[:,None]).reshape(no.shape)*mask
    gain=np.full((len(data["runs"]),2),np.nan)
    nuisance_kind="estimated_rank1_No_residual_after_measured_Direct_1_projection"

residual_component=(no-foreground_component-background_component)*mask
trend_component=data["signal"]-no
common_input=background_component
np.testing.assert_allclose(foreground_component+background_component+residual_component+trend_component,
    data["signal"],rtol=1e-12,atol=1e-10)

fg_rms=np.sqrt(np.sum(fg**2,axis=1)/count)
fg_unit=np.divide(fg,fg_rms[:,None],out=np.zeros_like(fg),where=fg_rms[:,None]>0.)
fg_complete=np.all(fg_present|~mask,axis=1)&(fg_rms>0.)
background_rms=np.sqrt(np.sum(background_component**2,axis=1)/count)
background_unit=np.divide(background_component,background_rms[:,None],out=np.zeros_like(background_component),where=background_rms[:,None]>0.)
available=membership@mask.astype(float)
common=((membership@background_component)/np.maximum(available,1.))[run]*mask
common_rms=np.sqrt(np.sum(common**2,axis=1)/count)
residual_rms=np.sqrt(np.sum(residual_component**2,axis=1)/count)

# A cell is a Native FG source; all four ECG leads observe one window source.
reference_source_id=origin.run.to_numpy(dtype=str) if args.dataset=="adfecg" else origin.trace_id.to_numpy(dtype=str)
source_index,source_names=pd.factorize(reference_source_id,sort=True)
source_member=(np.arange(len(source_names))[:,None]==source_index[None]).astype(float)
source_first=source_member.argmax(1)
source_mask=fg_present[source_first]
source_waveform=fg_unit[source_first]
source_count=count[source_first]
source_qualified=(source_member@qualified.astype(float))>0.
source_reference_count=source_member@qualified.astype(float)
source_reference=np.divide(source_member@np.where(qualified,target,0.),source_reference_count,
    out=np.full(len(source_names),np.nan),where=source_reference_count>0.)
source_geometry=np.load(root/"results/0.3.1"/(args.dataset+"_source_waveform_source_statistics.npz"),allow_pickle=False)
np.testing.assert_array_equal(source_geometry["trace_id"],origin.trace_id.to_numpy(dtype=str))
np.testing.assert_allclose(source_geometry["time"]*time_factor,time,rtol=0.,atol=1e-12)
np.testing.assert_array_equal(source_geometry["mask"],mask)
source_unit,unit_names=pd.factorize(independent[source_first],sort=True)
unit_member=(np.arange(len(unit_names))[:,None]==source_unit[None]).astype(float)
unit_reference_count=unit_member@source_qualified.astype(float)
unit_reference=np.divide(unit_member@np.where(source_qualified,source_reference,0.),unit_reference_count,
    out=np.full(len(unit_names),np.nan),where=unit_reference_count>0.)
source_deviation=np.where(source_qualified,source_reference-unit_reference[source_unit],0.)
unit_spread=np.sqrt(np.divide(unit_member@(source_deviation**2),unit_reference_count-1.,
    out=np.zeros(len(unit_names)),where=unit_reference_count>1.))
source_clock=np.where(np.isfinite(source_reference),source_reference,unit_reference[source_unit])
source_clock=np.where(np.isfinite(source_clock)&(source_clock>0.),source_clock,period_scale)
reference_n=membership@qualified.astype(float)
reference=np.divide(membership@np.where(qualified,target,0.),reference_n,
    out=np.full(len(data["runs"]),np.nan),where=reference_n>0.)

# Estimate a phase only to translate the measured waveform. A zero phase
# spread is a valid identity transformation, including ECG's one source/run.
source_center=(time[None]-(source_mask*time[None]).sum(1)[:,None]/source_count[:,None])*source_mask
source_trend=np.stack([source_mask/np.sqrt(source_count[:,None]),
    source_center/np.sqrt(np.sum(source_center**2,axis=1))[:,None]],axis=-1)
source_angle=2*np.pi*time[None]/source_clock[:,None]
source_harmonic=np.stack([np.cos(source_angle),np.sin(source_angle)],axis=-1)*source_mask[:,:,None]
source_harmonic-=source_trend@(source_trend.transpose(0,2,1)@source_harmonic)
source_beta=(np.linalg.pinv(source_harmonic.transpose(0,2,1)@source_harmonic)@
    (source_harmonic.transpose(0,2,1)@source_waveform[:,:,None]))[:,:,0]
source_phase=np.angle(source_beta[:,0]+1j*source_beta[:,1])
group_source=(membership@source_member.T)>0.
phase_weight=group_source*fg_complete[source_first][None]
phase_number=phase_weight.sum(1)
phase_resultant=(phase_weight@np.exp(1j*source_phase))/np.maximum(phase_number,1.)
common_phase=np.angle(phase_resultant)
source_group=group_source.argmax(0)
source_phase_error=np.angle(np.exp(1j*(source_phase-common_phase[source_group])))
phase_variance=np.divide(phase_weight@(source_phase_error**2),phase_number,
    out=np.zeros(len(phase_number)),where=phase_number>0.)
source_phase_sd=np.sqrt(phase_variance)[source_group]
qualified_group_source=group_source&source_qualified[None]&fg_complete[source_first][None]
cohort_median=pd.Series(np.where(source_qualified,source_reference,np.nan),index=data["runs"][source_group]).groupby(level=0).median().reindex(data["runs"]).to_numpy()
donor_allowed=group_source&fg_complete[source_first][None]&(source_qualified[None]|~qualified_group_source.any(1)[:,None])
donor_distance=abs(source_reference[None]-cohort_median[:,None])
donor_distance=np.where(np.isfinite(donor_distance),donor_distance,0.)
donor_index=np.where(donor_allowed,donor_distance,np.inf).argmin(1)
phase_source_index=donor_index[source_group]
foreground_basis_source_index=np.stack([np.arange(len(source_names)),phase_source_index,
    np.arange(len(source_names)),np.arange(len(source_names))])
unit_period_scale=1.+np.asarray([-.5,.5])[:,None]*np.divide(unit_spread,unit_reference,
    out=np.zeros_like(unit_spread),where=unit_reference>0.)[None]
source_period=np.concatenate([np.broadcast_to(source_clock,(2,len(source_clock))),source_clock[None]*unit_period_scale[:,source_unit]])
source_period[1]=source_clock[phase_source_index]
source_time_scale=np.concatenate([np.ones((2,len(source_clock))),1./unit_period_scale[:,source_unit]])
source_time_shift=np.zeros((4,len(source_names)))

# Exact observed-time interpolation; only outside the measured interval is a
# neighbouring measured cycle repeated. No sine carrier or random noise is used.
query=time[0]+source_time_scale[1:,:,None]*(time[None,None]-time[0])+source_time_shift[1:,:,None]
source_low=np.min(np.where(source_mask,time[None],np.inf),axis=1)
source_high=np.max(np.where(source_mask,time[None],-np.inf),axis=1)
input_source=foreground_basis_source_index[1:]
extension_clock=source_clock[input_source]
query+=np.ceil(np.maximum(source_low[input_source][:,:,None]-query,0.)/extension_clock[:,:,None])*extension_clock[:,:,None]
query-=np.ceil(np.maximum(query-source_high[input_source][:,:,None],0.)/extension_clock[:,:,None])*extension_clock[:,:,None]
grid_index=np.arange(len(time))[None]
previous=np.maximum.accumulate(np.where(source_mask,grid_index,-1),axis=1)
following=np.minimum.accumulate(np.where(source_mask,grid_index,len(time))[:,::-1],axis=1)[:,::-1]
interval=np.clip(np.searchsorted(time,query,side="right")-1,0,len(time)-1)
left=np.take_along_axis(previous[input_source],interval,axis=-1)
right=np.take_along_axis(following[input_source],np.minimum(interval+1,len(time)-1),axis=-1)
left=np.clip(left,0,len(time)-1);right=np.clip(right,0,len(time)-1)
span=time[right]-time[left]
fraction=np.divide(query-time[left],span,out=np.zeros_like(query),where=span>0.)
source_values=source_waveform[input_source]
warped=np.take_along_axis(source_values,left,axis=-1)*(1.-fraction)+np.take_along_axis(source_values,right,axis=-1)*fraction
warped=detrend(warped.reshape(-1,len(time)),time,np.broadcast_to(source_mask,(3,*source_mask.shape)).reshape(-1,len(time))).reshape(warped.shape)
warped_rms=np.sqrt(np.sum(warped**2,axis=-1)/source_count[None])
resample_gain=np.divide(1.,warped_rms,out=np.zeros_like(warped_rms),where=warped_rms>0.)
warped*=resample_gain[:,:,None]
foreground_source_basis=np.concatenate([source_waveform[None],warped])
foreground_loading=np.sum(foreground_component*fg_unit,axis=1)/count
foreground_basis=foreground_source_basis[:,source_index]*foreground_loading[None,:,None]
foreground_resample_gain=np.concatenate([np.ones((1,len(source_names))),resample_gain])[:,source_index]
phase_sd=source_phase_sd[source_index]
driver_period=source_reference[phase_source_index][source_index]
reference_valid=source_qualified[phase_source_index][source_index]
driver_ok=fg_complete[source_first][phase_source_index][source_index]
common_ok=np.all(mask,axis=1)|(count>0)
direct_ok=fg_complete
view_id=np.asarray(["native","common050","common200","direct050","direct200","phase050","phase200"])
view_foreground_index=np.asarray([0,0,0,0,0,1,1])
low,high=CONFIG["background_gains"]
background_gain=np.asarray([1.,1.,1.,low,high,low,high])
common_background_gain=np.asarray([0.,low,high,0.,0.,0.,0.])
residual_gain=np.ones(len(view_id))
constructible=np.stack([np.ones(len(origin),dtype=bool),common_ok,common_ok,direct_ok,direct_ok,driver_ok,driver_ok])
view_target=np.broadcast_to(target,(len(view_id),len(origin))).copy()
view_qualified=np.broadcast_to(qualified,view_target.shape).copy()
view_target[5:]=driver_period
view_qualified[5:]=reference_valid&driver_ok
positive_added=bool(args.positive_shifts)
positive_context={}
if args.positive_shifts:
    positive_period=target[None]*unit_period_scale[:,source_unit[source_index]]
    positive_ok=fg_complete&qualified&np.all(np.isfinite(positive_period)&(positive_period>=period_min)&(positive_period<=period_max),axis=0)
    view_id=np.concatenate([view_id,np.asarray(["shiftminus","shiftplus"])])
    view_foreground_index=np.r_[view_foreground_index,2,3]
    background_gain=np.r_[background_gain,1.,1.];common_background_gain=np.r_[common_background_gain,0.,0.]
    residual_gain=np.r_[residual_gain,1.,1.]
    constructible=np.concatenate([constructible,np.broadcast_to(fg_complete,(2,len(origin)))])
    view_target=np.concatenate([view_target,positive_period])
    view_qualified=np.concatenate([view_qualified,np.broadcast_to(positive_ok,(2,len(origin)))])
    group_unit=np.rint((membership@source_unit[source_index])/membership.sum(1)).astype(int)
    period_shift=unit_spread[group_unit]
    positive_period_by_run=reference[None]*unit_period_scale[:,group_unit]
    positive_run_valid=(membership@positive_ok.astype(float))>0.
    positive_context={"positive_shifts":np.asarray(True),"positive_views_added":np.asarray(True),
        "positive_shift_sd":period_shift,"positive_reference_count":unit_reference_count[group_unit],
        "positive_period_by_run":positive_period_by_run,"positive_run_valid":positive_run_valid,
        "positive_background_gain":np.asarray(1.),"positive_spread_unit_names":np.asarray(unit_names,dtype=str),
        "positive_spread_unit_sd":unit_spread,"positive_spread_unit_reference_count":unit_reference_count,
        "positive_period_scale_by_unit":unit_period_scale,"positive_spread_unit_mean_period":unit_reference}
style=np.arange(len(view_id))>=3
foreground_rms=np.sqrt(np.sum(foreground_component**2,axis=1)/count)
foreground_unit_gain=np.divide(1.,foreground_rms,out=np.ones_like(foreground_rms),where=foreground_rms>0.)
background_unit_gain=np.divide(1.,background_rms,out=np.ones_like(background_rms),where=background_rms>0.)
original_No_unit_gain=np.divide(1.,no_rms,out=np.ones_like(no_rms),where=no_rms>0.)
foreground_gain=np.where(style[:,None],foreground_unit_gain[None],1.)
background_gain=background_gain[:,None]*np.where(style[:,None],background_unit_gain[None],1.)
common_background_gain=np.broadcast_to(common_background_gain[:,None],(len(view_id),len(origin))).copy()
residual_gain=np.where(style[:,None],original_No_unit_gain[None],1.)
trend_gain=residual_gain.copy()
signals=trend_gain[:,:,None]*trend_component[None]+foreground_gain[:,:,None]*foreground_basis[view_foreground_index]
signals+=background_gain[:,:,None]*background_component[None]
signals+=common_background_gain[:,:,None]*common[None]+residual_gain[:,:,None]*residual_component[None]
signals[0]=data["signal"]
view_reference_source_id=np.asarray(source_names,dtype=str)[foreground_basis_source_index[view_foreground_index][:,source_index]]
np.testing.assert_allclose(background_gain[4]**2/background_gain[3]**2,16.,rtol=1e-15,atol=0.)
# Retained artificial cohorts also supply at least two experimental peers.
constructible[1:]&=((constructible[1:].astype(float)@membership.T)[:,run]>=2)
nonoriginal_count=constructible[1:].sum(axis=0)
view_weight=np.divide(.5*constructible,np.maximum(nonoriginal_count,1)[None])
view_weight[0]=np.where(nonoriginal_count>0,.5,1.)
view_qualified&=constructible
assert np.allclose(view_weight.sum(axis=0),1.)
assert np.array_equal(signals[0],data["signal"])

background_reference=np.full(view_target.shape,np.nan)
background_reference_kind=np.repeat(nuisance_kind+"_H1_spectral_reference",len(view_id))
bg_profile=harmonic_profile(background_unit,data,period,period_scale,harmonics=1)
native_bg=period[bg_profile["energy_one"].argmax(axis=1)]
native_bg=np.where(background_rms>0.,native_bg,np.nan)
common_low=harmonic_profile(background_component+low*common,data,period,period_scale,harmonics=1)
common_high=harmonic_profile(background_component+high*common,data,period,period_scale,harmonics=1)
background_reference[:]=native_bg
background_reference[1]=period[common_low["energy_one"].argmax(axis=1)]
background_reference[2]=period[common_high["energy_one"].argmax(axis=1)]

vi,ri=np.nonzero(constructible)
prefix=np.where(vi==0,"",np.char.add(view_id[vi],":"))
meta=pd.DataFrame({"trace_id":np.char.add(prefix,origin.trace_id.to_numpy(dtype=str)[ri]),
    "run":np.char.add(prefix,origin.run.to_numpy(dtype=str)[ri]),"condition":origin.condition.to_numpy(dtype=str)[ri],
    "origin_trace_id":origin.trace_id.to_numpy(dtype=str)[ri],"origin_run":origin.run.to_numpy(dtype=str)[ri],
    "reference_source_id":view_reference_source_id[vi,ri],
    "view_id":view_id[vi],"view_weight":view_weight[vi,ri],"independent_unit":independent[ri],
    "is_original":vi==0,"target":view_target[vi,ri],"qualified":view_qualified[vi,ri],
    "background_reference":background_reference[vi,ri],"background_reference_kind":background_reference_kind[vi],
    "view_index":vi,"origin_index":ri}).sort_values("trace_id").reset_index(drop=True)
runs=np.unique(meta.run.to_numpy(dtype=str))
run_meta=meta.drop_duplicates("run").set_index("run").loc[runs]
source_run=pd.Index(old["run_name"]).get_indexer(run_meta.origin_run)
source_trace=pd.Index(old["trace_id"]).get_indexer(meta.origin_trace_id)
assert np.all(source_run>=0) and np.all(source_trace>=0)
queried_run=(old["outer_query"][:,origin_index].astype(float)@membership.T)>0.
training=(~queried_run)[:,pd.Index(data["runs"]).get_indexer(run_meta.origin_run)]
origin_query=old["outer_query"][:,source_trace]
outer_query=origin_query&meta.is_original.to_numpy()[None]
assert not np.any(training[:,pd.Index(runs).get_indexer(meta.run)]&origin_query)
assert int(meta.is_original.sum())==len(old["trace_id"])
pairs=np.asarray([[1,2],[3,4],[5,6]])
pair_names=np.asarray(["measured_common","recorded_target","common_driver"])
if positive_added:
    pairs=np.vstack([pairs,np.asarray([7,8])])
    pair_names=np.concatenate([pair_names,np.asarray(["foreground_period_change"])])
pair_left=pd.Index(runs).get_indexer(np.char.add(np.char.add(view_id[pairs[:,0]],":")[:,None],data["runs"][None]).ravel()).reshape(len(pairs),-1)
pair_right=pd.Index(runs).get_indexer(np.char.add(np.char.add(view_id[pairs[:,1]],":")[:,None],data["runs"][None]).ravel()).reshape(len(pairs),-1)
pair_eligible=(pair_left>=0)&(pair_right>=0)
group_reference_by_view=np.broadcast_to(reference,(len(view_id),len(reference))).copy()
pair_effect=np.zeros((len(pairs),len(reference)))
if positive_added:
    group_reference_by_view[-2:]=positive_period_by_run
    pair_effect[-1]=np.where(pair_eligible[-1],positive_period_by_run[1]-positive_period_by_run[0],0.)
view_reference_count=view_qualified.astype(float)@membership.T
group_reference_by_view=np.divide(np.where(view_qualified,np.nan_to_num(view_target,nan=0.),0.)@membership.T,
    view_reference_count,out=np.full_like(view_reference_count,np.nan),where=view_reference_count>0.)
pair_effect=group_reference_by_view[pairs[:,1]]-group_reference_by_view[pairs[:,0]]
pair_eligible&=np.isfinite(pair_effect)
pair_effect=np.where(pair_eligible,pair_effect,0.)
group_reference=group_reference_by_view[run_meta.view_index.to_numpy(),pd.Index(data["runs"]).get_indexer(run_meta.origin_run)]

# The source interface is in the original No scale. The same waveform basis is
# reused across views and, for ECG, across its four measurement leads.
component_path=folder/(output_tag+"_source_components.npz")
np.savez_compressed(component_path,trace_id=origin.trace_id.to_numpy(dtype=str),
    run=origin.run.to_numpy(dtype=str),condition=origin.condition.to_numpy(dtype=str),
    reference_source_id=reference_source_id,source_names=np.asarray(source_names,dtype=str),source_index=source_index,
    independent_unit=independent,source_independent_unit=independent[source_first],time=time,mask=mask,count=count,
    source_mask=source_mask,foreground_source_waveform=foreground_source_basis,foreground_loading=foreground_loading,
    foreground_basis_names=np.asarray(["measured","shared_measured_donor","period_minus","period_plus"]),
    foreground_basis_source_index=foreground_basis_source_index,
    foreground_basis_source_id=np.asarray(source_names,dtype=str)[foreground_basis_source_index],
    donor_index_by_cohort=donor_index,donor_source_id_by_cohort=np.asarray(source_names,dtype=str)[donor_index],
    donor_selection=np.asarray("qualified_distinct_source_closest_to_cohort_median_period; source_id_breaks_ties"),
    foreground_resample_gain=np.concatenate([np.ones((1,len(source_names))),resample_gain]),
    background=background_component,residual=residual_component,trend=trend_component,common_background=common,
    candidate_view_id=view_id,view_foreground_index=view_foreground_index,foreground_gain=foreground_gain,background_gain=background_gain,
    reference_source_id_by_view=view_reference_source_id,
    common_background_gain=common_background_gain,residual_gain=residual_gain,trend_gain=trend_gain,
    foreground_period=view_target,foreground_period_qualified=view_qualified,
    foreground_time_scale=source_time_scale[view_foreground_index][:,source_index],
    foreground_time_shift=source_time_shift[view_foreground_index][:,source_index],
    source_reference_period=source_reference,source_qualified=source_qualified,extension_period=source_clock,
    extension_period_is_qualified_reference=source_qualified,view_constructible=constructible,
    original_qualified=qualified,original_target=target,unit=np.asarray(unit),
    source_harmonics=source_geometry["harmonics"],nuisance_reference_kind=np.asarray(nuisance_kind),
    component_order=np.asarray(["FG","BG","residual","trend","common_BG"]),
    observation_normalization=np.asarray("base_components_raw_No_instrument_scale; explicit_per_measurement_view_gains; model_preprocessing_not_applied"),
    foreground_normalization=np.asarray("measured_FG_unit_total_RMS_then_restore_total_RMS_after_actual_time_warp; physical_loading_preserved"),
    interpolation=np.asarray("linear_on_actual_observed_FG_times; nearest_measured_cycle_extension_only_outside_support"),
    period_truth=np.asarray("known_time_dilation_of_original_reference_clock; nonstationary_waveform_and_unknown_qualification_retained"),
    unit_source_names=np.asarray(unit_names,dtype=str),unit_source_mean_period=unit_reference,
    unit_source_period_sd=unit_spread,unit_qualified_source_count=unit_reference_count,
    unit_positive_period_scale=unit_period_scale)

# Arrow dictionaries reuse observation indices; each view is written to the
# same Parquet file sequentially, without expanding ECG identifiers in memory.
row_record,row_time=np.nonzero(mask)
row_codes=pa.array(row_record.astype(np.int32))
record_column=lambda values:pa.DictionaryArray.from_arrays(row_codes,pa.array(values,type=pa.string()))
origin_trace=origin.trace_id.to_numpy(dtype=str)
origin_run=origin.run.to_numpy(dtype=str)
fixed_columns=[record_column(origin.condition.to_numpy(dtype=str)),pa.array(time[row_time]),
    record_column(origin_trace),record_column(origin_run),record_column(independent)]
schema=pa.schema([("trace_id",pa.dictionary(pa.int32(),pa.string())),("run",pa.dictionary(pa.int32(),pa.string())),
    ("condition",pa.dictionary(pa.int32(),pa.string())),("x",pa.float64()),("signal_y",pa.float64()),
    ("origin_trace_id",pa.dictionary(pa.int32(),pa.string())),("origin_run",pa.dictionary(pa.int32(),pa.string())),
    ("view_id",pa.dictionary(pa.int32(),pa.string())),("view_weight",pa.float64()),
    ("independent_unit",pa.dictionary(pa.int32(),pa.string())),("is_original",pa.bool_())])
writer=pq.ParquetWriter(folder/(output_tag+"_views.parquet"),schema,compression="zstd",use_dictionary=True)
write_view=lambda v:writer.write_table(pa.Table.from_arrays([
    record_column(np.char.add("" if v==0 else view_id[v]+":",origin_trace)),
    record_column(np.char.add("" if v==0 else view_id[v]+":",origin_run)),fixed_columns[0],fixed_columns[1],
    pa.array(signals[v,row_record,row_time]),fixed_columns[2],fixed_columns[3],
    record_column(np.repeat(view_id[v],len(origin))),pa.array(view_weight[v,row_record]),fixed_columns[4],
    pa.array(np.repeat(v==0,len(row_record)))],schema=schema).filter(pa.array(constructible[v,row_record])),
    row_group_size=CONFIG["row_group_size"])
write_indices=np.flatnonzero(constructible.any(axis=1))
if args.coherent_driver:
    write_indices=write_indices[write_indices>=5]
list(map(write_view,write_indices))
writer.close()
np.savez_compressed(folder/(output_tag+"_context.npz"),trace_id=meta.trace_id.to_numpy(dtype=str),
    target=meta.target.to_numpy(),qualified=meta.qualified.to_numpy(),run_name=runs,
    outer_train_run=training,outer_query=outer_query,origin_query=origin_query,
    outer_protocol=old["outer_protocol"],outer_condition=old["outer_condition"],outer_control_run=old["outer_control_run"],
    is_original=meta.is_original.to_numpy(),origin_trace_id=meta.origin_trace_id.to_numpy(dtype=str),
    origin_run=meta.origin_run.to_numpy(dtype=str),view_id=meta.view_id.to_numpy(dtype=str),view_weight=meta.view_weight.to_numpy(),
    reference_source_id=meta.reference_source_id.to_numpy(dtype=str),
    independent_unit=meta.independent_unit.to_numpy(dtype=str),background_reference=meta.background_reference.to_numpy(),
    background_reference_kind=meta.background_reference_kind.to_numpy(dtype=str),
    nuisance_reference_kind=np.asarray(nuisance_kind),unit=np.asarray(unit),period_bounds=np.asarray([period_min,period_max]),
    candidate_origin_trace_id=origin_trace,candidate_view_id=view_id,view_constructible=constructible,
    foreground_phase_sd=phase_sd,driver_residual_rms=residual_rms,driver_reference_period=driver_period,
    driver_qualified=driver_ok&reference_valid,optical_to_No_gain=gain,
    donor_source_id_by_cohort=np.asarray(source_names,dtype=str)[donor_index],
    source_component_path=np.asarray(component_path.relative_to(root).as_posix()),
    original_qualified=qualified,original_target=target,group_reference=group_reference,
    group_reference_by_view=group_reference_by_view,group_reference_origin_run=data["runs"],
    pairs=pairs,pair_names=pair_names,
    pair_effect=pair_effect,pair_eligible=pair_eligible,pair_left_run=pair_left,pair_right_run=pair_right,**positive_context)
definition={"dataset":args.dataset,"original_records":len(origin),"training_records_with_views":len(meta),
    "view_counts":dict(zip(view_id,constructible.sum(axis=1).tolist())),"unit":unit,"period_bounds":[period_min,period_max],
    "nuisance_reference_kind":nuisance_kind,"original_weight":.5,"total_constructible_artificial_weight":.5,
    "background_reference_kind_by_view":dict(zip(view_id.tolist(),background_reference_kind.tolist())),
    "common_background_reference":"H1 peak of the explicitly transformed measured optical BG or estimated rank1 ECG nuisance; ECG has no measured BG truth.",
    "original_only_weight":1.,"valid_zero_phase_scatter_records":int(np.sum(phase_sd<=tiny)),
    "phase_transformation":"Shared measured FG donor closest to cohort median qualified-source period; receiver No loading/sign preserved. ECG one-source cohorts are legal identity boundaries.",
    "target_eligibility":"Original qualification remains unchanged. Shared-FG views use the actual donor reference; copied sources add no independent observations. Positive labels are r_unit times qualified original periods. Unknown periods remain unknown.",
    "foreground_waveform":"Measured FG with actual-time linear interpolation and neighbouring measured-cycle boundary extension; total FG energy and physical No loading retained.",
    "source_component_path":component_path.relative_to(root).as_posix(),
    "source_component_axes":{"foreground_source_waveform":"basis,distinct_source,time","background_residual_trend_common":"measurement,time","gains":"view,measurement","foreground_time_scale_shift":"view,measurement"},
    "constructed_view_gain_policy":"direct/shared/positive FG and BG have explicit unit-total-RMS gains; real residual and trend retain their original No-relative scale. Native/common remain in original units.",
    "source_counts":{"original_qualified_measurements":int(qualified.sum()),"distinct_qualified_FG_sources":int(source_qualified.sum()),"independent_units":len(unit_names)},
    "component_reconstruction_max_abs_error":float(np.max(abs(foreground_component+background_component+residual_component+trend_component-data["signal"]))),
    "raw_BG_variance_ratio_direct_high_low":float(high**2/low**2),
    "evaluation":"Original observations only; every view inherits the origin experiment or subject exclusion.",
    "effective_sample_size":"Artificial views retain origin_trace_id and independent_unit; they add no independent observations."}
if args.positive_shifts:
    definition.update({"positive_shifts_requested":True,"positive_views_added":positive_added,
        "positive_pair_effect":"Same r_unit=1+-SD_unit/(2*mean_period_unit) for every actual FG waveform of a unit. Mean and SD count distinct qualified sources. Pair effects use the actual transformed qualified-reference means.",
        "positive_background":"Identical measured/estimated BG and real residual with the same explicitly recorded gains in both views.",
        "positive_shift_run_count":int(np.sum(pair_eligible[-1])) if positive_added else 0,
        "positive_shift_zero_sd_run_count":int(np.sum(np.isfinite(period_shift)&(period_shift==0.)))})
(folder/(args.dataset+("_positive" if args.positive_shifts else "")+("_coherent_driver" if args.coherent_driver else "")+"_view_definition.json")).write_text(json.dumps(definition,indent=2)+"\n",encoding="utf-8")
print("SOURCE_REFERENCE_VIEWS",json.dumps(definition),flush=True)
