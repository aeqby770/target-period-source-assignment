"""Create real-waveform coherent training views through the matched prediction chain."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1]},"threads":os.cpu_count(),"plot":{},"period_shift_h":2.}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
import sys,ast,inspect,time,argparse
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from scipy.signal import hilbert
from scipy.special import softmax,expit
root=CONFIG["paths"]["root"];clean=root/"data/sota_dev/recomputed_clean"
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--foreground-presence",action="store_true")
parser.add_argument("--causal-views",action="store_true")
args=parser.parse_args()
out=clean/("matched_causal_views" if args.causal_views else "matched_foreground_presence" if args.foreground_presence else "matched_real_coherence");out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/"software"))
import source_amplitude_normalization as normalization
from source_assignment_core import role_geometry,role_assignment_evidence,role_score_features,measured_peer_partial_power
torch.set_num_threads(CONFIG["threads"])
data=np.load(clean/"lemna_task_period_supervised_context.npz",allow_pickle=False)
optical=np.load(clean/"lemna_task_period_optical_source_geometry.npz",allow_pickle=False)
gain=np.load(clean/"lemna_optical_source_gain.npz",allow_pickle=False)["optical_gain"]
full=dict(np.load(clean/"lemna_task_period_source_assignment_frequency_geometry_model.npz",allow_pickle=False))
replay=dict(np.load(clean/"lemna_task_period_source_assignment_frequency_geometry_replay_model.npz",allow_pickle=False))
native=np.load(clean/"correlated_order_all_rank_one/real_features.npz",allow_pickle=False)
np.testing.assert_array_equal(data["trace_id"],native["trace_id"])
np.testing.assert_array_equal(data["trace_id"],optical["trace_id"])
time_h=data["time"];mask=data["mask"];count=mask.sum(axis=-1);period=full["period"]
membership=data["run_membership"];average=membership/membership.sum(axis=-1)[:,None];run=data["run_index"]
foreground=optical["normalized_sources"][1]*gain[run,0,None]
background=optical["normalized_sources"][2]*gain[run,1,None]
design_f,layout_f,joint_f,rhs_f=role_geometry(time_h,count,foreground,foreground,period)
grid=np.arange(len(period));gf=joint_f[layout_f[:,None],grid[None],grid[None],:2,:2]
power_f=np.sum(rhs_f[:,:,:,0]*np.linalg.solve(gf,rhs_f[:,:,:,0,None])[...,0],axis=-1)
reference_period=np.where(data["qualified"],data["target"],period[power_f.argmax(axis=-1)])
center=time_h[None]-(count[:,None]-1.)/2.
angle=2*np.pi*center/reference_period[:,None]
hf=np.stack([np.cos(angle),np.sin(angle)],axis=-1)*mask[:,:,None]
trend_center=(time_h[None]-(mask*time_h[None]).sum(axis=-1)[:,None]/count[:,None])*mask
trend=np.stack([mask/np.sqrt(count[:,None]),trend_center/np.sqrt((trend_center**2).sum(axis=-1))[:,None]],axis=-1)
hf-=np.einsum("nti,nij->ntj",trend,np.einsum("nti,ntj->nij",trend,hf,optimize=True),optimize=True)
gram_f=np.einsum("nti,ntj->nij",hf,hf,optimize=True)
coef_f=np.linalg.solve(gram_f,np.einsum("nti,nt->ni",hf,foreground,optimize=True)[...,None])[...,0]
phase=np.angle(coef_f[:,0]-1j*coef_f[:,1])
analytic=hilbert(foreground,axis=-1)
rotated=analytic*np.exp(-1j*phase[:,None]);aligned=rotated.real
long=count==145
analytic_error=np.max(abs(hilbert(aligned[long],axis=-1)-rotated[long]))
spectrum_error=np.max(abs(abs(np.fft.rfft(aligned[long],axis=-1))-abs(np.fft.rfft(foreground[long],axis=-1))))
assert analytic_error<1e-10 and spectrum_error<1e-10,(analytic_error,spectrum_error)
foreground_unit=aligned/np.sqrt((aligned**2).sum(axis=-1)/count)[:,None]
shared=(average@background)[run]
background_unit=shared/np.sqrt((shared**2).sum(axis=-1)/count)[:,None]
optical_residual=data["mixed"]-foreground-background
alpha=np.asarray([.5,2.])
views=foreground_unit[None]+alpha[:,None,None]*background_unit[None]+optical_residual[None]
views=np.where(long[None,:,None],views,data["mixed"][None])
np.testing.assert_allclose(views[1,long]-views[0,long],1.5*background_unit[long],rtol=0.,atol=1e-12)
target_views=np.broadcast_to(data["target"],(3,len(count))).copy()
if args.causal_views:
    foreground_reference=(membership@data["target"])/(membership@data["qualified"])
    generated_period=np.where(data["qualified"],data["target"],foreground_reference[run])
    shifted_period=generated_period[None]+np.asarray([-CONFIG["period_shift_h"],CONFIG["period_shift_h"]])[:,None]
    assert np.all((shifted_period>period.min())&(shifted_period<period.max()))
    shifted_wave=np.cos(2*np.pi*center[None]/shifted_period[:,:,None])*mask[None]
    shifted_wave-=np.einsum("nti,vni->vnt",trend,
        np.einsum("nti,vnt->vni",trend,shifted_wave,optimize=True),optimize=True)
    shifted_wave/=np.sqrt(np.sum(shifted_wave**2,axis=-1)/count[None])[:,:,None]
    shifted_view=shifted_wave+background_unit[None]+optical_residual[None]
    shifted_view=np.where(long[None,:,None],shifted_view,data["mixed"][None])
    views=np.concatenate([views,shifted_view])
    shifted_target=np.where(long[None],shifted_period,data["target"][None])
    target_views=np.concatenate([target_views,shifted_target])
    generated_mean=shifted_period@average.T
    expected_mean=foreground_reference[None]+np.asarray([-CONFIG["period_shift_h"],CONFIG["period_shift_h"]])[:,None]
    np.testing.assert_allclose(generated_mean,expected_mean,rtol=0.,atol=1e-12)
tags=np.asarray(["v1","v2","v3","v4"]) if args.causal_views else np.asarray(["v1","v2"])
nview=1+len(views)

vi,ri,ti=np.nonzero(np.broadcast_to(mask,views.shape))
query=pd.DataFrame({"trace_id":np.char.add(np.char.add(tags[vi],"|"),data["trace_id"][ri]),
    "run":np.char.add(np.char.add(tags[vi],"|"),data["run_name"][run[ri]]),
    "condition":data["run_condition"][run[ri]],"x":time_h[ti],"signal_y":views[vi,ri,ti]})
np.savez_compressed(out/"feature_parent_model.npz",**full)
entry_path=root/"software/predict_source_assignment.py"
source=entry_path.read_text(encoding="utf-8")
read_line='query = pd.read_parquet(args.input,columns=["trace_id","run","condition","x","signal_y"])'
assert source.count(read_line)==1
source=source.replace(read_line,"query=input_query.copy()")
prefix=source[:source.index("shape_phase,diagnostic=role_assignment_evidence")]
core_block=source[source.index("core_logit_all=np.einsum"):source.index("core_log_mass=")]
entry={"__file__":str(entry_path),"__name__":"__matched_real_source_features__","input_query":query}
# Reuse the tested AR1 clock-order adapter verbatim.
probe_path=root/"experiments/probe_correlated_clock_order.py"
probe_source=probe_path.read_text(encoding="utf-8");tree=ast.parse(probe_source)
adapter_node=next(n for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1
    and isinstance(n.targets[0],ast.Attribute) and isinstance(n.targets[0].value,ast.Name)
    and n.targets[0].value.id=="normalization" and n.targets[0].attr=="balance_source_amplitudes")
adapter=ast.get_source_segment(probe_source,adapter_node)
adapter_namespace={"np":np,"expit":expit,"normalization":normalization,
    "actual_balance":normalization.balance_source_amplitudes,"balance_result":{},
    "entry":entry,"diagnostics":{},"args":SimpleNamespace(all_lengths=True)}
exec(compile(adapter,"<same_tested_correlated_order>","exec"),adapter_namespace)
sys.argv=[str(entry_path),"--model",str(out/"feature_parent_model.npz"),
    "--input",str(clean/"lemna_task_period_acf_input.parquet"),"--output",str(out/"unused.csv")]
started=time.perf_counter()
print("MATCHED_REAL_VIEWS_START",len(query),len(views)*len(count),"analytic_error",analytic_error,
    "spectrum_error",spectrum_error,flush=True)
exec(compile(prefix+"\n"+core_block,"<actual_prediction_feature_entry>","exec"),entry)
expected=np.char.add(np.char.add(tags[:,None],"|"),data["trace_id"][None]).ravel()
order=pd.Index(entry["trace_id"]).get_indexer(expected);assert (order>=0).all()
base=np.concatenate([native["base_features"],entry["base_features"][order]])
left,right=np.triu_indices(base.shape[-1])
raw=np.concatenate([base,base[:,:,left]*base[:,:,right]],axis=-1)
native_design,native_layout,joint,native_rhs=role_geometry(time_h,count,native["mixed"],native["peer"],period)
native_power=measured_peer_partial_power(time_h,count,native["mixed"],native["peer"],
    native_design,native_layout,coherent=True,half_gram=True)
native_coherent=native["base_features"].copy();native_coherent[:,:,11]=native_power
native_coherent=np.concatenate([native_coherent,native_coherent[:,:,left]*native_coherent[:,:,right]],axis=-1)
coherent=np.concatenate([native_coherent,entry["coherent_features"][order]])
all_rhs=np.concatenate([native_rhs,entry["rhs"][order]])
all_layout=np.tile(native_layout,nview)
np.testing.assert_allclose(joint,entry["joint_gram"],rtol=0.,atol=1e-12)
bg_coef=np.concatenate([full["background_coefficient"],replay["background_coefficient"]])
bg_probability=softmax(np.einsum("npd,gd->gnp",raw,bg_coef,optimize=False),axis=-1)
cov=np.concatenate([full["role_covariance"],replay["role_covariance"]])
gpu=inspect.getsource(role_assignment_evidence)
gpu=gpu.replace("dtype=torch.float64)","dtype=torch.float64,device='cuda')")
gpu=gpu.replace("dtype=torch.long)","dtype=torch.long,device='cuda')")
gpu=gpu.replace("torch.arange(len(layout))","torch.arange(len(layout),device='cuda')")
gpu=gpu.replace("return features.numpy(), diagnostic.numpy()","return features.cpu().numpy(), diagnostic.cpu().numpy()")
ns={"np":np,"torch":torch};exec(compile(gpu,"<same_frozen_role_likelihood_cuda>","exec"),ns)
print("MATCHED_REAL_ROLE_START",len(cov),len(base),flush=True)
role_presence=np.r_[full["core_coherent_peer"],replay["core_coherent_peer"]]&args.foreground_presence
reduction=np.where(role_presence,"foreground_presence","expected_bf")
phase_features,_=ns["role_assignment_evidence"](joint,all_rhs,all_layout,cov,bg_probability,len(period)//2,len(period)//2,reduction=reduction)
role=role_score_features(phase_features,base).reshape(len(cov),nview,len(count),len(period),-1)
native_noise=pd.read_csv(clean/"correlated_order_all_gls_peer/native_deployment_order.csv").set_index("trace_id").loc[data["trace_id"],"rho"].to_numpy()
view_rho=np.concatenate([native_noise,adapter_namespace["diagnostics"]["rho"][order]]).reshape(nview,len(count))
np.savez_compressed(out/"features.npz",trace_id=data["trace_id"],period=period,rho=view_rho,
    raw=raw.reshape(nview,len(count),len(period),-1),coherent=coherent.reshape(nview,len(count),len(period),-1),
    role=role,target=data["target"],target_views=target_views,qualified=data["qualified"],mixing_alpha=alpha,
    period_shift_h=np.asarray(CONFIG["period_shift_h"]),causal_views=np.asarray(args.causal_views),
    role_presence=role_presence,base_features=base.reshape(nview,len(count),len(period),-1),
    rhs=all_rhs.reshape(nview,len(count),len(period),2,2),layout=native_layout,joint_gram=joint,
    time=time_h,count=count,background_coefficient=bg_coef,
    mixed=np.concatenate([native["mixed"],entry["mixed"][order]]).reshape(nview,len(count),-1),
    peer=np.concatenate([native["peer"],entry["peer"][order]]).reshape(nview,len(count),-1),
    analytic_error=analytic_error,spectrum_error=spectrum_error,
    definition=np.asarray(("Training-only causal views add exact unit-RMS sine foreground clocks at qualified reference periods plus/minus 2h; unknown native periods are assigned the within-run qualified mean only for these generated clocks; same observed background/noise; 104 records remain unchanged. " if args.causal_views else "")+"Original observed foreground waveforms undergo constant analytic-phase rotation, preserving Fourier amplitudes and instantaneous frequency; unit-RMS foreground and actual shared background with alpha .5/2, actual optical reconstruction residual; original native baseline and full production amplitude/AR1-order representation; original masks and labels; 104-point coefficients remain frozen"))
print("MATCHED_REAL_VIEWS_DONE",time.perf_counter()-started,"shapes",raw.shape,role.shape,flush=True)
