"""Run native measured-period APIs and frozen scalar-label readouts on fixed inputs."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1],
    "java":"build/java_runtime_25/jdk-25.0.4.1+1-jre/bin/java.exe",
    "maps":"data/sota_dev/recomputed_clean/lemna_task_period_classical_predictions.npz",
    "model":"data/sota_dev/recomputed_clean/lemna_task_period_source_assignment_frequency_geometry_model.npz",
    "datasets":{"controlled":"data/sota_dev/final_model_controlled","wang":"data/external/wang2026_mechanism","shan":"data/external/shan2020_semisynthetic","single":"data/external/lemna_watanabe2021/final_0_2_4_classical","dual":"data/external/lemna_watanabe2021"},
    "input_overrides":{"single":"data/sota_dev/recomputed_clean/lemna2021_single_reporter_input.parquet","dual":"data/external/lemna_watanabe2021/dual_reporter_observations.parquet"}},
    "threads":os.cpu_count(),"plot":{},"java_heap":"8g","period_tolerance_h":1e-8,"max_iterations":200,
    "shorter_counts":{104:95,145:120},"missing_patterns":5,"missing_stride":11}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
import argparse,subprocess,time as timing,sys
import numpy as np
import pandas as pd
from scipy.optimize import elementwise
sys.path.insert(0,str(CONFIG["paths"]["root"]/"software"))
from source_assignment_core import interpolate_mean
from period_support import project_periods_with_mean
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset",required=True,choices=CONFIG["paths"]["datasets"])
parser.add_argument("--java",type=Path,default=CONFIG["paths"]["root"]/CONFIG["paths"]["java"])
parser.add_argument("--output-dir",type=Path)
parser.add_argument("--raw-only",action="store_true",help="Run every unlabelled native API without a reference-trained readout")
parser.add_argument("--acquisition",choices=["native","shorter","missing"],default="native")
args=parser.parse_args()
root=CONFIG["paths"]["root"]
input_folder=root/CONFIG["paths"]["datasets"][args.dataset]
folder=root/args.output_dir if args.output_dir is not None else input_folder
work=root/"build/classical_transfer"/(args.dataset+"_"+args.acquisition)
work.mkdir(parents=True,exist_ok=True)
folder.mkdir(parents=True,exist_ok=True)
input_path=root/CONFIG["paths"]["input_overrides"].get(args.dataset,input_folder/"inputs.parquet")
query=pd.read_parquet(input_path)
if args.dataset=="dual":
    query=query[["trace_id","run","time_h","no"]].rename(columns={"time_h":"x","no":"signal_y"}).assign(condition="observational")
query=query[["trace_id","run","condition","x","signal_y"]].sort_values(["trace_id","x"]).reset_index(drop=True)
if args.acquisition!="native":
    assert args.dataset=="controlled" and args.raw_only
    truth=pd.read_csv(input_folder/"truth_cells.csv",encoding="utf-8")
    sample=query.groupby("trace_id").cumcount()
    original=query.trace_id.map(query.groupby("trace_id").size())
    pair=pd.Series(pd.factorize(truth.pair_id.str.replace(r"_l\d+(?=_|$)","",regex=True),sort=True)[0],index=truth.trace_id)
    pattern=query.trace_id.map(pair)%CONFIG["missing_patterns"]
    keep_short=sample<original.map(CONFIG["shorter_counts"])
    keep_missing=~((sample>0)&(sample<original-1)&(sample%CONFIG["missing_stride"]==pattern))
    query=query.loc[{"shorter":keep_short,"missing":keep_missing}[args.acquisition]].copy()
metadata=query[["trace_id","run","condition"]].drop_duplicates().set_index("trace_id")
query["time_relative_h"]=query.x-query.groupby("run").x.transform("min")
table=query.pivot(index="trace_id",columns="time_relative_h",values="signal_y")
trace=table.index.to_numpy(dtype=str)
metadata=metadata.loc[trace]
time=table.columns.to_numpy(dtype=float)
signal=table.to_numpy(dtype=float)
mask=np.isfinite(signal)
count=mask.sum(axis=-1)
models=np.load(root/CONFIG["paths"]["model"],allow_pickle=False)
period=models["period"]
# The original APIs receive every observed native timestamp and raw No intensity.
query.sort_values(["trace_id","time_relative_h"])[["trace_id","time_relative_h","signal_y"]].to_csv(work/"native_input.tsv",sep="\t",header=False,index=False,encoding="utf-8",lineterminator="\n")
classpath=os.pathsep.join([str(root/"scripts"),str(root/"data/sota_dev/competitors/official_jars/*")])
started=timing.perf_counter()
log=(work/"native_api.log").open("wb")
subprocess.run([str(args.java),"-Xmx"+CONFIG["java_heap"],"-Djava.util.concurrent.ForkJoinPool.common.parallelism="+str(CONFIG["threads"]),
    "-cp",classpath,"OfficialPeriodRunner",str(work/"native_input.tsv"),str(folder/"classical_native_results.csv"),
    str(period.min()),str(period.max()),"0","raw"],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
log.close()
native=pd.read_csv(folder/"classical_native_results.csv",encoding="utf-8")
assert native.groupby("method").trace_id.nunique().eq(len(trace)).all()
native=native.rename(columns={"period_h":"estimate_h","status":"method_status"})
native=native[["trace_id","method","estimate_h","method_status"]]
print("NATIVE_CLASSICAL_APIS",args.dataset,len(trace),timing.perf_counter()-started,native.groupby(["method","method_status"]).size().to_dict(),flush=True)
# A dense H1 grid initializes a continuous one-harmonic variable projection.
center=(time[None]-(mask*time[None]).sum(axis=-1)[:,None]/count[:,None])*mask
trend=np.stack([mask/np.sqrt(count[:,None]),center/np.sqrt((center**2).sum(axis=-1))[:,None]],axis=-1)
y=np.where(mask,signal,0.)
y-=np.einsum("nti,ni->nt",trend,np.einsum("nti,nt->ni",trend,y,optimize=True),optimize=True)
y/=np.sqrt((y**2).sum(axis=-1)/count)[:,None]
layout_mask,layout=np.unique(mask,axis=0,return_inverse=True)
first=np.unique(layout,return_index=True)[1]
layout_trend=trend[first]
angle=2*np.pi*time[None]/period[:,None]
harmonic=np.stack([np.cos(angle),np.sin(angle),np.cos(2*angle),np.sin(2*angle)],axis=-1)
design=harmonic[None]*layout_mask[:,None,:,None]-np.einsum("lti,lpij->lptj",layout_trend,np.einsum("lti,ptj->lpij",layout_trend,harmonic,optimize=True),optimize=True)
gram=np.einsum("lpti,lptj->lpij",design,design,optimize=True)
rhs=np.einsum("pti,nt->npi",harmonic,y,optimize=True)
h1=(rhs[:,:,:2]*np.linalg.solve(gram[layout,:,:2,:2],rhs[:,:,:2,None])[...,0]).sum(axis=-1)
h2=(rhs*np.linalg.solve(gram[layout],rhs[...,None])[...,0]).sum(axis=-1)
initial=period[h1.argmax(axis=-1)]
loss=lambda p,i:(
    angle:=2*np.pi*time[None]/p[:,None],
    h:=np.stack([np.cos(angle),np.sin(angle)],axis=-1)*mask[i,:,None],
    h:=h-np.einsum("nti,nij->ntj",trend[i],np.einsum("nti,ntj->nij",trend[i],h,optimize=True),optimize=True),
    g:=np.einsum("nti,ntj->nij",h,h,optimize=True),
    r:=np.einsum("nti,nt->ni",h,y[i],optimize=True),
    -(r*np.linalg.solve(g,r[...,None])[...,0]).sum(axis=-1))[-1]
step=period[1]-period[0]
lo=np.maximum(period.min(),initial-step);hi=np.minimum(period.max(),initial+step)
bracket=elementwise.bracket_minimum(loss,(lo+hi)/2.,xl0=lo,xr0=hi,xmin=lo,xmax=hi,args=(np.arange(len(lo)),),maxiter=CONFIG["max_iterations"])
assert np.isin(bracket.status,[0,-1]).all(),np.unique(bracket.status,return_counts=True)
bx=np.stack(bracket.bracket);bf=np.stack(bracket.f_bracket)
continuous=np.take_along_axis(bx,bf.argmin(axis=0)[None],axis=0)[0]
active=bracket.success
minimum=elementwise.find_minimum(loss,tuple(bx[:,active]),args=(np.flatnonzero(active),),tolerances={"xatol":CONFIG["period_tolerance_h"],"xrtol":0.,"fatol":0.,"frtol":0.},maxiter=CONFIG["max_iterations"])
assert minimum.success.all(),np.unique(minimum.status,return_counts=True)
continuous[active]=minimum.x
harmonic_names=np.asarray(["native_H1","native_H2","native_H1_continuous"])
harmonic_values=np.stack([initial,period[h2.argmax(axis=-1)],continuous],axis=-1)
harmonic_predictions=pd.DataFrame({"trace_id":np.repeat(trace,len(harmonic_names)),"method":np.tile(harmonic_names,len(trace)),"estimate_h":harmonic_values.ravel(),"method_status":"ok"})
if args.raw_only:
    predictions=pd.concat([native,harmonic_predictions],ignore_index=True).merge(metadata.reset_index(),on="trace_id",validate="many_to_one")
    predictions["input_protocol"]="all_observed_No_no_target_reference"
    predictions["readout_training"]="none"
    predictions["acquisition"]=args.acquisition
    predictions.to_csv(folder/"classical_predictions.csv",index=False,encoding="utf-8",lineterminator="\n")
    print("RAW_CLASSICAL_TRANSFER_COMPLETE",args.dataset,args.acquisition,len(trace),len(query),len(predictions),flush=True)
    raise SystemExit(0)
# Only the frozen original-cohort optical labels fit this shared readout.
saved=np.load(root/CONFIG["paths"]["maps"],allow_pickle=False)
author_methods=np.asarray(["MESA","MFourFit"])
author=native.pivot(index="trace_id",columns="method",values="estimate_h").reindex(index=trace,columns=author_methods).to_numpy().T
available=np.isfinite(author)
runs,run_index=np.unique(metadata.run.to_numpy(dtype=str),return_inverse=True)
member=(runs[:,None]==metadata.run.to_numpy(dtype=str)[None]).astype(float)
run_count=available.astype(float)@member.T
assert (run_count>0).all()
mean=np.where(available,author,0.)@member.T/run_count
run_length=pd.Series(count,index=metadata.run).groupby(level=0).first().reindex(runs).to_numpy()
head=pd.Index(saved["full_fit_length"]).get_indexer(run_length)
method_index=pd.Index(saved["native_methods"]).get_indexer(author_methods)
assert (head>=0).all() and (method_index>=0).all()
x=saved["full_calibration_x"][method_index[:,None],head[None]]
z=saved["full_calibration_y"][method_index[:,None],head[None]]
n=saved["full_calibration_n"][method_index[:,None],head[None]]
calibrated=interpolate_mean(mean[:,:,None],x,z,n,saved["full_identity_tails"][head][None])[:,:,0]
a=saved["full_within_coefficient"][method_index[:,None],head[None]]
deviation=author-mean[:,run_index]
unbounded=calibrated[:,run_index]+a[:,run_index]*deviation
supported=np.stack(list(map(lambda i:project_periods_with_mean(np.where(available[i],unbounded[i],0.)[None],calibrated[i,None],member*available[i][None],period.min(),period.max())[0],np.arange(len(author_methods)))))
supported=np.where(available,supported,np.nan)
matched=pd.DataFrame({"trace_id":np.tile(trace,len(author_methods)),"method":np.repeat(np.char.add("target_mean__",author_methods),len(trace)),"estimate_h":supported.ravel(),"method_status":np.where(available.ravel(),"ok","native_no_output")})
predictions=pd.concat([native,harmonic_predictions,matched],ignore_index=True).merge(metadata.reset_index(),on="trace_id",validate="many_to_one")
predictions["input_protocol"]="all_native_hourly_No_no_target_reference"
predictions["readout_training"]=np.where(predictions.method.str.startswith("target_mean__"),"original_Lemna_optical_labels_only","none") 
assert not predictions.duplicated(["trace_id","method"]).any()
predictions.to_csv(folder/"classical_predictions.csv",index=False,encoding="utf-8",lineterminator="\n")
print("FIXED_CLASSICAL_TRANSFER_COMPLETE",args.dataset,len(predictions),timing.perf_counter()-started,flush=True)
