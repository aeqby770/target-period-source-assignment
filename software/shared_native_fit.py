"""Fit shared native-time period scores without using acquisition length as a label stratum."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1]},"threads":os.cpu_count(),"plot":{},
    "solver":{"maxiter":900,"ftol":1e-11,"gtol":1e-6,"maxls":30}}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
import argparse,time
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from source_assignment_core import role_geometry,role_assignment_evidence,role_score_features,fit_monotone_mean
torch.set_num_threads(CONFIG["threads"])
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--contexts",choices=["full","outer","inner"],default="full")
parser.add_argument("--query-length",type=int,default=0)
parser.add_argument("--parent-model",type=Path)
parser.add_argument("--parent-head",type=int,default=145)
args=parser.parse_args()
root=CONFIG["paths"]["root"]
output=root/"data/sota_dev/recomputed_clean"
data=np.load(output/"lemna_task_period_supervised_context.npz",allow_pickle=False)
cache=np.load(output/"lemna_task_period_transfer_period_profile_context.npz",allow_pickle=False)
phase=np.load(output/"lemna_task_period_transfer_period_density_quadrature_fixed.npz",allow_pickle=False)
inner=np.load(root/"data/sota_dev/recomputed_phase/lemna_task_period_source_crossfit_inner_fits.npz",allow_pickle=False)
background=np.load(output/"lemna_task_period_background_density_fit.npz",allow_pickle=False)
outer_query=data["outer_query"]
inner_query=inner["inner_query"]
owner=inner["inner_owner"]
allowances={"full":np.ones((1,len(data["trace_id"])),dtype=bool),
    "outer":~outer_query,"inner":~outer_query[owner]&~inner_query}
queries={"full":np.zeros((1,len(data["trace_id"])),dtype=bool),"outer":outer_query,"inner":inner_query}
allow=allowances[args.contexts]
query=queries[args.contexts]
case_index=np.arange(len(allow))
context_suffix=""
if args.contexts=="outer" and args.query_length>0:
    case_index=np.flatnonzero(data["outer_length"]==args.query_length)
    allow=allow[case_index]
    query=query[case_index]
    context_suffix="_"+str(args.query_length)
train=allow&data["qualified"][None]
assert not np.any(train&query)
assert args.contexts!="inner" or not np.any(train&outer_query[owner])
membership=data["run_membership"]
average=membership/membership.sum(axis=1)[:,None]
train_run=(allow.astype(float)@membership.T)>0
reference=(membership@data["target"])/(membership@data["qualified"].astype(float))
weight=train/train.sum(axis=1)[:,None]
n=train.sum(axis=1)
b=len(train)
period=cache["period"]
step=period[1]-period[0]
base=np.concatenate([cache["base_features"],phase["quadrature"][:,:,None]],axis=-1)
left,right=np.triu_indices(base.shape[-1])
raw=np.concatenate([base,base[:,:,left]*base[:,:,right]],axis=-1)
phase_terms=np.r_[np.arange(base.shape[-1])==12,(left==12)|(right==12)]
feature_mean=np.einsum("gn,nd->gd",weight,raw.mean(axis=1),optimize=True)
feature_scale=np.sqrt(np.maximum(np.einsum("gn,nd->gd",weight,(raw**2).mean(axis=1),optimize=True)-feature_mean**2,0.))
feature_scale=np.where(feature_scale>0,feature_scale,1.)
teacher=np.maximum(1.-abs(period[None]-data["target"][:,None])/step,0.)
teacher_cdf=teacher.cumsum(axis=-1)[:,:-1]
cdf_constant=step*np.sum(teacher_cdf*(1.-teacher_cdf),axis=-1)
target_mean=weight@data["target"]
target_sd=np.sqrt(weight@(data["target"]**2)-target_mean**2)
rival=np.stack([base[:,:,0].argmax(axis=1),base[:,:,4].argmax(axis=1)],axis=-1)
negative=teacher[np.arange(len(teacher))[:,None],rival]==0.
t_weight=torch.as_tensor(weight,dtype=torch.float64)
t_n=torch.as_tensor(n,dtype=torch.float64)
t_sd=torch.as_tensor(target_sd,dtype=torch.float64)
t_teacher=torch.as_tensor(teacher,dtype=torch.float64)
t_cdf=torch.as_tensor(teacher_cdf,dtype=torch.float64)
t_constant=torch.as_tensor(cdf_constant,dtype=torch.float64)
t_period=torch.as_tensor(period,dtype=torch.float64)
t_average=torch.as_tensor(average,dtype=torch.float64)
t_reference=torch.as_tensor(reference,dtype=torch.float64)
t_run_weight=torch.as_tensor(train_run/train_run.sum(axis=1)[:,None],dtype=torch.float64)
t_rival=torch.as_tensor(rival,dtype=torch.long)[None].expand(b,-1,-1)
t_negative=torch.as_tensor(negative,dtype=torch.float64)
t_negative_n=t_negative.sum(dim=-1).clamp_min(1.)
t_fixed=torch.zeros((b,len(teacher),len(period)),dtype=torch.float64)
t_feature=torch.as_tensor(raw[:,:,~phase_terms],dtype=torch.float64)[None].expand(b,-1,-1,-1)
t_scale=torch.as_tensor(feature_scale[:,~phase_terms],dtype=torch.float64)
shape=(b,int((~phase_terms).sum()))
crps_strength=0.
rank_strength=0.
mean_strength=0.
logit=lambda parameter:t_fixed+torch.einsum("gnpd,gd->gnp",t_feature,parameter.reshape(shape)/t_scale)
losses=lambda parameter:(
    score:=logit(parameter),
    log_mass:=torch.log_softmax(score,dim=-1),
    individual:=-(log_mass*t_teacher[None]*t_weight[:,:,None]).sum(dim=(-1,-2)),
    cdf:=log_mass.exp().cumsum(dim=-1)[:,:,:-1],
    ordered:=(((cdf-t_cdf[None]).square().sum(dim=-1)*step+t_constant[None])*t_weight).sum(dim=-1)/t_sd,
    positive:=(score*t_teacher[None]).sum(dim=-1),
    ranking:=((torch.nn.functional.softplus(score.gather(-1,t_rival)-positive[:,:,None])*t_negative[None]).sum(dim=-1)/t_negative_n[None]*t_weight).sum(dim=-1),
    prior:=parameter.reshape(shape).square().sum(dim=-1)/(2*t_n),
    group_mean:=(log_mass.exp()*t_period).sum(dim=-1)@t_average.T,
    mean_loss:=.5*((group_mean-t_reference)/t_sd[:,None]).square().mul(t_run_weight).sum(dim=-1),
    torch.stack([individual,crps_strength*ordered,rank_strength*ranking,prior,mean_strength*mean_loss],dim=-1))[-1]
objective=lambda parameter:losses(parameter).sum(dim=-1).mean()
grad_value=torch.func.grad_and_value(objective)
scipy_objective=lambda parameter:(pair:=grad_value(torch.as_tensor(parameter,dtype=torch.float64)),
    (pair[1].detach().item(),pair[0].detach().numpy()))[-1]
started=time.perf_counter()
print("SHARED_NATIVE_TRAINING",args.contexts,"cases",b,"labels",n.tolist(),"threads",CONFIG["threads"],flush=True)
if args.parent_model is None:
    likelihood=minimize(scipy_objective,np.zeros(shape).ravel(),jac=True,method="L-BFGS-B",options=CONFIG["solver"])
    assert likelihood.success
    crps_strength=1.
    initial_fit=minimize(scipy_objective,likelihood.x,jac=True,method="L-BFGS-B",options=CONFIG["solver"])
    assert initial_fit.success
    anchor_coefficient=np.zeros((b,raw.shape[-1]))
    anchor_coefficient[:,~phase_terms]=initial_fit.x.reshape(shape)/feature_scale[:,~phase_terms]
    anchor_score=np.einsum("npd,gd->gnp",raw,anchor_coefficient,optimize=True)
    anchor_probability=torch.softmax(torch.as_tensor(anchor_score),dim=-1).numpy()
    print("SHARED_INITIAL_FITTED",initial_fit.nit,np.max(abs(initial_fit.jac)),time.perf_counter()-started,flush=True)
    t_fixed=torch.as_tensor(anchor_score,dtype=torch.float64)
    t_feature=torch.as_tensor(raw[:,:,phase_terms],dtype=torch.float64)[None].expand(b,-1,-1,-1)
    t_scale=torch.as_tensor(feature_scale[:,phase_terms],dtype=torch.float64)
    shape=(b,int(phase_terms.sum()))
    phase_fit=minimize(scipy_objective,np.zeros(shape).ravel(),jac=True,method="L-BFGS-B",options=CONFIG["solver"])
    assert phase_fit.success
    phase_coefficient=anchor_coefficient.copy()
    phase_coefficient[:,phase_terms]=phase_fit.x.reshape(shape)/feature_scale[:,phase_terms]
    phase_score=np.einsum("npd,gd->gnp",raw,phase_coefficient,optimize=True)
    t_fixed=torch.as_tensor(phase_score,dtype=torch.float64)
    t_feature=torch.as_tensor(raw,dtype=torch.float64)[None].expand(b,-1,-1,-1)
    t_scale=torch.as_tensor(feature_scale,dtype=torch.float64)
    shape=(b,raw.shape[-1])
    rank_strength=1.
    core_fit=minimize(scipy_objective,np.zeros(shape).ravel(),jac=True,method="L-BFGS-B",options=CONFIG["solver"])
    assert core_fit.success
    core_coefficient=phase_coefficient+core_fit.x.reshape(shape)/feature_scale
    core_score=np.einsum("npd,gd->gnp",raw,core_coefficient,optimize=True)
    print("SHARED_CORE_FITTED",core_fit.nit,np.max(abs(core_fit.jac)),time.perf_counter()-started,flush=True)
    
    parent_status="refitted_shared_native"
    variant_suffix=""
    method_prefix="shared_native"
else:
    assert args.contexts=="full"
    parent=np.load(args.parent_model,allow_pickle=False)
    parent_case=np.flatnonzero(parent["fit_length"]==args.parent_head).item()
    assert b==1
    anchor_coefficient=parent["anchor_coefficient"][parent_case][None]
    core_coefficient=parent["coefficient"][parent_case][None]
    anchor_score=np.einsum("npd,gd->gnp",raw,anchor_coefficient,optimize=True)
    core_score=np.einsum("npd,gd->gnp",raw,core_coefficient,optimize=True)
    anchor_probability=torch.softmax(torch.as_tensor(anchor_score),dim=-1).numpy()
    parent_status="frozen_native_time_encoder"
    variant_suffix="_transfer"
    method_prefix="shared_native_transfer"
    print("FROZEN_NATIVE_ENCODER",str(args.parent_model),"head",args.parent_head,flush=True)
crps_strength=1.
rank_strength=1.

t_raw=torch.as_tensor(raw,dtype=torch.float64)
bg_index=background["background_index"]
t_bg_target=torch.as_tensor(raw[np.arange(len(raw)),bg_index],dtype=torch.float64)
bg_logit=lambda parameter:torch.einsum("npd,gd->gnp",t_raw,parameter.reshape(b,-1)/torch.as_tensor(feature_scale))
bg_objective=lambda parameter:(
    score:=bg_logit(parameter),
    positive:=torch.einsum("nd,gd->gn",t_bg_target,parameter.reshape(b,-1)/torch.as_tensor(feature_scale)),
    (((torch.logsumexp(score,dim=-1)-positive)*t_weight).sum(dim=-1)+parameter.reshape(b,-1).square().sum(dim=-1)/(2*t_n)).mean())[-1]
bg_grad=torch.func.grad_and_value(bg_objective)
bg_scipy=lambda parameter:(pair:=bg_grad(torch.as_tensor(parameter,dtype=torch.float64)),
    (pair[1].detach().item(),pair[0].detach().numpy()))[-1]
bg_fit=minimize(bg_scipy,np.zeros((b,raw.shape[-1])).ravel(),jac=True,method="L-BFGS-B",options=CONFIG["solver"])
assert bg_fit.success
bg_coefficient=bg_fit.x.reshape(b,-1)/feature_scale
bg_probability=torch.softmax(bg_logit(torch.as_tensor(bg_fit.x)),dim=-1).numpy()
count=data["mask"].sum(axis=1)
design,layout,joint_gram,rhs=role_geometry(data["time"],count,data["mixed"],data["peer"],period)
lengths=np.unique(count)
layout_mask=np.arange(len(data["time"]))[None]<lengths[:,None]
centered=(data["time"][None]-(layout_mask*data["time"][None]).sum(axis=1)[:,None]/lengths[:,None])*layout_mask
trend=np.stack([layout_mask/np.sqrt(lengths[:,None]),centered/np.sqrt((centered**2).sum(axis=1))[:,None]],axis=-1)
training_period=np.where(data["qualified"],data["target"],period.mean())
angle=2*np.pi*data["time"][None]/training_period[:,None]
target_harmonic=np.stack([np.cos(angle),np.sin(angle)],axis=-1)*data["mask"][:,:,None]
target_harmonic-=np.einsum("nti,nij->ntj",trend[layout],np.einsum("nti,ntj->nij",trend[layout],target_harmonic,optimize=True),optimize=True)
training_design=np.concatenate([target_harmonic,design[layout,bg_index]],axis=-1)
t_gram=torch.as_tensor(np.einsum("nti,ntj->nij",training_design,training_design,optimize=True))
t_rhs=torch.as_tensor(np.stack([np.einsum("nti,nt->ni",training_design,data["mixed"],optimize=True),
    np.einsum("nti,nt->ni",training_design,data["peer"],optimize=True)]))
t_energy=torch.as_tensor(np.stack([(data["mixed"]**2).sum(axis=1),(data["peer"]**2).sum(axis=1)]))
t_count=torch.as_tensor(count-2,dtype=torch.float64)
cov_shape=(b,2,3)
cov_objective=lambda parameter:(
    log_h:=parameter.reshape(cov_shape),h:=log_h.exp(),
    amplitude:=torch.repeat_interleave(h[:,:,:2],2,dim=-1),sd:=amplitude.sqrt(),
    precision:=torch.eye(4,dtype=torch.float64)[None,None,None]+t_gram[None,None]*sd[:,:,None,:,None]*sd[:,:,None,None,:]/h[:,:,2,None,None,None],
    scaled_rhs:=t_rhs[None]*sd[:,:,None,:]/h[:,:,2,None,None],
    gain:=(scaled_rhs*torch.linalg.solve(precision,scaled_rhs[:,:,:,:,None])[:,:,:,:,0]).sum(dim=-1),
    nll:=.5*(t_count[None,None]*log_h[:,:,2,None]+torch.linalg.slogdet(precision)[1]+t_energy[None]/h[:,:,2,None]-gain),
    ((nll*t_weight[:,None]).sum(dim=-1)+log_h.square().sum(dim=-1)/(2*t_n[:,None])).mean())[-1]
cov_grad=torch.func.grad_and_value(cov_objective)
cov_scipy=lambda parameter:(pair:=cov_grad(torch.as_tensor(parameter,dtype=torch.float64)),
    (pair[1].detach().item(),pair[0].detach().numpy()))[-1]
cov_fit=minimize(cov_scipy,np.zeros(cov_shape).ravel(),jac=True,method="L-BFGS-B",options=CONFIG["solver"])
assert cov_fit.success
covariance=np.exp(cov_fit.x.reshape(cov_shape))
evidence,diagnostic=role_assignment_evidence(joint_gram,rhs,layout,covariance,bg_probability,len(period)//2,len(period)//2)
assert np.max(abs(diagnostic[:,1:]))<1e-9
role_features=role_score_features(evidence,base)
role_mean=np.einsum("gn,gnd->gd",weight,role_features.mean(axis=2),optimize=True)
role_scale=np.sqrt(np.maximum(np.einsum("gn,gnd->gd",weight,(role_features**2).mean(axis=2),optimize=True)-role_mean**2,0.))
role_scale=np.where(role_scale>0,role_scale,1.)
t_feature=torch.as_tensor(role_features,dtype=torch.float64)
t_scale=torch.as_tensor(role_scale,dtype=torch.float64)
t_fixed=torch.as_tensor(core_score,dtype=torch.float64)
shape=(b,role_features.shape[-1])
proper=minimize(scipy_objective,np.zeros(shape).ravel(),jac=True,method="L-BFGS-B",options=CONFIG["solver"])
assert proper.success
proper_coefficient=proper.x.reshape(shape)/role_scale
proper_probability=torch.softmax(logit(torch.as_tensor(proper.x)),dim=-1).numpy()
mean_strength=1.
weak=minimize(scipy_objective,proper.x,jac=True,method="L-BFGS-B",options=CONFIG["solver"])
assert weak.success
weak_coefficient=weak.x.reshape(shape)/role_scale
weak_probability=torch.softmax(logit(torch.as_tensor(weak.x)),dim=-1).numpy()
print("SHARED_ROLE_FITTED",proper.nit,weak.nit,np.max(abs(weak.jac)),time.perf_counter()-started,flush=True)

density_evidence=np.stack([torch.log_softmax(torch.as_tensor(core_score),dim=-1).numpy(),
    np.log(np.maximum(bg_probability,np.finfo(float).tiny))],axis=-1)
density_features=role_score_features(density_evidence,base)
density_mean=np.einsum("gn,gnd->gd",weight,density_features.mean(axis=2),optimize=True)
density_scale=np.sqrt(np.maximum(np.einsum("gn,gnd->gd",weight,(density_features**2).mean(axis=2),optimize=True)-density_mean**2,0.))
density_scale=np.where(density_scale>0,density_scale,1.)
t_feature=torch.as_tensor(density_features,dtype=torch.float64)
t_scale=torch.as_tensor(density_scale,dtype=torch.float64)
shape=(b,density_features.shape[-1])
mean_strength=0.
density_proper=minimize(scipy_objective,np.zeros(shape).ravel(),jac=True,method="L-BFGS-B",options=CONFIG["solver"])
assert density_proper.success
density_proper_coefficient=density_proper.x.reshape(shape)/density_scale
density_proper_probability=torch.softmax(logit(torch.as_tensor(density_proper.x)),dim=-1).numpy()
mean_strength=1.
density_weak=minimize(scipy_objective,density_proper.x,jac=True,method="L-BFGS-B",options=CONFIG["solver"])
assert density_weak.success
density_weak_coefficient=density_weak.x.reshape(shape)/density_scale
density_weak_probability=torch.softmax(logit(torch.as_tensor(density_weak.x)),dim=-1).numpy()
probability=np.stack([proper_probability,weak_probability,density_proper_probability,density_weak_probability])
source_mean=(probability@period)@average.T
calibrated,calibration_x,calibration_y,calibration_n=fit_monotone_mean(source_mean,reference,train_run)
anchor=anchor_probability@period
anchor_deviation=anchor-(anchor@average.T)[:,data["run_index"]]
target_deviation=data["target"]-reference[data["run_index"]]
within_variance=np.einsum("gn,n->g",weight,target_deviation**2,optimize=True)
within_coefficient=np.maximum(0.,((anchor_deviation*target_deviation[None]*train).sum(axis=1)+within_variance)/
    ((anchor_deviation**2*train).sum(axis=1)+within_variance))
np.savez_compressed(output/("lemna_task_period_shared_native_"+args.contexts+context_suffix+variant_suffix+"_fit.npz"),
    methods=method_prefix+np.asarray(["_role_proper","_role_mean","_density_proper","_density_mean"]),
    trace_id=data["trace_id"],period=period,coefficient=core_coefficient,anchor_coefficient=anchor_coefficient,
    feature_scale=feature_scale,within_coefficient=within_coefficient,
    role_covariance=covariance,background_coefficient=bg_coefficient,posterior=probability,
    role_coefficient=np.stack([proper_coefficient,weak_coefficient,density_proper_coefficient,density_weak_coefficient]),
    role_feature_scale=np.stack([role_scale,role_scale,density_scale,density_scale]),
    source_mean=source_mean,calibrated_mean=calibrated,calibration_prediction=calibration_x,
    calibration_mean=calibration_y,calibration_count=calibration_n,training_mask=train,training_run=train_run,
    query=query,outer_case=case_index,query_length=args.query_length,inner_owner=owner,inner_query=inner_query,contexts=args.contexts,
    parent_strategy=parent_status,parent_model=str(args.parent_model),
    background_optimizer_success=bg_fit.success,covariance_optimizer_success=cov_fit.success,
    source_optimizer_success=weak.success,density_optimizer_success=density_weak.success,
    source_gradient=np.max(abs(weak.jac)),density_gradient=np.max(abs(density_weak.jac)),
    elapsed_s=time.perf_counter()-started,
    definition="same_777_native_observations_same_outer_queries_and_original_228_inner_queries_shared_physical_time_scores_no_length_label_strata_no_new_seed")
print("SHARED_NATIVE_SAVED",args.contexts,"a",within_coefficient.tolist(),"source_mean_range",
    [source_mean.min(),source_mean.max()],"seconds",time.perf_counter()-started,flush=True)
