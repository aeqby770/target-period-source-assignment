"""Fit the background law using only collective-reference features; freeze the target encoder."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1]},"threads":os.cpu_count(),"plot":{},
    "device":"cuda","solver":{"maxiter":900,"ftol":1e-11,"gtol":1e-6,"maxls":30}}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
import sys,ast,inspect,time,argparse
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import softmax
root=CONFIG["paths"]["root"];sys.path.insert(0,str(root/"software"))
from source_assignment_core import role_geometry,role_assignment_evidence,role_score_features,fit_monotone_mean,interpolate_mean
from period_support import project_periods_with_mean
torch.set_num_threads(CONFIG["threads"])
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--matched-views",action="store_true");args=parser.parse_args()
clean=root/"data/sota_dev/recomputed_clean";parent=clean/"original_scale_peer_frozen_parameters"
tag="matched_peer_background" if args.matched_views else "peer_only_background";out=clean/tag;out.mkdir(exist_ok=True)
data=np.load(clean/"lemna_task_period_supervised_context.npz",allow_pickle=False)
f=np.load(clean/"original_scale_peer_source/features.npz",allow_pickle=False)
full=dict(np.load(parent/"frozen_mean_model.npz",allow_pickle=False));replay=dict(np.load(parent/"frozen_mean_replay_model.npz",allow_pickle=False))
fit=np.load(parent/"lemna_task_period_original_scale_peer_frozen_parameters_fit.npz",allow_pickle=False)
old=np.load(clean/"lemna_task_period_actual_source_assignment_frequency_geometry_fit.npz",allow_pickle=False)
labels=np.load(clean/"lemna_task_period_background_density_fit.npz",allow_pickle=False)
np.testing.assert_array_equal(f["trace_id"],data["trace_id"])
np.testing.assert_array_equal(labels["trace_id"],data["trace_id"])
length=np.r_[full["fit_length"],replay["fit_length"]];active=np.flatnonzero(length==145)
train=old["training_mask"][active];train_run=old["training_run"][active]
assert not np.any(old["training_mask"][2:]&data["outer_query"])
n=train.sum(axis=-1);weight=train/n[:,None];raw=f["raw"];base=f["base_features"];period=full["period"]
# Existing schema: collective H1, H2-extra, period/half ACF and spectral coherence.
reference_terms=np.isin(np.arange(13),[4,5,6,7,8]);left,right=np.triu_indices(13)
feature_mask=np.r_[reference_terms,reference_terms[left]&reference_terms[right]]
native_x=raw[:,:,feature_mask]
if args.matched_views:
    training_views=np.load(out/"features.npz",allow_pickle=False)
    np.testing.assert_array_equal(training_views["trace_id"],data["trace_id"])
    np.testing.assert_array_equal(training_views["feature_mask"],feature_mask)
    x=training_views["reference_features"];target_index=training_views["background_index"];view_weight=training_views["view_weight"]
else:
    x=native_x[None];target_index=labels["background_index"][None];view_weight=np.ones(1)
view_mask=np.broadcast_to(train[:,None],(len(active),len(x),len(raw))).copy()
if args.matched_views:
    view_mask[:,1:]=train_run[:,data["run_index"]][:,None]
assert not np.any(view_mask[active>=2]&data["outer_query"][active[active>=2]-2,None,:])
view_train_weight=view_mask/view_mask.sum(axis=-1)[:,:,None]
moment=weight@native_x.mean(axis=1)
scale=np.sqrt(np.maximum(weight@(native_x**2).mean(axis=1)-moment**2,0.));scale=np.where(scale>0.,scale,1.)
all_bg=np.concatenate([full["background_coefficient"],replay["background_coefficient"]])
initial=all_bg[active][:,feature_mask]*scale
device=CONFIG["device"];dtype=torch.float64;tensor=lambda a:torch.as_tensor(a,dtype=dtype,device=device)
tx=tensor(x);tscale=tensor(scale);tw=tensor(view_train_weight);tn=tensor(n);tview=tensor(view_weight)
target=tx[torch.arange(len(x),device=device)[:,None],torch.arange(len(raw),device=device)[None],torch.as_tensor(target_index,device=device)]
shape=initial.shape;transform=torch.eye(shape[-1],dtype=dtype,device=device)[None].expand(len(active),-1,-1)
loss=lambda parameter:(
    theta:=torch.einsum("gij,gj->gi",transform,parameter.reshape(shape)),
    score:=torch.einsum("vnpd,gd->gvnp",tx,theta/tscale),
    positive:=torch.einsum("vnd,gd->gvn",target,theta/tscale),
    ((((torch.logsumexp(score,dim=-1)-positive)*tw).sum(dim=-1)*tview).sum(dim=-1)+theta.square().sum(dim=-1)/(2*tn)).mean())[-1]
fisher_case=lambda s,current,w,cell_n:(
    z:=tx/s,p:=torch.softmax(torch.einsum("vnpd,d->vnp",z,current),dim=-1),
    weighted:=z*torch.sqrt(p*w[:,:,None]*tview[:,None,None])[:,:,:,None],
    mean_z:=torch.einsum("vnp,vnpd->vnd",p,z),
    torch.einsum("vnpi,vnpj->ij",weighted,weighted)-torch.einsum("vn,vni,vnj->ij",w*tview[:,None],mean_z,mean_z)
        +torch.eye(shape[-1],dtype=dtype,device=device)/cell_n)[-1]
fisher=torch.vmap(fisher_case,in_dims=(0,0,0,0),chunk_size=1)(tscale,tensor(initial),tw,tn)
chol=torch.linalg.cholesky((fisher+fisher.transpose(-1,-2))/2.)
transform=torch.linalg.solve_triangular(chol.transpose(-1,-2),
    torch.eye(shape[-1],dtype=dtype,device=device)[None].expand(len(active),-1,-1),upper=True)
initial=torch.einsum("gij,gj->gi",chol.transpose(-1,-2),tensor(initial)).cpu().numpy().ravel()
grad_value=torch.func.grad_and_value(loss)
objective=lambda parameter:(result:=grad_value(tensor(parameter)),(result[1].item(),result[0].cpu().numpy()))[-1]
probe=np.zeros_like(initial);probe[3]=1e-5
check=abs((objective(initial+probe)[0]-objective(initial-probe)[0])/(2e-5)-objective(initial)[1][3])
assert check<1e-6,check
print("PEER_ONLY_BG_START",shape,"gradient",check,"loss",objective(initial)[0],flush=True)
optimized=minimize(objective,initial,jac=True,method="L-BFGS-B",options=CONFIG["solver"]|{"maxcor":shape[-1]})
coefficient=np.zeros_like(all_bg[active])
coefficient[:,feature_mask]=(torch.einsum("gij,gj->gi",transform,tensor(optimized.x.reshape(shape))).cpu().numpy())/scale
np.savez_compressed(out/"background_fit.npz",coefficient=coefficient,feature_mask=feature_mask,case_index=active,
    training_mask=train,training_run=train_run,optimizer_success=optimized.success,optimizer_message=np.asarray(optimized.message))
assert optimized.success,(optimized.message,optimized.nit,np.max(abs(optimized.jac)))
all_bg[active]=coefficient
background=softmax(np.einsum("npd,gd->gnp",raw,all_bg,optimize=False),axis=-1)
design,layout,joint,rhs=role_geometry(f["time"],f["count"],f["mixed"],f["peer"],period)
covariance=np.concatenate([full["role_covariance"],replay["role_covariance"]])
gpu=inspect.getsource(role_assignment_evidence).replace("dtype=torch.float64)","dtype=torch.float64,device='cuda')").replace("dtype=torch.long)","dtype=torch.long,device='cuda')")
gpu=gpu.replace("torch.arange(len(layout))","torch.arange(len(layout),device='cuda')")
gpu=gpu.replace("return features.numpy(), diagnostic.numpy()","return features.cpu().numpy(), diagnostic.cpu().numpy()")
scope={"np":np,"torch":torch};exec(compile(gpu,"<same_role_likelihood_gpu>","exec"),scope)
phase,_=scope["role_assignment_evidence"](joint,rhs,layout,covariance,background,len(period)//2,len(period)//2)
role=role_score_features(phase,base)
role_coefficient=np.concatenate([full["role_coefficient"],replay["role_coefficient"]])
core_score=fit["core_score"].copy();role_score=np.einsum("gnpd,gd->gnp",role,role_coefficient,optimize=False)
probability=softmax(core_score+role_score,axis=-1)
membership=data["run_membership"];average=membership/membership.sum(axis=-1)[:,None]
source_mean=(probability@period)@average.T
evaluator=(root/"experiments/evaluate_correlated_clock_order.py").read_text(encoding="utf-8")
composition=evaluator[evaluator.index("old_x="):evaluator.index('np.savez_compressed(out/"real_features.npz"')]
composition=composition.replace('cal_x=np.stack([old_x,new_x]);',
    'new_x[length==104]=old_x[length==104];new_y[length==104]=old_y[length==104];new_n[length==104]=old_n[length==104]\ncal_x=np.stack([old_x,new_x]);')
exec(compile(composition,"<same_native_evaluation_peer_only_background>","exec"))
exec(compile(evaluator[evaluator.index("tree=ast.parse"):],"<same_native_report>","exec"))
short=length[2:]==104
np.testing.assert_allclose(estimate[short,:,0],fit["estimate"][short,:,0],rtol=0.,atol=1e-10)
np.savez_compressed(out/"frozen_mean_model.npz",**(full|{"method":np.asarray("task_"+tag+"_frozen_mean"),
    "background_coefficient":all_bg[:2]}))
np.savez_compressed(out/"frozen_mean_replay_model.npz",**(replay|{"method":np.asarray("task_"+tag+"_frozen_mean"),
    "background_coefficient":all_bg[2:]}))
print("PEER_ONLY_BG_DONE",optimized.nit,"fixed_target_encoder_and_covariance",flush=True)
