"""Wang-2026 coupled-clock equations with explicit mixed observations and realized-phase truth."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1],"output":"data/external/wang2026_mechanism"},"threads":os.cpu_count(),"plot":{},"seed":12345,"layouts":[104,145],"replicates":3,"cells":16,"burnin_h":500.,"intrinsic_h":[22.7,24.3],"regimes":[["uncoupled",0.,0.,0.],["weak_gated",.02,.2,.10],["strong_gated",.10,1.,.20]],"families":["mixing_only","VIP_blockade"],"noise_ar":.7,"noise_rms":.25,"phase_sd":[.6,.15],"log_amplitude_sd":.3,"solver":{"rtol":1e-9,"atol":1e-10}}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import lfilter
plt.rcParams.update(CONFIG["plot"])
root=CONFIG["paths"]["root"]
output=root/CONFIG["paths"]["output"]
output.mkdir(parents=True,exist_ok=True)
regime=pd.DataFrame(CONFIG["regimes"],columns=["regime","K_BtoA","K_AtoB_ratio","K_VIP"])
regime.to_csv(output/"mechanism_parameters.csv",index=False,encoding="utf-8",lineterminator="\n")
rng=np.random.default_rng(CONFIG["seed"])
shape=(2,len(regime),CONFIG["replicates"],CONFIG["cells"],2)
initial=rng.normal(size=(1,1,CONFIG["replicates"],CONFIG["cells"],2))*np.asarray(CONFIG["phase_sd"])+np.asarray([0.,.1])
initial=np.broadcast_to(initial,shape).copy()
K_BtoA=regime.K_BtoA.to_numpy()[None,:,None,None]
K_AtoB=(regime.K_BtoA*regime.K_AtoB_ratio).to_numpy()[None,:,None,None]
K_VIP=np.stack([regime.K_VIP.to_numpy(),np.zeros(len(regime))])[:,:,None,None]
omega=2*np.pi/np.asarray(CONFIG["intrinsic_h"])
derivative=lambda t,y:(
    theta:=y.reshape(shape),
    a:=theta[...,0],b:=theta[...,1],
    np.stack([omega[0]+K_BtoA*np.sin(b-a)-K_VIP*np.maximum(np.sin(b),0.)*np.maximum(np.sin(a-np.pi/2.),0.),
        omega[1]+K_AtoB*np.sin(a-b)],axis=-1).ravel())[-1]
time=np.arange(max(CONFIG["layouts"]),dtype=float)
sample_time=CONFIG["burnin_h"]+time
solution=solve_ivp(derivative,(0.,sample_time[-1]),initial.ravel(),t_eval=sample_time,method="DOP853",**CONFIG["solver"])
assert solution.success and solution.y.shape[-1]==len(time)
theta=solution.y.reshape(shape+(len(time),))
amplitude=np.exp(CONFIG["log_amplitude_sd"]*rng.normal(size=(CONFIG["replicates"],CONFIG["cells"],2)))
raw=amplitude[None,None,:,:,:,None]*np.cos(theta)
noise=lfilter([1.],[1.,-CONFIG["noise_ar"]],rng.normal(size=(CONFIG["replicates"],CONFIG["cells"],len(time))),axis=-1)
noise/=np.sqrt(np.mean(noise**2,axis=-1))[:,:,None]
family,scene,layout,replicate,arm,cell=np.indices((len(CONFIG["families"]),len(regime),len(CONFIG["layouts"]),CONFIG["replicates"],2,CONFIG["cells"])).reshape(6,-1)
length=np.asarray(CONFIG["layouts"])[layout]
dynamics=np.where(family==0,0,arm)
mask=time[None]<length[:,None]
source=raw[dynamics,scene,replicate,cell]
source*=mask[:,None]
centered=(time[None]-(mask*time[None]).sum(axis=-1)[:,None]/length[:,None])*mask
trend=np.stack([mask/np.sqrt(length[:,None]),centered/np.sqrt((centered**2).sum(axis=-1))[:,None]],axis=-1)
source-=np.einsum("nti,nis->nst",trend,np.einsum("nti,nst->nis",trend,source,optimize=True),optimize=True)
source/=np.sqrt((source**2).sum(axis=-1)/length[:,None])[:,:,None]
alpha=np.where(family==0,np.where(arm==0,.5,2.),1.)
observed=source[:,0]+alpha[:,None]*source[:,1]+CONFIG["noise_rms"]*noise[replicate,cell]*mask
phase=theta[dynamics,scene,replicate,cell]
end_phase=np.take_along_axis(phase,np.broadcast_to((length-1)[:,None,None],(len(length),2,1)),axis=-1)[:,:,0]
period=2*np.pi*(length-1)[:,None]/(end_phase-phase[:,:,0])
assert np.isfinite(period).all() and (period>0.).all()
family_name=np.asarray(CONFIG["families"])[family]
pair=("Wang__"+pd.Series(family_name)+"__s"+pd.Series(scene+1).astype(str)+"__l"+pd.Series(length).astype(str)+"__r"+pd.Series(replicate+1).astype(str)).to_numpy(dtype=str)
condition=np.where(arm==0,"control","treated")
run=np.char.add(np.char.add(pair,"__"),condition)
trace=np.char.add(np.char.add(run,"__c"),(cell+1).astype(str))
truth=pd.DataFrame({"trace_id":trace,"run":run,"condition":condition,"pair_id":pair,"family":family_name,"scenario":regime.regime.to_numpy()[scene],"length":length,"replicate":replicate+1,"cell":cell+1,"mixing_alpha":alpha,"true_period_h":period[:,0],"true_background_h":period[:,1],"intrinsic_target_h":CONFIG["intrinsic_h"][0],"intrinsic_background_h":CONFIG["intrinsic_h"][1]})
truth.to_csv(output/"truth_cells.csv",index=False,encoding="utf-8",lineterminator="\n")
np.savez_compressed(output/"latent_components.npz",trace_id=trace,time=time,count=length,target=source[:,0],background=source[:,1],target_phase=phase[:,0],background_phase=phase[:,1],observation_noise=CONFIG["noise_rms"]*noise[replicate,cell]*mask,observation_alpha=alpha,observed=observed)
row,col=np.nonzero(mask)
inputs=truth.iloc[row][["trace_id","run","condition"]].reset_index(drop=True).assign(x=time[col],signal_y=observed[row,col])
inputs.to_parquet(output/"inputs.parquet",index=False)
pair_truth=truth.groupby(["pair_id","family","scenario","length","condition"]).true_period_h.mean().unstack("condition")
pair_truth["true_effect_h"]=pair_truth.treated-pair_truth.control
pair_truth.to_csv(output/"truth_effects.csv",encoding="utf-8",lineterminator="\n")
assert pair_truth.loc[pair_truth.index.get_level_values("family")=="mixing_only","true_effect_h"].abs().max()==0.
operator_error=np.max(abs(observed-source[:,0]-alpha[:,None]*source[:,1]-CONFIG["noise_rms"]*noise[replicate,cell]*mask))
assert operator_error<1e-12
print("WANG_MECHANISM_READY",len(truth),"records",truth.run.nunique(),"runs","ODE_evaluations",solution.nfev,"operator_error",operator_error,flush=True)
print(pair_truth.groupby(["family","scenario","length"]).true_effect_h.mean().to_string(),flush=True)
