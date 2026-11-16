"""Deterministic mixtures constructed from recorded Shan-2020 source waveforms."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1],"archive":"data/external/independent_mixed_reporter_search/Shan_Neuron_2020_full.zip","output":"data/external/shan2020_semisynthetic"},"threads":os.cpu_count(),"plot":{},"layouts":[104,145],"mixing":[.5,2.],"max_interpolation_steps":3}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
from io import BytesIO
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update(CONFIG["plot"])
root=CONFIG["paths"]["root"]
output=root/CONFIG["paths"]["output"]
output.mkdir(parents=True,exist_ok=True)
archive=zipfile.ZipFile(root/CONFIG["paths"]["archive"])
names=sorted(filter(lambda s:s.endswith('_signal.csv'),archive.namelist()))
frames=list(map(lambda s:pd.read_csv(BytesIO(archive.read(s)),encoding="utf-8-sig"),names))
relative=list(map(lambda d:d.set_index(d.TimesH-d.TimesH.min()).drop(columns=["TimesH","Frame"]),frames))
regular=list(map(lambda d:d.reindex(np.arange(int(round(d.index.max()*4))+1)/4.).interpolate(method="index",limit=CONFIG["max_interpolation_steps"],limit_area="inside"),relative))
hourly=list(map(lambda d:d.reindex(np.arange(max(CONFIG["layouts"]))),regular))
sources=pd.concat(map(lambda a:a[1].rename_axis("x").reset_index().melt(id_vars="x",var_name="cell",value_name="source_y").assign(archive_member=a[0]),zip(names,hourly)),ignore_index=True)
sources["genotype"]=sources.archive_member.str.split('/').str[2]
sources["slice"]=sources.archive_member.str.split('/').str[3]
sources["phase"]=sources.archive_member.str.extract(r'_(Pre|TTX|Wash)_')[0]
sources["channel"]=sources.archive_member.str.rsplit('_',n=2).str[-2]
sources["source_id"]=sources.slice+"__"+sources.phase+"__"+sources.channel+"__cell"+sources.cell
sources.to_parquet(output/"recorded_sources_hourly.parquet",index=False)
wide=sources.pivot(index="source_id",columns="x",values="source_y")
meta=sources[["source_id","archive_member","genotype","slice","phase","channel","cell"]].drop_duplicates().set_index("source_id").loc[wide.index]
complete=np.stack([np.isfinite(wide.to_numpy()[:,:104]).all(axis=-1),np.isfinite(wide.to_numpy()[:,:145]).all(axis=-1)],axis=-1)
inventory=meta.reset_index().loc[meta.reset_index().index.repeat(2)].reset_index(drop=True)
inventory["length"]=np.tile(CONFIG["layouts"],len(meta))
inventory["complete_window"]=complete.ravel()
inventory["availability"]=np.where(inventory.complete_window,"available","insufficient_contiguous_observations")
inventory.to_csv(output/"source_availability.csv",index=False,encoding="utf-8",lineterminator="\n")
layout,record=np.nonzero(complete.T)
length=np.asarray(CONFIG["layouts"])[layout]
time=wide.columns.to_numpy(dtype=float)
mask=time[None]<length[:,None]
y=np.where(mask,wide.to_numpy()[record],0.)
centered=(time[None]-(mask*time[None]).sum(axis=-1)[:,None]/length[:,None])*mask
trend=np.stack([mask/np.sqrt(length[:,None]),centered/np.sqrt((centered**2).sum(axis=-1))[:,None]],axis=-1)
residual=y-np.einsum("nti,ni->nt",trend,np.einsum("nti,nt->ni",trend,y,optimize=True),optimize=True)
rms=np.sqrt((residual**2).sum(axis=-1)/length)
assert (rms>0.).all()
unit=residual/rms[:,None]
records=meta.iloc[record].reset_index()
records["length"]=length
records["group_id"]=records.slice+"__"+records.phase+"__l"+records.length.astype(str)
records["run_source_id"]=records.group_id+"__"+records.channel
pool=pd.DataFrame(unit,index=records.run_source_id).groupby(level=0).mean()
pool_n=records.groupby("run_source_id").size()
other_key=records.group_id+"__"+records.channel.replace({"Green":"Red","Red":"Green"})
other=pool.reindex(other_key).to_numpy()
other/=np.sqrt((other**2).sum(axis=-1)/length)[:,None]
valid=np.isfinite(other).all(axis=-1)&records.run_source_id.map(pool_n).gt(1).to_numpy()
records["mixture_available"]=valid
records["target_pool_n"]=records.run_source_id.map(pool_n).to_numpy()
records["background_pool_n"]=other_key.map(pool_n).to_numpy()
records["background_pool_id"]=other_key
records["source_rms"]=rms
records.to_csv(output/"mixture_source_map.csv",index=False,encoding="utf-8",lineterminator="\n")
records=records.loc[valid].reset_index(drop=True)
unit=unit[valid]
other=other[valid]
length=length[valid]
alpha,source_index=np.indices((len(CONFIG["mixing"]),len(records))).reshape(2,-1)
a=np.asarray(CONFIG["mixing"])[alpha]
observed=unit[source_index]+a[:,None]*other[source_index]
trace=np.char.add(np.char.add(np.char.add(np.char.add(records.source_id.to_numpy(dtype=str)[source_index],"__l"),length[source_index].astype(str)),"__a"),alpha.astype(str))
run=np.char.add(np.char.add(records.run_source_id.to_numpy(dtype=str)[source_index],"__a"),alpha.astype(str))
mixture=records.iloc[source_index].reset_index(drop=True).assign(trace_id=trace,run=run,condition=np.where(alpha==0,"control","treated"),mixing_alpha=a,alpha_index=alpha)
mixture.to_csv(output/"mixture_records.csv",index=False,encoding="utf-8",lineterminator="\n")
np.savez_compressed(output/"latent_components.npz",source_id=records.source_id.to_numpy(dtype=str),length=length,time=time,target=unit,background=other,observation_source_index=source_index,observation_alpha=a,trace_id=trace)
row,col=np.nonzero(time[None]<mixture.length.to_numpy()[:,None])
inputs=mixture.iloc[row][["trace_id","run","condition"]].reset_index(drop=True).assign(x=time[col],signal_y=observed[row,col])
inputs.to_parquet(output/"inputs.parquet",index=False)
print("SHAN_SOURCES",len(names),meta.slice.nunique(),len(meta),"ELIGIBLE_LAYOUT_SOURCES",len(records),"MIXED_RECORDS",len(mixture),flush=True)
print(inventory.groupby(["phase","length","availability"]).size().to_string(),flush=True)
print("MIXTURE_EQUATION_MAX",float(np.max(abs(observed-unit[source_index]-a[:,None]*other[source_index]))),flush=True)
