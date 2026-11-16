"""Prepare all public EDF recordings using abdominal inputs and EDF+ references."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1],"data":"data/external/adfecgdb"},
    "threads":os.cpu_count(),"plot":{},"window_seconds":10.,"sample_rate":1000.,
    "period_bounds_s":(.25,1.5),"minimum_reference_intervals":9}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
import zipfile,re,json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update(CONFIG["plot"])
root=CONFIG["paths"]["root"]
folder=root/CONFIG["paths"]["data"]
archive=zipfile.ZipFile(folder/"adfecgdb-1.0.0.zip")
assert archive.testzip() is None
prefix=archive.namelist()[0].split("/")[0]+"/"
manifest=pd.read_csv(folder/"SHA256SUMS.txt",sep=r"\s+",names=["sha256","file"],encoding="utf-8")
verified=np.asarray(list(map(lambda name:hashlib.sha256(archive.read(prefix+name)).hexdigest(),manifest.file)))
assert np.array_equal(verified,manifest.sha256.to_numpy())
records=archive.read(prefix+"RECORDS").decode("ascii").splitlines()
assert len(records)==5


def read_record(name):
    content=archive.read(prefix+name)
    header=int(content[184:192])
    channels=int(content[252:256])
    n_records=int(content[236:244])
    duration=float(content[244:252])
    labels=np.frombuffer(content[256:256+16*channels],dtype="S16").astype(str)
    labels=np.char.strip(labels)
    samples=np.frombuffer(content[256+216*channels:256+224*channels],dtype="S8").astype(int)
    assert np.array_equal(labels[:5],np.asarray(["Direct_1","Abdomen_1","Abdomen_2","Abdomen_3","Abdomen_4"]))
    assert labels[-1]=="EDF Annotations" and np.all(samples[:5]/duration==CONFIG["sample_rate"])
    digital=np.frombuffer(content[header:],dtype="<i2").reshape(n_records,samples.sum())
    physical_min=np.frombuffer(content[256+104*channels:256+112*channels],dtype="S8").astype(float)[:5]
    physical_max=np.frombuffer(content[256+112*channels:256+120*channels],dtype="S8").astype(float)[:5]
    digital_min=np.frombuffer(content[256+120*channels:256+128*channels],dtype="S8").astype(float)[:5]
    digital_max=np.frombuffer(content[256+128*channels:256+136*channels],dtype="S8").astype(float)[:5]
    signal=digital[:,:samples[:5].sum()].reshape(n_records,5,samples[0]).transpose(1,0,2).reshape(5,-1).astype(float)
    signal=(signal-digital_min[:,None])*(physical_max-physical_min)[:,None]/(digital_max-digital_min)[:,None]+physical_min[:,None]
    annotation=digital[:,-samples[-1]:].astype("<i2").tobytes()
    peak_time=np.asarray(re.findall(rb"([+-]\d+(?:\.\d*)?)\x14QRS\x14\x00",annotation),dtype=float)
    assert len(peak_time)>300 and np.all(np.diff(peak_time)>0)
    n_windows=int(n_records*duration/CONFIG["window_seconds"])
    window_size=int(CONFIG["window_seconds"]*CONFIG["sample_rate"])
    assert n_windows*window_size==signal.shape[1]
    wave=signal[1:].reshape(4,n_windows,window_size).transpose(1,0,2).reshape(4*n_windows,window_size)
    window=np.repeat(np.arange(n_windows),4)
    channel=np.tile(np.arange(1,5),n_windows)
    record=Path(name).stem
    run=np.char.add(record+"_w",np.char.mod("%03d",window))
    trace=np.char.add(np.char.add(run,"_c"),channel.astype(str))
    rr=np.diff(peak_time)
    midpoint=(peak_time[:-1]+peak_time[1:])/2
    rr_window=np.floor(midpoint/CONFIG["window_seconds"]).astype(int)
    counts=np.bincount(rr_window,minlength=n_windows)
    reference=np.divide(np.bincount(rr_window,weights=rr,minlength=n_windows),counts,
        out=np.full(n_windows,np.nan),where=counts>0)
    outside=(rr<CONFIG["period_bounds_s"][0])|(rr>CONFIG["period_bounds_s"][1])
    outside_count=np.bincount(rr_window,weights=outside,minlength=n_windows).astype(int)
    median=pd.DataFrame({"window":rr_window,"rr":rr}).groupby("window").rr.median().reindex(np.arange(n_windows)).to_numpy()
    qualified=(counts>=CONFIG["minimum_reference_intervals"])&(outside_count==0)&np.isfinite(reference)
    reference_qc=np.select([counts<CONFIG["minimum_reference_intervals"],outside_count>0],
        ["insufficient_rr_intervals","rr_outside_period_support"],default="qualified")
    assert len(counts)==n_windows
    truth=pd.DataFrame({"trace_id":trace,"run":run,"subject":record,"window":window,
        "channel":channel,"condition":"unperturbed","reference_period_s":np.repeat(reference,4),
        "reference_intervals":np.repeat(counts,4),"reference_median_rr_s":np.repeat(median,4),
        "reference_outside_intervals":np.repeat(outside_count,4),
        "reference_qc":np.repeat(reference_qc,4),"qualified":np.repeat(qualified,4)})
    frame=pd.DataFrame({"trace_id":np.repeat(trace,window_size),"run":np.repeat(run,window_size),
        "condition":"unperturbed","x":np.tile(np.arange(window_size)/CONFIG["sample_rate"],len(trace)),
        "signal_y":wave.reshape(-1)})
    foreground=pd.DataFrame({"run":np.repeat(np.char.add(record+"_w",np.char.mod("%03d",np.arange(n_windows))),window_size),
        "x":np.tile(np.arange(window_size)/CONFIG["sample_rate"],n_windows),"foreground_y":signal[0]})
    info={"subject":record,"seconds":n_records*duration,"windows":n_windows,
        "input_channels":4,"reference_channel":"Direct_1","reference_qrs":len(peak_time),
        "sample_rate_hz":CONFIG["sample_rate"],"source_file":name}
    return frame,truth,info,foreground


prepared=list(map(read_record,records))
frame=pd.concat(list(map(lambda item:item[0],prepared)),ignore_index=True)
truth=pd.concat(list(map(lambda item:item[1],prepared)),ignore_index=True)
info=pd.DataFrame(list(map(lambda item:item[2],prepared)))
foreground=pd.concat(list(map(lambda item:item[3],prepared)),ignore_index=True)
frame.to_parquet(folder/"inputs.parquet",index=False)
foreground.to_parquet(folder/"foreground_reference.parquet",index=False)
truth.to_csv(folder/"references.csv",index=False,encoding="utf-8")
info.to_csv(folder/"record_metadata.csv",index=False,encoding="utf-8")
meta=truth.sort_values("trace_id").reset_index(drop=True)
runs=np.unique(meta.run.to_numpy(dtype=str))
membership=(runs[:,None]==meta.run.to_numpy()[None]).astype(float)
subjects=np.unique(meta.subject)
subject_code=pd.Index(subjects).get_indexer(meta.subject)
run_subject=pd.Index(subjects).get_indexer(meta.drop_duplicates("run").set_index("run").loc[runs,"subject"])
query=np.arange(len(subjects))[:,None]==subject_code[None]
training=np.arange(len(subjects))[:,None]!=run_subject[None]
np.savez_compressed(folder/"supervised_context.npz",trace_id=meta.trace_id.to_numpy(dtype=str),
    target=meta.reference_period_s.to_numpy(),qualified=meta.qualified.to_numpy(),run_name=runs,
    outer_query=query,outer_train_run=training,outer_protocol=np.full(len(subjects),"leave_subject"),
    outer_condition=np.full(len(subjects),"unperturbed"),outer_control_run=np.full(len(subjects),-1),
    subject=meta.subject.to_numpy(dtype=str),window=meta.window.to_numpy(),unit=np.asarray("s"))
provenance={"source":"https://physionet.org/content/adfecgdb/1.0.0/","doi":"10.13026/C2RP4B",
    "license":"Open Data Commons Attribution License v1.0","official_files_verified":len(manifest),
    "subjects":len(subjects),"input_traces":len(meta),"input_rows":len(frame),
    "qualified_traces":int(meta.qualified.sum()),"qualified_windows":int(meta.loc[meta.qualified,"run"].nunique()),
    "predictor_channels":["Abdomen_1","Abdomen_2","Abdomen_3","Abdomen_4"],
    "reference":"EDF+ QRS annotations and Direct_1; excluded from prediction inputs",
    "window_seconds":CONFIG["window_seconds"],"reference_definition":"mean RR interval whose midpoint lies inside the observation window",
    "reference_period_support_s":CONFIG["period_bounds_s"],"minimum_reference_intervals":CONFIG["minimum_reference_intervals"],
    "reference_qc":"Every RR must lie inside the predefined period support; raw mean RR and all inputs are retained",
    "foreground_training_reference":"foreground_reference.parquet: one Direct_1 waveform per window; excluded from prediction inputs",
    "split_unit":"whole subject; all windows and channels from the held-out subject remain excluded"}
(folder/"source.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
print("ADFECG_PREPARED",json.dumps(provenance),flush=True)
