"""Subject-held-out cardiac periods with source and same-information baselines."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1],"data":"data/external/adfecgdb","output":"results/0.3.1"},
    "threads":os.cpu_count(),"plot":{},"seed":12345}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
import sys,json,argparse
import numpy as np
import pandas as pd
from scipy.special import softmax,logsumexp
from scipy.fft import rfft,irfft,next_fast_len
from scipy.optimize import minimize
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
plt.rcParams.update(CONFIG["plot"])
root=CONFIG["paths"]["root"]
sys.path.insert(0,str(root/"software"))
from source_assignment_unified import prediction_matrix,observations,detrend,harmonic_profile,MODEL_KIND,SOURCE_OPERATOR,CORE_POLICY,POOL_PARAMETER_KEYS,predict_general_source_roles,predict_latent_representation_density,foreground_reference_log_prior
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model",type=Path)
parser.add_argument("--feature-cache",type=Path)
parser.add_argument("--output-dir",type=Path)
parser.add_argument("--device",choices=["cpu","cuda"],default="cuda",help="Device for conditional source-density inference")
parser.add_argument("--spatial-reference-probe",action="store_true",help="Evaluate all source-reference folds using the complete synchronous channel space")
parser.add_argument("--event-phase-reference-probe",action="store_true",help="Train-only event shape and No-only phase nuisance subtraction with interval-density readout")
parser.add_argument("--event-landmark-reference-probe",action="store_true",help="Use training-subject cardiologist QRS landmarks for the same event-phase probe")
parser.add_argument("--supervised-event-reference-probe",action="store_true",help="Train one No-only FG event classifier on all training timestamps and reuse source renewal likelihood")
parser.add_argument("--ablation-suite",action="store_true",help="Refit the fixed adopted shared-source configuration and its five one-factor ablations")
args=parser.parse_args()
folder=args.output_dir or root/CONFIG["paths"]["output"]
folder.mkdir(parents=True,exist_ok=True)
model_path=args.model or folder/"adfecg_model.npz"
data_folder=root/CONFIG["paths"]["data"]
frame=pd.read_parquet(data_folder/"inputs.parquet")
if not (args.ablation_suite or args.supervised_event_reference_probe or args.event_phase_reference_probe or args.event_landmark_reference_probe or args.spatial_reference_probe):
    selected_metadata=np.load(model_path,allow_pickle=False)
    if str(selected_metadata.get("measurement_kind",""))=="shared_landmark_renewal":
        args.supervised_event_reference_probe=True
        args.model=model_path
shared_no_only_started=None
shared_no_only_value=None
shared_no_only_model=None
shared_no_only_data=None
if args.supervised_event_reference_probe and args.model is not None:
    import time as prediction_clock
    shared_no_only_started=prediction_clock.perf_counter()
    from source_assignment_distribution import shared_landmark_observations,load_shared_landmark_renewal_model,predict_shared_landmark_renewal
    shared_no_only_model=load_shared_landmark_renewal_model(args.model)
    shared_no_only_data=shared_landmark_observations(frame,source_grouping=str(shared_no_only_model["source_grouping"]),time_unit="s")
    # All six predictions are completed before evaluation references are opened.
    shared_no_only_value=predict_shared_landmark_renewal(shared_no_only_data,shared_no_only_model,include_event_score=False)
    print("ADFECG_NO_ONLY_CORE_PREDICTION",shared_no_only_value["case_index"].tolist(),shared_no_only_value["estimate"].shape,flush=True)
reference=pd.read_csv(data_folder/"references.csv",encoding="utf-8").set_index("trace_id")
context=np.load(data_folder/"supervised_context.npz",allow_pickle=False)


def write_shared_landmark_ablation_tables(entries,data,source_reference,folder,*,complete_count=9):
    """The same source/window collection for fitted suites and cached-model replay."""
    bysource=source_reference;source_names=np.asarray(data.get("source_names",data["runs"]),dtype=str)
    ref=bysource.reference_period_s.to_numpy(dtype=float)
    qualified=bysource.qualified.to_numpy(dtype=bool);subject=bysource.subject.to_numpy(dtype=str)
    full=entries[0][1];cases=full["case_index"];training=full["training_source"];units=np.unique(subject)
    def make_rows(item):
        method,model,retrained,path=item
        separate_detector=Path(str(model["detector_joblib_path"])).resolve()!=Path(str(full["detector_joblib_path"])).resolve()
        detector_policy="primary_fit" if method==entries[0][0] else "separate_fit" if separate_detector else "shared_primary_detector"
        prediction=np.asarray(model["source_period"])
        case,source=np.indices(prediction.shape);query=np.vstack([np.zeros((1,len(ref)),dtype=bool),~training[1:]])
        rows=pd.DataFrame({"method":method,"case_index":case.ravel(),"source_id":np.tile(source_names,len(cases)),
            "subject":np.tile(subject,len(cases)),"window":np.tile(bysource.window.to_numpy(dtype=int),len(cases)),
            "qualified":np.tile(qualified,len(cases)),"query":query.ravel(),
            "estimate_period_s":prediction.ravel(),"reference_period_s":np.tile(ref,len(cases)),"estimate_available":np.isfinite(prediction).ravel(),
            "training_N_source":np.repeat(model["train_reference_source_count"],len(ref)),"training_G":np.repeat(model["train_reference_unit_count"],len(ref)),
            "N_source":int(qualified.sum()),"G":len(units),"ablation_field_scope":"shared-source","detector_refitted_for_configuration":separate_detector,
            "detector_fitted_in_this_call":retrained,"detector_training_policy":detector_policy,
            "information_budget":"No_query; training_period_waveform_landmarks","uses_training_event_supervision":True,
            "feature_options":str(model.get("feature_options_json","{}")),"measurement_options":str(model.get("measurement_options_json","{}")),
            "model_path":str(path),"detector_path":str(model["detector_joblib_path"])})
        rows["absolute_error_ms"]=pd.Series(np.where(rows.estimate_available,1000*abs(rows.estimate_period_s-rows.reference_period_s),np.inf),index=rows.index).where(rows.qualified)
        for label in ["interval_mean","interval_median","interval_ridge"]:
            rows[label+"_s"]=np.asarray(model[label]).ravel()
            rows[label+"_error_ms"]=pd.Series(np.where(np.isfinite(rows[label+"_s"]),1000*abs(rows[label+"_s"]-rows.reference_period_s),np.inf),index=rows.index).where(rows.qualified)
        return rows
    cells=pd.concat(list(map(make_rows,entries)),ignore_index=True)
    subjects=cells.loc[cells["query"]].groupby(["method","subject"],sort=False).agg(n_input_sources=("source_id","size"),n_reference_sources=("qualified","sum"),
        n_estimated_sources=("estimate_available","sum"),period_mae_ms=("absolute_error_ms","mean"),detector_refitted_for_configuration=("detector_refitted_for_configuration","first"),
        detector_fitted_in_this_call=("detector_fitted_in_this_call","first"),detector_training_policy=("detector_training_policy","first"),
        ablation_field_scope=("ablation_field_scope","first"),information_budget=("information_budget","first"),
        feature_options=("feature_options","first"),measurement_options=("measurement_options","first"),model_path=("model_path","first"),detector_path=("detector_path","first")).reset_index()
    subjects["n_input"]=subjects.n_input_sources*int(full["measurement_channel_count"]);subjects["n_reference"]=subjects.n_reference_sources*int(full["measurement_channel_count"])
    subjects["N_source"]=int(qualified.sum());subjects["G"]=len(units)
    summary=subjects.groupby("method",sort=False).agg(subjects=("subject","size"),n_input=("n_input","sum"),n_reference=("n_reference","sum"),
        n_input_sources=("n_input_sources","sum"),n_reference_sources=("n_reference_sources","sum"),n_estimated_sources=("n_estimated_sources","sum"),period_mae_ms=("period_mae_ms","mean"),
        detector_refitted_for_configuration=("detector_refitted_for_configuration","first"),
        detector_fitted_in_this_call=("detector_fitted_in_this_call","first"),detector_training_policy=("detector_training_policy","first"),ablation_field_scope=("ablation_field_scope","first"),information_budget=("information_budget","first"),
        feature_options=("feature_options","first"),measurement_options=("measurement_options","first"),model_path=("model_path","first"),detector_path=("detector_path","first")).reset_index()
    adjacent=cells.loc[cells["query"]].sort_values(["method","subject","window"]).copy()
    grouped=adjacent.groupby(["method","subject"])
    adjacent["predicted_change_s"]=grouped.estimate_period_s.diff();adjacent["reference_change_s"]=grouped.reference_period_s.diff()
    adjacent["change_pair_input"]=grouped.window.diff().eq(1)
    adjacent["change_reference_usable"]=adjacent.qualified&grouped.qualified.shift(fill_value=False)&adjacent.change_pair_input
    adjacent["change_estimate_available"]=np.isfinite(adjacent.predicted_change_s)&adjacent.change_pair_input
    adjacent["change_failed"]=adjacent.change_pair_input&~adjacent.change_estimate_available
    adjacent["change_reference_unusable"]=adjacent.change_pair_input&~adjacent.change_reference_usable
    adjacent["change_error_ms"]=pd.Series(np.where(adjacent.change_estimate_available,1000*abs(adjacent.predicted_change_s-adjacent.reference_change_s),np.inf),index=adjacent.index).where(adjacent.change_reference_usable)
    change_subject=adjacent.groupby(["method","subject"],sort=False).agg(change_mae_ms=("change_error_ms","mean"),n_possible_change_pairs=("change_pair_input","sum"),n_change_pairs=("change_reference_usable","sum"),n_failed_change_pairs=("change_failed","sum"),n_unusable_reference_change_pairs=("change_reference_unusable","sum")).reset_index()
    subjects=subjects.merge(change_subject,on=["method","subject"],validate="one_to_one")
    change_macro=change_subject.groupby("method",sort=False).agg(change_mae_ms=("change_mae_ms","mean"),n_possible_change_pairs=("n_possible_change_pairs","sum"),n_change_pairs=("n_change_pairs","sum"),n_failed_change_pairs=("n_failed_change_pairs","sum"),n_unusable_reference_change_pairs=("n_unusable_reference_change_pairs","sum")).reset_index()
    summary=summary.merge(change_macro,on="method",validate="one_to_one")
    summary["N_source"]=int(qualified.sum());summary["G"]=len(units);summary["suite_complete"]=len(entries)==complete_count
    cells.to_csv(folder/"adfecg_ablation_cells.csv",index=False,encoding="utf-8");subjects.to_csv(folder/"adfecg_ablation_subjects.csv",index=False,encoding="utf-8");summary.to_csv(folder/"adfecg_ablation_summary.csv",index=False,encoding="utf-8")
    return summary


def write_shared_landmark_standard_tables(value,model,data,reference,context,folder):
    """Original standard schemas plus source-level counts and supervision budgets."""
    order=pd.Index(context["trace_id"]).get_indexer(data["metadata"].index)
    outer_query=context["outer_query"][:,order]
    case,cell=np.nonzero(outer_query)
    positions=pd.Index(value["case_index"]).get_indexer(case+1)
    owner=value["source_index"];cohort=value["run_index"]
    truth=reference.loc[data["metadata"].index]
    source_truth=truth.groupby("run",sort=True).first().loc[value["source_id"]]
    nsource=int(source_truth.qualified.sum());units=int(source_truth.subject.nunique())
    success=np.broadcast_to(np.asarray(model["success"],dtype=bool),(len(model["case_index"]),))
    primary=pd.DataFrame({"method":"task_source_assignment_unified","case":case,
        "trace_id":data["metadata"].index.to_numpy(dtype=str)[cell],
        "estimate_period_s":value["estimate"][positions,cell],
        "source_period_s":value["source_period"][positions,owner[cell]],
        "original_period_s":np.nan,"finite_cohort_mean_s":value["group_mean"][positions,cohort[cell]],
        "population_location_s":np.nan,"deviation_s":value["deviation"][positions,cell],
        "fit_status":np.where(success[positions],"converged","not_converged"),"applicable":True,
        "parameter_policy":"shared_source_HGB100_leaf15_fullbin_renewal_beta",
        "posterior_kind":"shared_source_period_posterior"})
    primary=primary.merge(reference.reset_index(),on="trace_id",validate="many_to_one")
    def control(item):
        method,key=item
        result=primary.copy()
        result["method"]=method;result["estimate_period_s"]=value[key][positions,owner[cell]]
        result[["source_period_s","original_period_s","finite_cohort_mean_s","population_location_s","deviation_s"]]=np.nan
        result["posterior_kind"]="point_control"
        result["parameter_policy"]="same_No_event_detector_untrimmed_interval_readout" if key!="interval_ridge" else "same_No_event_detector_standardized_ridge_lambda_1_over_Nsource"
        result["fit_status"]="not_fitted" if key!="interval_ridge" else "converged"
        return result
    controls=pd.concat(list(map(control,[("shared_detector_interval_mean","interval_mean"),("shared_detector_interval_median","interval_median"),("shared_detector_interval_ridge","interval_ridge")])),ignore_index=True)
    old=pd.read_csv(root/CONFIG["paths"]["output"]/"adfecg_conditional_baseline_cells.csv",encoding="utf-8")
    assert set(old.trace_id)==set(primary.trace_id)
    old["baseline_original_method"]=old.method
    old["method"]=old.method.replace({"source_roles":"conditional_density_baseline"})
    old["posterior_kind"]="saved_baseline_readout"
    predictions=pd.concat([primary,old,controls],ignore_index=True,sort=False)
    predictions["source_id"]=predictions.run
    predictions["model_case_index"]=predictions["case"].astype(int)+1
    source_train=np.asarray(model["train_reference_source_count"])
    unit_train=np.asarray(model["train_reference_unit_count"])
    model_position=pd.Index(model["case_index"]).get_indexer(predictions.model_case_index)
    predictions["training_N_source"]=source_train[model_position]
    predictions["training_G"]=unit_train[model_position]
    predictions["N_source"]=nsource;predictions["G"]=units
    same_event=predictions.method.isin(["task_source_assignment_unified","shared_detector_interval_mean","shared_detector_interval_median","shared_detector_interval_ridge"])
    no_reference=predictions.method.eq("dominant_H1")
    predictions["uses_training_event_supervision"]=same_event
    predictions["supervision"]=np.where(same_event,"training_No_timestamps+Direct_1_shape+cardiologist_QRS+meanRR",
        np.where(no_reference,"none","training_meanRR_and_saved_reference_pipeline"))
    predictions["information_budget"]=np.where(same_event,"No_query; training_period_waveform_landmarks",
        np.where(no_reference,"No_query_only","No_query; training_period_reference_no_landmarks"))
    predictions.loc[predictions.method.eq("conditional_density_baseline"),"information_budget"]="No_query; legacy_period_and_waveform_view_training_no_landmarks"
    predictions.loc[no_reference,["training_N_source","training_G"]]=0
    predictions["fit_status"]=pd.Categorical(predictions.fit_status,categories=["not_fitted","converged","not_converged"],ordered=True)
    predictions["estimate_available"]=np.isfinite(predictions.estimate_period_s)
    predictions["fit_not_converged"]=predictions.fit_status.eq("not_converged")
    predictions["raw_mean_rr_error_ms"]=1000*abs(predictions.estimate_period_s-predictions.reference_period_s)
    predictions["median_rr_error_ms"]=1000*abs(predictions.estimate_period_s-predictions.reference_median_rr_s)
    predictions["absolute_error_ms"]=pd.Series(np.where(predictions.estimate_available,predictions.raw_mean_rr_error_ms,np.inf),index=predictions.index).where(predictions.qualified&predictions.applicable)
    predictions["relative_error"]=pd.Series(np.where(predictions.estimate_available,abs(predictions.estimate_period_s/predictions.reference_period_s-1.),np.inf),index=predictions.index).where(predictions.qualified&predictions.applicable)
    predictions.to_csv(folder/"adfecg_cells.csv",index=False,encoding="utf-8")
    # Window/source errors are formed before subject macro-averaging for every method.
    windows=predictions.groupby(["method","subject","run","window"],sort=False).agg(
        estimate_period_s=("estimate_period_s","mean"),reference_period_s=("reference_period_s","first"),
        qualified=("qualified","all"),reference_qc=("reference_qc","first"),fit_status=("fit_status","max"),applicable=("applicable","all"),
        estimate_available=("estimate_available","all"),n_input_measurements=("trace_id","size"),n_estimated_measurements=("estimate_available","sum"),
        supervision=("supervision","first"),information_budget=("information_budget","first"),uses_training_event_supervision=("uses_training_event_supervision","first")).reset_index()
    windows.loc[~windows.estimate_available,"estimate_period_s"]=np.nan
    windows["absolute_error_ms"]=pd.Series(np.where(windows.estimate_available,1000*abs(windows.estimate_period_s-windows.reference_period_s),np.inf),index=windows.index).where(windows.qualified&windows.applicable)
    windows["relative_error"]=pd.Series(np.where(windows.estimate_available,abs(windows.estimate_period_s/windows.reference_period_s-1.),np.inf),index=windows.index).where(windows.qualified&windows.applicable)
    lead_subjects=predictions.groupby(["method","subject"],sort=False).agg(n_input=("trace_id","size"),n_reference=("qualified","sum"),n_estimated=("estimate_available","sum"),
        n_fit_not_converged=("fit_not_converged","sum"),lead_period_mae_ms=("absolute_error_ms","mean"),lead_relative_mae=("relative_error","mean")).reset_index()
    subjects=windows.groupby(["method","subject"],sort=False).agg(n_input_sources=("run","size"),n_reference_sources=("qualified","sum"),n_estimated_sources=("estimate_available","sum"),
        fit_status=("fit_status","max"),applicable=("applicable","all"),period_mae_ms=("absolute_error_ms","mean"),relative_mae=("relative_error","mean"),
        supervision=("supervision","first"),information_budget=("information_budget","first"),uses_training_event_supervision=("uses_training_event_supervision","first")).reset_index().merge(lead_subjects,on=["method","subject"],validate="one_to_one")
    subjects["N_source"]=nsource;subjects["G"]=units
    subjects.to_csv(folder/"adfecg_subjects.csv",index=False,encoding="utf-8")
    summary=subjects.groupby("method",sort=False).agg(subjects=("subject","size"),n_input=("n_input","sum"),n_reference=("n_reference","sum"),n_estimated=("n_estimated","sum"),
        fit_status=("fit_status","max"),applicable=("applicable","all"),n_fit_not_converged=("n_fit_not_converged","sum"),period_mae_ms=("period_mae_ms","mean"),relative_mae=("relative_mae","mean"),
        n_input_sources=("n_input_sources","sum"),n_reference_sources=("n_reference_sources","sum"),n_estimated_sources=("n_estimated_sources","sum"),
        lead_period_mae_ms=("lead_period_mae_ms","mean"),lead_relative_mae=("lead_relative_mae","mean"),
        supervision=("supervision","first"),information_budget=("information_budget","first"),uses_training_event_supervision=("uses_training_event_supervision","first")).reset_index()
    summary["N_source"]=nsource;summary["G"]=units
    summary.to_csv(folder/"adfecg_summary.csv",index=False,encoding="utf-8")
    win=windows.sort_values(["method","subject","window"]).copy()
    win["estimated_change_s"]=win.groupby(["method","subject"]).estimate_period_s.diff()
    win["reference_change_s"]=win.groupby(["method","subject"]).reference_period_s.diff()
    win["change_pair_input"]=win.groupby(["method","subject"]).window.diff().eq(1)
    win["change_reference_usable"]=win.qualified&win.groupby(["method","subject"]).qualified.shift(fill_value=False)&win.change_pair_input
    win["change_estimate_available"]=np.isfinite(win.estimated_change_s)&win.change_pair_input
    win["absolute_change_error_ms"]=pd.Series(np.where(win.change_estimate_available,1000*abs(win.estimated_change_s-win.reference_change_s),np.inf),index=win.index).where(win.change_reference_usable&win.applicable)
    win["change_reference_unusable"]=win.change_pair_input&~win.change_reference_usable
    win["change_estimate_failed"]=win.change_pair_input&~win.change_estimate_available
    win["N_source"]=nsource;win["G"]=units
    win.to_csv(folder/"adfecg_window_changes.csv",index=False,encoding="utf-8")
    changes=win.groupby(["method","subject"],sort=False).agg(n_input_windows=("window","size"),n_reference_windows=("qualified","sum"),n_change_pairs=("change_reference_usable","sum"),
        n_possible_change_pairs=("change_pair_input","sum"),n_estimated_change_pairs=("change_estimate_available","sum"),n_failed_change_pairs=("change_estimate_failed","sum"),n_unusable_reference_change_pairs=("change_reference_unusable","sum"),
        fit_status=("fit_status","max"),applicable=("applicable","all"),change_mae_ms=("absolute_change_error_ms","mean"),
        supervision=("supervision","first"),information_budget=("information_budget","first"),uses_training_event_supervision=("uses_training_event_supervision","first")).reset_index()
    change_summary=changes.groupby("method",sort=False).agg(subjects=("subject","size"),n_input_windows=("n_input_windows","sum"),n_reference_windows=("n_reference_windows","sum"),n_change_pairs=("n_change_pairs","sum"),
        n_possible_change_pairs=("n_possible_change_pairs","sum"),n_estimated_change_pairs=("n_estimated_change_pairs","sum"),n_failed_change_pairs=("n_failed_change_pairs","sum"),n_unusable_reference_change_pairs=("n_unusable_reference_change_pairs","sum"),
        fit_status=("fit_status","max"),applicable=("applicable","all"),change_mae_ms=("change_mae_ms","mean"),
        supervision=("supervision","first"),information_budget=("information_budget","first"),uses_training_event_supervision=("uses_training_event_supervision","first")).reset_index()
    change_summary["N_source"]=nsource;change_summary["G"]=units
    change_summary.to_csv(folder/"adfecg_change_summary.csv",index=False,encoding="utf-8")
    primary_summary=summary.loc[summary.method.eq("task_source_assignment_unified")].iloc[0]
    assert primary_summary.n_input==600 and primary_summary.n_reference==592 and primary_summary.n_input_sources==150 and primary_summary.n_reference_sources==148
    np.testing.assert_allclose(primary.estimate_period_s,primary.source_period_s,rtol=0.,atol=1e-12)
    np.testing.assert_allclose(primary.estimate_period_s,primary.finite_cohort_mean_s,rtol=0.,atol=1e-12)
    assert primary.deviation_s.eq(0.).all()
    print("ADFECG_STANDARD_SHARED_SOURCE",summary[["method","n_input","n_reference","n_input_sources","n_reference_sources","N_source","G","period_mae_ms","lead_period_mae_ms"]].to_string(index=False),flush=True)
    print("ADFECG_STANDARD_PRIMARY_CHANGE",change_summary.loc[change_summary.method.eq("task_source_assignment_unified")].to_dict("records"),flush=True)
    return summary


if args.ablation_suite:
    from source_assignment_distribution import shared_landmark_observations,load_shared_landmark_renewal_model,save_shared_landmark_renewal_model,fit_shared_landmark_renewal,predict_shared_landmark_renewal,shared_landmark_renewal_log_likelihood,calibrate_shared_landmark_temperature
    import zipfile,re,time as clock,joblib
    started=clock.perf_counter();main_path=args.model or folder/"adfecg_model.npz"
    full=load_shared_landmark_renewal_model(main_path)
    data=shared_landmark_observations(frame,source_grouping=str(full["source_grouping"]),time_unit="s")
    truth=reference.loc[data["metadata"].index];bysource=truth.drop_duplicates("run").set_index("run").loc[data["source_names"]]
    qualified=bysource.qualified.to_numpy(dtype=bool);ref=bysource.reference_period_s.to_numpy(dtype=float);subject=bysource.subject.to_numpy(dtype=str)
    training=full["training_source"];cases=full["case_index"];period=full["period"]
    assert len(qualified)==150 and qualified.sum()==148 and len(np.unique(subject))==5
    foreground=pd.read_parquet(data_folder/"foreground_reference.parquet").pivot(index="run",columns="x",values="foreground_y").loc[data["source_names"]].to_numpy(dtype=float)
    archive=zipfile.ZipFile(data_folder/"adfecgdb-1.0.0.zip");prefix=archive.namelist()[0].split("/")[0]+"/"
    recording=pd.read_csv(data_folder/"record_metadata.csv",encoding="utf-8").set_index("subject")
    duration=float(json.loads((data_folder/"source.json").read_text(encoding="utf-8"))["window_seconds"])
    def read_events(name):
        content=archive.read(prefix+str(recording.loc[name,"source_file"]))
        header=int(content[184:192]);channels=int(content[252:256]);records=int(content[236:244])
        samples=np.frombuffer(content[256+216*channels:256+224*channels],dtype="S8").astype(int)
        digital=np.frombuffer(content[header:],dtype="<i2").reshape(records,samples.sum())
        annotation=digital[:,-samples[-1]:].astype("<i2").tobytes()
        return np.asarray(re.findall(rb"([+-]\d+(?:\.\d*)?)\x14QRS\x14\x00",annotation),dtype=float)
    units=np.unique(subject);annotations=dict(zip(units,map(read_events,units)))
    def source_events(index):
        events=annotations[subject[index]];window=int(bysource.window.iloc[index]);start=window*duration
        landmarks=events[(events>=start)&(events<start+duration)]-start
        midpoint=(events[:-1]+events[1:])/2.;rr=np.diff(events)[np.floor(midpoint/duration).astype(int)==window]
        if qualified[index]:
            np.testing.assert_allclose(rr.mean(),ref[index],rtol=0.,atol=1e-12)
        return landmarks,rr
    events=list(map(source_events,np.arange(len(subject))))
    accumulated=[];models=[]
    def record(method,model,retrained,path):
        accumulated.append((method,model,retrained,path));models.append((method,str(path)))
        summary=write_shared_landmark_ablation_tables(accumulated,data,bysource,folder)
        print("SHARED_ABLATION_COMPLETE",method,float(summary.loc[summary.method.eq(method),"period_mae_ms"].item()),"elapsed",clock.perf_counter()-started,flush=True)
        return None
    # Adopted simplification: one fixed sensor-coordinate model, then one-factor removals.
    base_options={"cross_sensor_whitening":False,"temporal_neighborhood":True,"derivative_curvature_energy":True}
    print("SHARED_FINAL_DEFAULT_COLD_START",base_options,flush=True)
    full=fit_shared_landmark_renewal(data,ref,qualified,training,subject,foreground,list(map(lambda row:row[0],events)),period,
        reference_intervals=list(map(lambda row:row[1],events)),time_unit="s",
        detector_path=folder/"adfecg_shared_landmark_sensor_rms_detector.joblib",feature_options=base_options)
    full.update(reference=ref,qualified=qualified,subject=subject,reference_source_count=full["train_reference_source_count"],
        effective_units=full["train_reference_unit_count"],trace_id=data["metadata"].index.to_numpy(dtype=str),run_index=data["run_index"],run_names=data["runs"])
    save_shared_landmark_renewal_model(full,main_path)
    save_shared_landmark_renewal_model(full,folder/"adfecg_shared_landmark_renewal_model.npz")
    current=predict_shared_landmark_renewal(data,full,include_event_score=False)
    default_difference=float(np.max(abs(current["source_probability"]-full["source_probability"])))
    assert default_difference<1e-7
    np.savez_compressed(folder/"adfecg_ablation_full_detector_cache.npz",case_index=cases,source_id=current["source_id"],
        source_event_time=current["source_event_time"],source_event_count=current["source_event_count"],source_log_likelihood=current["source_log_likelihood"],
        default_probability_max_abs=np.asarray(default_difference),ablation_field_scope=np.asarray("shared-source"))
    record("task_source_assignment_unified",full,True,main_path)
    zca_path=folder/"adfecg_ablation_add_spatial_whitening_model.npz";zca=load_shared_landmark_renewal_model(zca_path)
    zca_replay=predict_shared_landmark_renewal(data,zca,case_indices=[0]);assert np.max(abs(zca_replay["source_probability"][0]-zca["source_probability"][0]))<1e-7
    record("add_spatial_whitening",zca,False,zca_path)
    feature_cases=[("no_temporal_neighborhood",base_options|{"temporal_neighborhood":False}),
        ("no_derivative_curvature_energy",base_options|{"derivative_curvature_energy":False})]
    def cold_feature(item):
        name,options=item;path=folder/("adfecg_ablation_"+name+"_model.npz")
        print("SHARED_ABLATION_COLD_START",name,options,flush=True)
        model=fit_shared_landmark_renewal(data,ref,qualified,training,subject,foreground,list(map(lambda row:row[0],events)),period,
            reference_intervals=list(map(lambda row:row[1],events)),time_unit="s",detector_path=folder/("adfecg_ablation_"+name+"_detector.joblib"),feature_options=options)
        model.update(ablation_id=np.asarray(name),ablation_field_scope=np.asarray("shared-source"),reference=ref,qualified=qualified,subject=subject)
        save_shared_landmark_renewal_model(model,path)
        loaded=load_shared_landmark_renewal_model(path);replay=predict_shared_landmark_renewal(data,loaded,case_indices=[0])
        error=float(np.max(abs(replay["source_probability"][0]-model["source_probability"][0])))
        assert error<1e-7
        model["validation_model_replay_error"]=np.asarray(error);save_shared_landmark_renewal_model(model,path)
        record(name,model,True,path)
        return None
    list(map(cold_feature,feature_cases))
    def measurement_case(name):
        model=dict(full)
        if name=="no_missed_clutter":
            def one_case(case):
                def one_source(source):
                    count=int(current["source_event_count"][case,source]);intervals=np.diff(current["source_event_time"][case,source,:count])
                    return shared_landmark_renewal_log_likelihood(intervals,period,full["renewal_RR_variance"][case],full["renewal_timing_variance"][case],
                        full["renewal_detector_precision"][case],full["renewal_detector_recall"][case],duration,missed_clutter=False)
                return np.stack(list(map(one_source,np.arange(len(ref)))))
            likelihood=np.stack(list(map(one_case,np.arange(len(cases)))));prior=full["foreground_log_prior"]
            options={"missed_clutter":False}
        else:
            likelihood=current["source_log_likelihood"];prior=np.full_like(full["foreground_log_prior"],-np.log(len(period)))
            options={"foreground_prior":"uniform"}
        calibrated=calibrate_shared_landmark_temperature(likelihood,period,ref,qualified,training,subject,prior)
        model.update(calibrated);model.update(measurement_options_json=np.asarray(json.dumps(options,sort_keys=True)),
            ablation_id=np.asarray(name),ablation_field_scope=np.asarray("shared-source"))
        path=folder/("adfecg_ablation_"+name+"_model.npz");save_shared_landmark_renewal_model(model,path)
        record(name,model,False,path)
        return None
    list(map(measurement_case,["no_missed_clutter","uniform_FG_prior"]))
    def same_information_control(item):
        method,key=item;control=dict(full);control["source_period"]=full[key]
        record(method,control,False,main_path)
        return None
    list(map(same_information_control,[("shared_detector_interval_mean","interval_mean"),("shared_detector_interval_median","interval_median"),("shared_detector_interval_ridge","interval_ridge")]))
    print("SHARED_ABLATION_SUITE_FINISHED",len(accumulated),"models","default_probability_difference",default_difference,"elapsed_seconds",clock.perf_counter()-started,flush=True)
    raise SystemExit(0)

if args.supervised_event_reference_probe:
    from source_assignment_distribution import fit_shared_landmark_renewal,predict_shared_landmark_renewal,load_shared_landmark_renewal_model,save_shared_landmark_renewal_model
    import zipfile,re,time as clock
    started=shared_no_only_started if shared_no_only_started is not None else clock.perf_counter()
    data=observations(frame);data["time_unit"]="s"
    truth=reference.loc[data["metadata"].index]
    qualified=(data["membership"]@truth.qualified.to_numpy(dtype=float))>0.
    ref=data["average"]@truth.reference_period_s.to_numpy()
    bysource=truth.drop_duplicates("run").set_index("run").loc[data["runs"]]
    subject=bysource.subject.to_numpy(dtype=str)
    all_train=np.r_[np.ones((1,len(data["runs"])),dtype=bool),context["outer_train_run"][:,pd.Index(context["run_name"]).get_indexer(data["runs"])]]
    model_path=args.model or folder/"adfecg_shared_landmark_renewal_model.npz"
    if args.model is None:
        foreground=pd.read_parquet(data_folder/"foreground_reference.parquet").pivot(index="run",columns="x",values="foreground_y").loc[data["runs"]].to_numpy(dtype=float)
        archive=zipfile.ZipFile(data_folder/"adfecgdb-1.0.0.zip");prefix=archive.namelist()[0].split("/")[0]+"/"
        recording=pd.read_csv(data_folder/"record_metadata.csv",encoding="utf-8").set_index("subject")
        duration=float(json.loads((data_folder/"source.json").read_text(encoding="utf-8"))["window_seconds"])
        def read_events(name):
            content=archive.read(prefix+str(recording.loc[name,"source_file"]))
            header=int(content[184:192]);channels=int(content[252:256]);records=int(content[236:244])
            samples=np.frombuffer(content[256+216*channels:256+224*channels],dtype="S8").astype(int)
            digital=np.frombuffer(content[header:],dtype="<i2").reshape(records,samples.sum())
            annotation=digital[:,-samples[-1]:].astype("<i2").tobytes()
            return np.asarray(re.findall(rb"([+-]\d+(?:\.\d*)?)\x14QRS\x14\x00",annotation),dtype=float)
        names=np.unique(subject);annotations=dict(zip(names,map(read_events,names)))
        def source_events(index):
            current=annotations[subject[index]];window=int(bysource.window.iloc[index]);start=window*duration
            landmarks=current[(current>=start)&(current<start+duration)]-start
            midpoint=(current[:-1]+current[1:])/2.;intervals=np.diff(current)[np.floor(midpoint/duration).astype(int)==window]
            if qualified[index]:
                np.testing.assert_allclose(intervals.mean(),ref[index],rtol=0.,atol=1e-12)
            return landmarks,intervals
        events=list(map(source_events,np.arange(len(subject))))
        period=np.load(folder/"adfecg_spatial_periodic_reference.npz",allow_pickle=False)["period"]
        model=fit_shared_landmark_renewal(data,ref,qualified,all_train,subject,foreground,
            list(map(lambda item:item[0],events)),period,reference_intervals=list(map(lambda item:item[1],events)),time_unit="s",
            detector_path=folder/"adfecg_shared_landmark_sensor_rms_detector.joblib")
        model.update(reference=ref,qualified=qualified,subject=subject,reference_source_count=model["train_reference_source_count"],
            effective_units=model["train_reference_unit_count"],trace_id=data["metadata"].index.to_numpy(dtype=str),run_index=data["run_index"],run_names=data["runs"])
        save_shared_landmark_renewal_model(model,model_path)
    else:
        # This path opens No observations and a fitted model only, never FG/EDF.
        model=shared_no_only_model if shared_no_only_model is not None else load_shared_landmark_renewal_model(model_path)
    value=shared_no_only_value if shared_no_only_value is not None else predict_shared_landmark_renewal(data,model,include_event_score=False)
    case,source=np.indices(value["source_period"].shape)
    selected=value["case_index"];query=np.r_[np.zeros((1,len(subject)),dtype=bool),~all_train[1:]][selected]
    pooled=np.sum((all_train[selected]&qualified[None])*np.where(qualified,ref,0.)[None],axis=1)/(all_train[selected]&qualified[None]).sum(1)
    cells=pd.DataFrame({"case":np.repeat(selected,len(subject)),"run":np.tile(data["runs"],len(selected)),"subject":np.tile(subject,len(selected)),
        "qualified":np.tile(qualified,len(selected)),"query":query.ravel(),"reference_s":np.tile(ref,len(selected)),"estimate_s":value["source_period"].ravel(),
        "pooled_training_mean_s":np.repeat(pooled,len(subject)),"event_count":value["source_event_count"].ravel(),
        "interval_mean_s":value["interval_mean"].ravel(),"interval_median_s":value["interval_median"].ravel(),"interval_ridge_s":value["interval_ridge"].ravel()})
    def error_column(name):
        cells[name+"_error_ms"]=np.where(cells.qualified,1000*abs(cells[name+"_s"]-cells.reference_s),np.nan)
        return None
    list(map(error_column,["estimate","pooled_training_mean","interval_mean","interval_median","interval_ridge"]))
    cells.to_csv(folder/"adfecg_shared_landmark_renewal_cells.csv",index=False,encoding="utf-8")
    query_cells=cells.loc[cells["query"]].copy();query_cells.to_csv(folder/"adfecg_shared_landmark_renewal_query.csv",index=False,encoding="utf-8")
    report=query_cells.groupby("subject").agg(windows=("run","size"),qualified=("qualified","sum"),renewal=("estimate_error_ms","mean"),
        mean=("interval_mean_error_ms","mean"),median=("interval_median_error_ms","mean"),ridge=("interval_ridge_error_ms","mean"),baseline=("pooled_training_mean_error_ms","mean"))
    write_shared_landmark_standard_tables(value,model,data,reference,context,folder)
    if len(model.get("comparison_model_paths",[])):
        from source_assignment_distribution import shared_landmark_renewal_log_likelihood
        from pathlib import Path as ModelPath
        main_entry=dict(model,source_period=value["source_period"],source_probability=value["source_probability"],
            interval_mean=value["interval_mean"],interval_median=value["interval_median"],interval_ridge=value["interval_ridge"])
        entries=[("task_source_assignment_unified",main_entry,False,model_path)]
        for relative in model["comparison_model_paths"]:
            comparison_path=ModelPath(str(relative))
            if not comparison_path.is_absolute():
                comparison_path=(model_path.parent/comparison_path) if str(model.get("comparison_model_path_base","model_directory"))=="model_directory" else root/comparison_path
            other=load_shared_landmark_renewal_model(comparison_path)
            method=str(other["ablation_id"])
            same_detector=ModelPath(str(other["detector_joblib_path"])).resolve()==ModelPath(str(model["detector_joblib_path"])).resolve()
            same_features=str(other.get("feature_options_json","{}"))==str(model.get("feature_options_json","{}"))
            if same_detector and same_features:
                options=json.loads(str(other.get("measurement_options_json","{}")))
                if options.get("missed_clutter",True):
                    likelihood=value["source_log_likelihood"]
                else:
                    def case_likelihood(case):
                        def source_likelihood(source):
                            count=int(value["source_event_count"][case,source]);moments=value["source_event_time"][case,source,:count]
                            duration=float(data["time"][-1]-data["time"][0]+np.median(np.diff(data["time"])))
                            return shared_landmark_renewal_log_likelihood(np.diff(moments),other["period"],other["renewal_RR_variance"][case],
                                other["renewal_timing_variance"][case],other["renewal_detector_precision"][case],other["renewal_detector_recall"][case],duration,missed_clutter=False)
                        return np.stack(list(map(source_likelihood,np.arange(len(subject)))))
                    likelihood=np.stack(list(map(case_likelihood,np.arange(len(other["case_index"])))))
                probability=softmax(other["renewal_temperature"][:,None,None]*likelihood+other["foreground_log_prior"][:,None],axis=-1)
                other["source_period"]=probability@other["period"]
                print("SHARED_COMPARISON_REPLAY",method,"same_fresh_event_stream",flush=True)
            else:
                other_value=predict_shared_landmark_renewal(data,other,include_event_score=False)
                other["source_period"]=other_value["source_period"]
                print("SHARED_COMPARISON_REPLAY",method,"fresh_No_detector",flush=True)
            entries.append((method,other,False,comparison_path))
        for method,key in [("shared_detector_interval_mean","interval_mean"),("shared_detector_interval_median","interval_median"),("shared_detector_interval_ridge","interval_ridge")]:
            point=dict(main_entry);point["source_period"]=value[key];entries.append((method,point,False,model_path))
        closure=write_shared_landmark_ablation_tables(entries,data,bysource,folder)
        print("SHARED_CACHED_COMPARISON_CLOSURE",closure[["method","period_mae_ms","change_mae_ms","suite_complete"]].to_string(index=False),flush=True)
    print("SHARED_LANDMARK_CORE_CLI",report.to_string(),sep="\n",flush=True)
    print("SHARED_LANDMARK_CORE_MACRO",report[["renewal","mean","median","ridge","baseline"]].mean().to_dict(),"elapsed_seconds",clock.perf_counter()-started,flush=True)
    print("SHARED_LANDMARK_POSTERIOR_READOUT",value["posterior"].shape,value["estimate"].shape,"mass_error",float(np.max(abs(value["posterior"].sum(-1)-1.))),
        "cohort_mean_error",float(np.max(abs(value["cohort_probability"]@value["period"]-value["group_mean"]))),flush=True)
    raise SystemExit(0)
if args.event_phase_reference_probe or args.event_landmark_reference_probe or args.supervised_event_reference_probe:
    from scipy.signal import find_peaks,fftconvolve
    from scipy.ndimage import gaussian_filter1d
    from source_assignment_unified import shared_background,period_reference_mass
    import time as clock
    started=clock.perf_counter()
    supervised_detector=bool(args.supervised_event_reference_probe)
    landmark_teacher=bool(args.event_landmark_reference_probe or supervised_detector)
    output_tag="adfecg_supervised_event_reference" if supervised_detector else "adfecg_landmark_phase_reference" if landmark_teacher else "adfecg_event_phase_reference"
    data=observations(frame);truth=reference.loc[data["metadata"].index]
    ns=len(data["runs"]);nt=len(data["time"]);nc=len(truth)//ns
    assert np.all(data["membership"].sum(1)==nc) and np.all(data["count"]==nt)
    period=np.load(folder/"adfecg_spatial_periodic_reference.npz",allow_pickle=False)["period"]
    dt=float(np.median(np.diff(data["time"])));pmin=float(period.min());pmax=float(period.max())
    signal=data["signal"].reshape(ns,nc,nt)
    fg_frame=pd.read_parquet(data_folder/"foreground_reference.parquet")
    fg=fg_frame.pivot(index="run",columns="x",values="foreground_y").loc[data["runs"]].to_numpy(dtype=float)
    combined=np.concatenate([signal.reshape(-1,nt),fg])
    _,periodic=shared_background(dict(data,signal=combined,mask=np.ones_like(combined,dtype=bool),count=np.full(len(combined),nt)),
        pmax,strength=(2.*np.pi)**-4,per_record=True)
    signal=periodic[:ns*nc].reshape(ns,nc,nt);fg=periodic[ns*nc:]
    all_train=np.r_[np.ones((1,ns),dtype=bool),context["outer_train_run"][:,pd.Index(context["run_name"]).get_indexer(data["runs"])]]
    qualified=(data["membership"]@truth.qualified.to_numpy(dtype=float))>0.
    subject=truth.drop_duplicates("run").set_index("run").loc[data["runs"],"subject"].to_numpy(dtype=str)
    unit_code,unit_names=pd.factorize(subject,sort=True);unit=(np.arange(len(unit_names))[:,None]==unit_code[None]).astype(float)
    train=all_train&qualified[None];counts=train.astype(float)@unit.T
    weights=np.divide(train.astype(float),counts[:,unit_code],out=np.zeros_like(train,dtype=float),where=counts[:,unit_code]>0.)
    weights/=weights.sum(1)[:,None]
    ref=data["average"]@truth.reference_period_s.to_numpy()
    minimum_distance=max(1,int(np.ceil(pmin/dt)))
    half=max(1,int(np.floor(pmin/(2.*dt))));offset=np.arange(-half,half+1)
    ncomponents=1 if supervised_detector else nc if landmark_teacher else 2*nc
    max_peaks=int(np.ceil((nt-1)*dt/pmin))+1
    curves=np.empty((len(train),ns,len(period)));background_count=np.zeros((len(train),ns),dtype=int)
    removed=np.zeros((len(train),ns));event_count=np.zeros((len(train),ns,ncomponents),dtype=int)
    interval_count=np.zeros_like(event_count);event_time=np.full((*event_count.shape,max_peaks),np.nan)
    background_time=np.full((len(train),ns,max_peaks),np.nan)
    templates=np.empty((len(train),len(offset)));shape_sigma=np.empty(len(train));bandwidth=np.empty(len(train))
    training_event_error=np.empty(len(train));training_event_windows=np.zeros(len(train),dtype=int)
    biological_rr_sd=np.full(len(train),np.nan);detector_jitter_sd=np.full(len(train),np.nan)
    landmark_count=np.zeros((len(train),ns),dtype=int)
    detector_matched=np.zeros((len(train),ncomponents),dtype=int);detector_detected=np.zeros_like(detector_matched)
    detector_reference=np.zeros_like(detector_matched)
    detector_variance=np.full((len(train),ncomponents),np.nan)
    detector_precision=np.zeros((len(train),ncomponents));detector_sensitivity=np.zeros_like(detector_precision)
    if landmark_teacher:
        import zipfile,re
        archive=zipfile.ZipFile(data_folder/"adfecgdb-1.0.0.zip")
        archive_prefix=archive.namelist()[0].split("/")[0]+"/"
        recording=pd.read_csv(data_folder/"record_metadata.csv",encoding="utf-8").set_index("subject")
        window_number=truth.drop_duplicates("run").set_index("run").loc[data["runs"],"window"].to_numpy(dtype=int)
        window_seconds=float(json.loads((data_folder/"source.json").read_text(encoding="utf-8"))["window_seconds"])

    classifier_models=[];classifier_training_seconds=np.zeros(len(train))
    classifier_positive_fraction=np.zeros(len(train));classifier_training_timestamps=np.zeros(len(train),dtype=int)
    classifier_lags=np.zeros((len(train),5),dtype=int)
    learned_event_score=np.empty((len(train),ns,nt),dtype=np.float32) if supervised_detector else None
    if supervised_detector:
        from sklearn.ensemble import HistGradientBoostingClassifier
        import joblib
        covariance=signal@signal.transpose(0,2,1)/nt
        eigen,vector=np.linalg.eigh(covariance)
        keep=eigen>np.finfo(float).eps*nc*eigen[:,-1,None]
        inverse=np.divide(1.,np.sqrt(np.maximum(eigen,0.)),out=np.zeros_like(eigen),where=keep)
        whitening=(vector*inverse[:,None])@vector.transpose(0,2,1)
        whitened=whitening@signal

    def event_features(sigma):
        derivative=gaussian_filter1d(whitened,sigma=sigma,order=1,axis=-1,mode="reflect")*sigma
        curvature=gaussian_filter1d(whitened,sigma=sigma,order=2,axis=-1,mode="reflect")*sigma**2
        energy=np.stack([np.sum(whitened**2,axis=1),np.sum(derivative**2,axis=1),np.sum(curvature**2,axis=1)],axis=-1)
        base=np.concatenate([abs(whitened).transpose(0,2,1),(whitened**2).transpose(0,2,1),energy],axis=-1).astype(np.float32)
        lags=np.rint(np.arange(-2,3)*sigma).astype(int)
        features=np.concatenate(list(map(lambda lag:base[:,np.clip(np.arange(nt)+lag,0,nt-1)],lags)),axis=-1)
        return features,lags

    def training_qrs(name):
        # Same EDF+ byte layout and QRS parser as prepare_adfecg.py::read_record.
        content=archive.read(archive_prefix+str(recording.loc[name,"source_file"]))
        header=int(content[184:192]);channels=int(content[252:256]);n_records=int(content[236:244])
        samples=np.frombuffer(content[256+216*channels:256+224*channels],dtype="S8").astype(int)
        digital=np.frombuffer(content[header:],dtype="<i2").reshape(n_records,samples.sum())
        annotation=digital[:,-samples[-1]:].astype("<i2").tobytes()
        peaks=np.asarray(re.findall(rb"([+-]\d+(?:\.\d*)?)\x14QRS\x14\x00",annotation),dtype=float)
        assert len(peaks)>300 and np.all(np.diff(peaks)>0.)
        return peaks

    def detect(response,sigma):
        amplitude=np.abs(response)
        noise=float(np.median(amplitude)/.6744897501960817)
        effective=max(2.,nt/(2.*max(sigma,1.)))
        threshold=noise*np.sqrt(2.*np.log(effective))
        peaks,_=find_peaks(amplitude,distance=minimum_distance,height=threshold)
        return peaks,amplitude[peaks]

    def subtract_cycles(values,sigma):
        covariance=values@values.T/nt
        _,vector=np.linalg.eigh(covariance)
        consensus=vector[:,-1]@values
        response=gaussian_filter1d(consensus,sigma=sigma,order=2,mode="reflect")
        peaks,_=detect(response,sigma)
        residual=values.copy()
        if len(peaks)<3:
            return residual,peaks,0.
        middle=(peaks[1:]+peaks[:-1])/2.
        current=peaks[1:-1];left=middle[:-1];right=middle[1:]
        size=2*int(np.ceil(np.median(np.diff(peaks))/2.))+1
        phase=np.linspace(-1.,1.,size)
        locations=current[:,None]+np.where(phase[None]<0.,phase[None]*(current-left)[:,None],phase[None]*(right-current)[:,None])
        segments=np.stack(list(map(lambda channel:np.interp(locations.ravel(),np.arange(nt),values[channel]).reshape(len(current),size),np.arange(nc))),axis=1)
        template=np.median(segments,axis=0)
        def cycle(index):
            positions=np.arange(int(np.ceil(left[index])),int(np.ceil(right[index])))
            actual=np.where(positions<current[index],(positions-current[index])/(current[index]-left[index]),(positions-current[index])/(right[index]-current[index]))
            fitted=np.stack(list(map(lambda channel:np.interp(actual,phase,template[channel]),np.arange(nc))))
            denominator=np.sum(fitted**2,axis=1)
            gain=np.divide(np.sum(values[:,positions]*fitted,axis=1),denominator,out=np.zeros(nc),where=denominator>0.)
            residual[:,positions]-=gain[:,None]*fitted
            return None
        list(map(cycle,np.arange(len(current))))
        fraction=1.-float(np.sum(residual**2)/np.sum(values**2))
        return residual,peaks,fraction

    def fit_case(case):
        take=np.flatnonzero(train[case]);records=[];rr_error=[]
        training_peaks={};timing_error=[[] for _ in np.arange(ncomponents)]
        if landmark_teacher:
            # Held-out subjects are never passed to the annotation reader.
            names=np.unique(subject[take]);annotations=dict(zip(names,map(training_qrs,names)))
            def foreground_window(index):
                absolute=annotations[subject[index]];start=window_number[index]*window_seconds
                peaks=absolute[(absolute>=start)&(absolute<start+window_seconds)]-start
                training_peaks[index]=peaks;landmark_count[case,index]=len(peaks)
                positions=np.rint(peaks/dt).astype(int)
                inside=positions[(positions>=half)&(positions<nt-half)]
                shape=fg[index,inside[:,None]+offset[None]];shape-=shape.mean(1)[:,None]
                largest=np.argmax(abs(shape),axis=1)
                shape*=np.sign(shape[np.arange(len(shape)),largest])[:,None]
                norm=np.sqrt(np.sum(shape**2,axis=1))
                records.append(np.divide(shape,norm[:,None],out=np.zeros_like(shape),where=norm[:,None]>0.))
                intervals=np.diff(absolute);midpoint=(absolute[:-1]+absolute[1:])/2.
                owned=np.floor(midpoint/window_seconds).astype(int)==window_number[index]
                current=intervals[owned];rr_error.append(current-ref[index])
                return float(current.mean())
            estimated=np.asarray(list(map(foreground_window,take)))
            template=np.median(np.concatenate(records,axis=0),axis=0);template-=template.mean();template/=np.sqrt(np.sum(template**2))
            spectrum=rfft(template,n=next_fast_len(2*len(template)-1))
            ac=irfft(spectrum*spectrum.conj(),n=next_fast_len(2*len(template)-1))[:half+1];ac/=ac[0]
            crossing=np.flatnonzero(ac<=0.);end=int(crossing[0]) if len(crossing) else len(ac)
            sigma=max(1.,float(np.sum(ac[:end])/np.sqrt(np.pi)))
            match_width=min(half,max(1,end))*dt
            errors=np.concatenate(rr_error);biological_rr_sd[case]=float(np.sqrt(np.mean(errors**2)))
            np.testing.assert_allclose(estimated,ref[take],rtol=0.,atol=1e-12)
        else:
            # Original automatic teacher retained as the comparison entry.
            spectrum=rfft(fg[take],n=next_fast_len(2*nt-1),axis=-1)
            ac=irfft(spectrum*spectrum.conj(),n=next_fast_len(2*nt-1),axis=-1)[:,:half+1]
            ac/=np.maximum(ac[:,:1],np.finfo(float).tiny)
            average=weights[case,take]@ac
            crossing=np.flatnonzero(average<=0.);end=int(crossing[0]) if len(crossing) else len(average)
            sigma=max(1.,float(np.sum(average[:end])/np.sqrt(np.pi)))
            def foreground_window(index):
                response=gaussian_filter1d(fg[index],sigma=sigma,order=2,mode="reflect")
                peaks,_=detect(response,sigma)
                inside=peaks[(peaks>=half)&(peaks<nt-half)]
                shape=fg[index,inside[:,None]+offset[None]];shape-=shape.mean(1)[:,None]
                shape*=np.sign(response[inside])[:,None]
                norm=np.sqrt(np.sum(shape**2,axis=1))
                records.append(np.divide(shape,norm[:,None],out=np.zeros_like(shape),where=norm[:,None]>0.))
                intervals=np.diff(peaks)*dt;valid=(intervals>=pmin)&(intervals<=pmax)
                rr_error.append(intervals[valid]-ref[index])
                return float(np.mean(intervals[valid])) if np.any(valid) else np.nan
            estimated=np.asarray(list(map(foreground_window,take)))
            template=np.median(np.concatenate(records,axis=0),axis=0);template-=template.mean();template/=np.sqrt(np.sum(template**2))
            errors=np.concatenate(rr_error)
            bandwidth[case]=max(float(np.sqrt(np.mean(errors**2))),float(period[1]-period[0])/np.sqrt(12.))
        shape_sigma[case]=sigma*dt;templates[case]=template
        training_event_windows[case]=int(np.isfinite(estimated).sum())
        training_event_error[case]=float(np.nanmean(abs(estimated-ref[take])))*1000.
        if supervised_detector:
            x,lags=event_features(sigma);classifier_lags[case]=lags
            tolerance=max(1,int(np.ceil(sigma)))
            labels=np.zeros((len(take),nt),dtype=bool)
            def training_labels(position):
                peaks=np.rint(training_peaks[take[position]]/dt).astype(int)
                indices=peaks[:,None]+np.arange(-tolerance,tolerance+1)[None]
                valid=(indices>=0)&(indices<nt)
                labels[position,indices[valid]]=True
                return None
            list(map(training_labels,np.arange(len(take))))
            source_weight=weights[case,take]
            positive=float(np.sum(source_weight[:,None]*labels)/nt)
            sample_weight=np.broadcast_to(source_weight[:,None],labels.shape)*np.where(labels,.5/positive,.5/(1.-positive))
            sample_weight*=len(take)
            estimator=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,early_stopping=False,random_state=CONFIG["seed"])
            train_started=clock.perf_counter()
            print("SUPERVISED_EVENT_FIT_START",case,"sources",len(take),"timestamps",labels.size,"features",x.shape[-1],"positive_fraction",positive,"lags_samples",lags.tolist(),flush=True)
            # sklearn samples quantile thresholds by default. Use all training
            # timestamps for binning as well as boosting, then restore its native method.
            original_bin_data=estimator._bin_data
            def full_timestamp_binning(values,is_training_data):
                if is_training_data:
                    estimator._bin_mapper.subsample=None
                return original_bin_data(values,is_training_data)
            estimator._bin_data=full_timestamp_binning
            estimator.fit(x[take].reshape(-1,x.shape[-1]),labels.ravel(),sample_weight=sample_weight.ravel())
            del estimator._bin_data
            assert estimator._bin_mapper.subsample is None
            classifier_training_seconds[case]=clock.perf_counter()-train_started
            classifier_positive_fraction[case]=positive;classifier_training_timestamps[case]=labels.size
            score=estimator.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(ns,nt).astype(np.float32)
            learned_event_score[case]=score;classifier_models.append(estimator)
            joblib.dump({"models":classifier_models,"case_index":np.arange(case+1),"shape_sigma_s":shape_sigma[:case+1],
                "lags_samples":classifier_lags[:case+1],"seed":CONFIG["seed"],"feature_kind":"No_ZCA_absolute_square_space_derivative_curvature_5lags",
                "training_run":all_train[:case+1],"reference_source_count":train[:case+1].sum(1),"binning_all_timestamps":True},folder/(output_tag+"_detector.joblib"))
            del x,labels,sample_weight
            print("SUPERVISED_EVENT_FIT_DONE",case,"iterations",estimator.n_iter_,"fit_seconds",classifier_training_seconds[case],"elapsed",clock.perf_counter()-started,flush=True)
        def window(index):
            if supervised_detector:
                residual=signal[index];peaks=np.asarray([],dtype=int);fraction=0.
            else:
                residual,peaks,fraction=subtract_cycles(signal[index],sigma)
            background_count[case,index]=len(peaks);background_time[case,index,:len(peaks)]=data["time"][peaks]
            removed[case,index]=fraction
            if supervised_detector:
                components=score[index,None]
            elif landmark_teacher:
                # Synchronous leads measure one FG source; PCs are not extra sources.
                components=residual
            else:
                _,vector=np.linalg.eigh(residual@residual.T/nt)
                components=np.concatenate([residual,vector.T@residual],axis=0)
            scale=np.sqrt(np.mean(components**2,axis=1))
            components=np.divide(components,scale[:,None],out=np.zeros_like(components),where=scale[:,None]>0.)
            def component(channel):
                if supervised_detector:
                    response=score[index]
                    peaks,properties=find_peaks(response,distance=minimum_distance,height=.5)
                    amplitude=response[peaks]
                else:
                    response=fftconvolve(components[channel],template[::-1],mode="same")
                    peaks,amplitude=detect(response,sigma)
                event_count[case,index,channel]=len(peaks);event_time[case,index,channel,:len(peaks)]=data["time"][peaks]
                if landmark_teacher and train[case,index]:
                    teacher=training_peaks[index];observed=data["time"][peaks]
                    right=np.searchsorted(teacher,observed);before=np.clip(right-1,0,len(teacher)-1);after=np.clip(right,0,len(teacher)-1)
                    first=observed-teacher[before];second=observed-teacher[after]
                    error=np.where(abs(first)<=abs(second),first,second)
                    matched=abs(error)<=match_width
                    timing_error[channel].append(error[matched])
                    detector_matched[case,channel]+=int(matched.sum());detector_detected[case,channel]+=len(peaks)
                    detector_reference[case,channel]+=len(teacher)
                intervals=np.diff(peaks)*dt
                # Every detected adjacent interval enters the renewal likelihood.
                valid=np.ones(len(intervals),dtype=bool) if landmark_teacher else ((intervals>=pmin)&(intervals<=pmax))
                interval_count[case,index,channel]=int(valid.sum())
                loading=np.sqrt(amplitude[1:]*amplitude[:-1])[valid]
                return intervals[valid],loading
            return list(map(component,np.arange(ncomponents)))
        measured=list(map(window,np.arange(ns)))
        if landmark_teacher:
            def component_variance(channel):
                error=np.concatenate(timing_error[channel])
                # Wrong detections remain in precision counts, never in biological RR scatter.
                return float(np.mean((error-error.mean())**2)) if len(error)>1 else (pmax-pmin)**2/12.
            detector_variance[case]=np.asarray(list(map(component_variance,np.arange(ncomponents))))
            jitter=float(np.mean(detector_variance[case]));detector_jitter_sd[case]=np.sqrt(jitter)
            detector_precision[case]=np.divide(detector_matched[case],detector_detected[case],out=np.zeros(ncomponents),where=detector_detected[case]>0)
            detector_sensitivity[case]=np.divide(detector_matched[case],detector_reference[case],out=np.zeros(ncomponents),where=detector_reference[case]>0)
            bandwidth[case]=np.sqrt(biological_rr_sd[case]**2+2.*jitter)
        def density_window(index):
            def density(pair):
                intervals,loading=pair
                if not len(intervals):
                    return np.ones(len(period))/len(period)
                kernel=np.exp(-.5*((period[None]-intervals[:,None])/bandwidth[case])**2)
                kernel/=np.maximum(kernel.sum(1),np.finfo(float).tiny)[:,None]
                return loading@kernel/max(float(loading.sum()),np.finfo(float).tiny)
            if landmark_teacher:
                cycles=np.arange(1,int(np.floor(window_seconds/pmin))+1,dtype=float)
                def renewal(channel):
                    intervals,_=measured[index][channel]
                    if not len(intervals):
                        return np.zeros(len(period))
                    sensitivity=float(np.clip(detector_sensitivity[case,channel],0.,1.))
                    precision=float(np.clip(detector_precision[case,channel],0.,1.))
                    missing=1.-sensitivity;real_pair=precision**2
                    geometric=sensitivity*np.power(missing,cycles-1.)
                    geometric=np.divide(geometric,geometric.sum(),out=np.full(len(cycles),1./len(cycles)),where=geometric.sum()>0.)
                    variance=cycles*biological_rr_sd[case]**2+2.*detector_variance[case,channel]
                    log_weight=np.log(np.maximum(geometric,np.finfo(float).tiny))
                    log_normal=-.5*((intervals[:,None,None]-cycles[None,:,None]*period[None,None])**2/variance[None,:,None]+np.log(2.*np.pi*variance)[None,:,None])
                    real=logsumexp(log_normal+log_weight[None,:,None],axis=1)
                    log_real=np.log(real_pair) if real_pair>0. else -np.inf
                    log_clutter=np.log1p(-real_pair)-np.log(window_seconds) if real_pair<1. else -np.inf
                    return np.logaddexp(log_real+real,log_clutter).sum(axis=0)
                # All intervals inform one mean-period parameter; synchronous leads are composite observations.
                curves[case,index]=np.mean(np.stack(list(map(renewal,np.arange(ncomponents)))),axis=0)
            else:
                mass=np.stack(list(map(density,measured[index])))
                curves[case,index]=np.log(np.maximum(mass.mean(axis=0),np.finfo(float).tiny))
            return None
        list(map(density_window,np.arange(ns)))
        print("EVENT_PHASE_CASE",case,"N",len(take),"G",int((counts[case]>0).sum()),"shape_sigma_s",shape_sigma[case],"interval_bw_s",bandwidth[case],"training_FG_event_windows",training_event_windows[case],"training_FG_event_mean_error_ms",training_event_error[case],"elapsed_s",clock.perf_counter()-started,flush=True)
        return None
    list(map(fit_case,np.arange(len(train))))
    teacher=period_reference_mass(period,ref,qualified)
    mean=weights@np.where(qualified,ref,0.)
    sd=np.sqrt(np.maximum(weights@np.where(qualified,ref**2,0.)-mean**2,(period[1]-period[0])**2/12.))
    prior=-.5*((period[None]-mean[:,None])/sd[:,None])**2;prior-=logsumexp(prior,axis=1,keepdims=True)
    neffective=train.sum(1)
    pooled_training_mean=np.sum(train*np.where(qualified,ref,0.)[None],axis=1)/neffective
    objective=lambda theta:(beta:=np.exp(theta),logits:=beta[:,None,None]*curves+prior[:,None],logp:=logits-logsumexp(logits,axis=-1,keepdims=True),
        (-(weights[:,:,None]*teacher[None]*logp).sum()+.5*np.sum(theta**2/neffective),beta*np.einsum("fn,fnp,fnp->f",weights,np.exp(logp)-teacher[None],curves)+theta/neffective))[-1]
    fit=minimize(objective,np.zeros(len(train)),jac=True,method="L-BFGS-B",options={"maxiter":300,"ftol":1e-10,"gtol":1e-7})
    posterior=softmax(np.exp(fit.x)[:,None,None]*curves+prior[:,None],axis=-1);estimate=posterior@period
    case,window=np.indices(estimate.shape)
    cells=pd.DataFrame({"case":case.ravel(),"run":np.tile(data["runs"],len(train)),"subject":np.tile(subject,len(train)),
        "qualified":np.tile(qualified,len(train)),"training":all_train.ravel(),"query":np.r_[np.zeros((1,ns),dtype=bool),~all_train[1:]].ravel(),
        "estimate_s":estimate.ravel(),"reference_s":np.tile(ref,len(train)),"training_mean_s":np.repeat(mean,ns),
        "pooled_training_mean_s":np.repeat(pooled_training_mean,ns),
        "background_events":background_count.ravel(),"background_variance_removed":removed.ravel(),
        "mean_residual_events":event_count.mean(-1).ravel(),"interval_components_available":(interval_count>0).sum(-1).ravel()})
    cells["absolute_error_ms"]=np.where(cells.qualified,1000*abs(cells.estimate_s-cells.reference_s),np.nan)
    cells["baseline_error_ms"]=np.where(cells.qualified,1000*abs(cells.training_mean_s-cells.reference_s),np.nan)
    cells["pooled_baseline_error_ms"]=np.where(cells.qualified,1000*abs(cells.pooled_training_mean_s-cells.reference_s),np.nan)
    cells.to_csv(folder/(output_tag+"_cells.csv"),index=False,encoding="utf-8")
    query=cells.loc[cells["query"]].copy();query.to_csv(folder/(output_tag+"_query.csv"),index=False,encoding="utf-8")
    np.savez_compressed(folder/(output_tag+".npz"),run_names=data["runs"],period=period,time=data["time"],training_run=all_train,
        reference_source_count=neffective,effective_units=(counts>0).sum(1),training_reference_access_mask=train,
        reference_source_id=data["runs"],measurement_source_id=data["metadata"].run.to_numpy(dtype=str),
        source_measurement_count=np.asarray(nc),
        source_likelihood_kind=np.asarray("renewal_single_source_supervised_event_stream" if supervised_detector else "renewal_mean_log_original_synchronous_leads" if landmark_teacher else "arithmetic_mixture_original_leads_and_residual_PCs"),
        pooled_training_mean=pooled_training_mean,subject_balanced_training_mean=mean,
        foreground_event_template=templates,shape_sigma_s=shape_sigma,
        event_interval_bandwidth_s=bandwidth,biological_RR_sd_s=biological_rr_sd,No_detector_jitter_sd_s=detector_jitter_sd,
        training_landmark_count=landmark_count,No_detector_matched_events=detector_matched,
        No_detector_detected_events=detector_detected,No_detector_reference_events=detector_reference,
        No_detector_precision=detector_precision,No_detector_sensitivity=detector_sensitivity,
        No_detector_timing_variance_s2=detector_variance,
        period_estimand=np.asarray("window_mean_RR_shared_renewal_period_parameter" if landmark_teacher else "single_interval_density"),
        training_reference_information=np.asarray("Direct_1_and_cardiologist_QRS" if landmark_teacher else "Direct_1_automatic_events"),
        training_FG_event_windows=training_event_windows,
        training_FG_event_failed_windows=neffective-training_event_windows,training_FG_event_mean_error_ms=training_event_error,
        background_event_time=background_time,background_event_count=background_count,background_variance_removed=removed,
        residual_event_time=event_time,residual_event_count=event_count,interval_count=interval_count,
        event_log_density=curves,source_interval_log_likelihood=curves,foreground_log_prior=prior,temperature=np.exp(fit.x),source_period=estimate,posterior=posterior,
        fit_success=np.asarray(fit.success),reference=ref,qualified=qualified,query=np.r_[np.zeros((1,ns),dtype=bool),~all_train[1:]],
        method=np.asarray("supervised_No_FG_event_shared_source_renewal_meanRR" if supervised_detector else "No_phase_median_nuisance_reference_event_interval_density"),
        No_input_kind=np.asarray("curvature_preprocessed_mixtures" if supervised_detector else "phase_template_residual"),
        binning_all_timestamps=np.asarray(supervised_detector))
    report=query.groupby("subject").agg(windows=("run","size"),qualified=("qualified","sum"),mae_ms=("absolute_error_ms","mean"),baseline_ms=("baseline_error_ms","mean"),pooled_baseline_ms=("pooled_baseline_error_ms","mean"),
        background_events=("background_events","median"),variance_removed=("background_variance_removed","median"),available_components=("interval_components_available","mean"))
    if supervised_detector:
        np.savez_compressed(folder/(output_tag+"_detector_scores.npz"),event_score=learned_event_score,
            shape_sigma_s=shape_sigma,lags_samples=classifier_lags,training_timestamps=classifier_training_timestamps,
            training_positive_fraction=classifier_positive_fraction,fit_seconds=classifier_training_seconds,
            reference_source_count=neffective,effective_units=(counts>0).sum(1),training_reference_access_mask=train,
            training_precision=detector_precision,training_recall=detector_sensitivity,time=data["time"],run_names=data["runs"],
            feature_kind=np.asarray("No_ZCA_absolute_square_space_derivative_curvature_5lags"),
            positive_event_tolerance_s=np.ceil(shape_sigma/dt)*dt,binning_all_timestamps=np.asarray(True),
            hyperparameters=np.asarray("HistGradientBoostingClassifier max_iter100 max_leaf_nodes15 early_stoppingFalse seed12345 full_timestamp_binning"))
        print("SUPERVISED_EVENT_TRAIN_PRECISION_RECALL",detector_precision.ravel().tolist(),detector_sensitivity.ravel().tolist(),flush=True)
    print("EVENT_PHASE_REFERENCE",fit.success,"N",neffective.tolist(),"temperature",np.exp(fit.x).tolist(),"elapsed_s",clock.perf_counter()-started,flush=True)
    print(report.to_string(),flush=True)
    print("EVENT_PHASE_MACRO_MAE_MS",report.mae_ms.mean(),"SUBJECT_BALANCED_BASELINE",report.baseline_ms.mean(),"POOLED_BASELINE",report.pooled_baseline_ms.mean(),flush=True)
    raise SystemExit(0)
if args.spatial_reference_probe:
    from source_assignment_unified import physical_autocorrelation,period_reference_mass,shared_background
    data=observations(frame)
    truth=reference.loc[data["metadata"].index]
    ns=len(data["runs"]);nt=len(data["time"]);nc=len(truth)//ns
    assert np.all(data["count"]==nt) and np.all(data["membership"].sum(1)==nc)
    signal=data["signal"].reshape(ns,nc,nt)
    fg_frame=pd.read_parquet(data_folder/"foreground_reference.parquet")
    fg=fg_frame.pivot(index="run",columns="x",values="foreground_y").loc[data["runs"]].to_numpy(dtype=float)
    combined=np.concatenate([signal.reshape(-1,nt),fg])
    combined_data=dict(data,signal=combined,mask=np.ones_like(combined,dtype=bool),count=np.full(len(combined),nt))
    _,periodic=shared_background(combined_data,1.5,strength=(2.*np.pi)**-4,per_record=True)
    signal=periodic[:ns*nc].reshape(ns,nc,nt);fg=periodic[ns*nc:]
    projection=np.einsum("nct,nt->nc",signal,fg)/np.sum(fg**2,axis=1)[:,None]
    background=signal-projection[:,:,None]*fg[:,None]
    yf=rfft(signal,axis=-1)
    ff=rfft(fg,axis=-1);bf=rfft(background,axis=-1)
    fpower=abs(ff)**2;fpower/=fpower.sum(1)[:,None]
    bpower=np.mean(abs(bf)**2/np.sum(abs(bf)**2,axis=-1)[:,:,None],axis=1)
    all_train=np.r_[np.ones((1,ns),dtype=bool),context["outer_train_run"][:,pd.Index(context["run_name"]).get_indexer(data["runs"])]]
    qualified=(data["membership"]@truth.qualified.to_numpy(dtype=float))>0.
    subject=truth.drop_duplicates("run").set_index("run").loc[data["runs"],"subject"].to_numpy(dtype=str)
    unit_code,unit_names=pd.factorize(subject,sort=True)
    unit=(np.arange(len(unit_names))[:,None]==unit_code[None]).astype(float)
    train=all_train&qualified[None]
    counts=train.astype(float)@unit.T
    weights=np.divide(train.astype(float),counts[:,unit_code],out=np.zeros_like(train,dtype=float),where=counts[:,unit_code]>0.)
    weights/=weights.sum(1)[:,None]
    ftemplate=weights@fpower;btemplate=weights@bpower
    covariance=np.einsum("nct,ndt->ncd",signal,signal)/nt
    foreground_covariance=np.einsum("fk,nck,ndk->fncd",ftemplate,yf,yf.conj(),optimize=True).real
    background_covariance=np.einsum("fk,nck,ndk->fncd",btemplate,yf,yf.conj(),optimize=True).real
    eigen,vector=np.linalg.eigh(background_covariance)
    threshold=np.finfo(float).eps*nc*eigen[...,-1:]
    inverse=np.divide(1.,np.sqrt(np.maximum(eigen,0.)),out=np.zeros_like(eigen),where=eigen>threshold)
    whitening=(vector*inverse[:,: ,None])@vector.swapaxes(-1,-2)
    matrices=whitening@foreground_covariance@whitening
    strength,rotation=np.linalg.eigh(matrices)
    spatial=np.einsum("fnij,fnj->fni",whitening,rotation[:,:,:,-1],optimize=True)
    extracted=np.einsum("fnc,nct->fnt",spatial,signal,optimize=True)
    corr=abs(np.einsum("fnt,nt->fn",extracted,fg))/np.sqrt(np.sum(extracted**2,axis=-1)*np.sum(fg**2,axis=-1)[None])
    original_corr=abs(np.einsum("nct,nt->nc",signal,fg))/np.sqrt(np.sum(signal**2,axis=-1)*np.sum(fg**2,axis=-1)[:,None])
    cross=np.einsum("nct,nt->nc",signal,fg)/nt
    oracle=np.sqrt(np.einsum("nc,nc->n",cross,np.linalg.solve(covariance,cross[:,:,None])[:,:,0])/np.mean(fg**2,axis=-1))
    period=np.linspace(.25,1.5,73)
    curve=physical_autocorrelation(extracted.reshape(-1,nt),np.ones((len(train)*ns,nt),dtype=bool),data["time"],period).reshape(len(train),ns,len(period))
    reference_window=(data["average"]@truth.reference_period_s.to_numpy())
    teacher=period_reference_mass(period,reference_window,qualified)
    mean=weights@np.where(qualified,reference_window,0.)
    sd=np.sqrt(np.maximum(weights@np.where(qualified,reference_window**2,0.)-mean**2,(period[1]-period[0])**2/12.))
    prior=-.5*((period[None]-mean[:,None])/sd[:,None])**2
    prior-=logsumexp(prior,axis=1,keepdims=True)
    neffective=train.sum(1)
    objective=lambda theta: (
        beta:=np.exp(theta),
        logits:=beta[:,None,None]*curve+prior[:,None],
        logp:=logits-logsumexp(logits,axis=-1,keepdims=True),
        (-(weights[:,:,None]*teacher[None]*logp).sum()+.5*np.sum(theta**2/neffective),
            beta*np.einsum("fn,fnp,fnp->f",weights,np.exp(logp)-teacher[None],curve)+theta/neffective))[-1]
    fit=minimize(objective,np.zeros(len(train)),jac=True,method="L-BFGS-B",options={"maxiter":300,"ftol":1e-10,"gtol":1e-7})
    posterior=softmax(np.exp(fit.x)[:,None,None]*curve+prior[:,None],axis=-1)
    estimate=posterior@period
    query=(~all_train[1:])&qualified[None]
    case,window=np.nonzero(query)
    summary=pd.DataFrame({"subject":subject[window],"run":data["runs"][window],"estimate_s":estimate[1:][case,window],"reference_s":reference_window[window],
        "absolute_error_ms":1000*abs(estimate[1:][case,window]-reference_window[window]),"extracted_reference_abs_correlation":corr[1:][case,window],
        "best_original_reference_abs_correlation":original_corr.max(1)[window]})
    summary["linear_span_oracle_correlation"]=oracle[window]
    summary.to_csv(folder/"adfecg_spatial_periodic_reference_query.csv",index=False,encoding="utf-8")
    np.savez_compressed(folder/"adfecg_spatial_periodic_reference.npz",run_names=data["runs"],period=period,training_run=all_train,
        foreground_source_count=train.sum(1),effective_units=(counts>0.).sum(1),foreground_spectral_template=ftemplate,background_spectral_template=btemplate,
        extracted_signal=extracted,spatial_filter=spatial,source_period=estimate,posterior=posterior,acf=curve,temperature=np.exp(fit.x),
        fit_success=np.asarray(fit.success),query=query,reference=reference_window,qualified=qualified,source_id=data["runs"],time=data["time"])
    print("SPATIAL_REFERENCE",fit.success,"N",train.sum(1).tolist(),"G",(counts>0.).sum(1).tolist(),"temperature",np.exp(fit.x).tolist(),flush=True)
    print(summary.groupby("subject").agg(n=("run","size"),mae_ms=("absolute_error_ms","mean"),correlation=("extracted_reference_abs_correlation","median"),best_input_correlation=("best_original_reference_abs_correlation","median"),span_oracle=("linear_span_oracle_correlation","median")).to_string(),flush=True)
    print("SPATIAL_REFERENCE_MACRO_MAE_MS",summary.groupby("subject").absolute_error_ms.mean().mean(),flush=True)
    raise SystemExit(0)
model=dict(np.load(model_path,allow_pickle=False))
conditional=str(model.get("readout_kind",""))=="conditional_latent_representation_density"
source_family=conditional or str(model.get("schema",""))=="source_assignment_unified_0.3.1"
if source_family:
    cache=args.feature_cache or folder/("adfecg_observed_features.npz" if conditional or str(model["source_representation"])=="observed_physical_peer" else "adfecg_features.npz")
    bundle=dict(np.load(cache,allow_pickle=False))
    assert conditional or str(bundle["source_representation"])==str(model["source_representation"])
    np.testing.assert_array_equal(bundle["trace_id"],model["trace_id"])
    qualified_reference=reference.loc[bundle["trace_id"],"qualified"].to_numpy(dtype=bool)
    fit={"trace_id":bundle["trace_id"],"qualified":qualified_reference,
        "training_mask":model["training_run"][:,bundle["run_index"]]&qualified_reference[None],
        "methods":np.asarray(["source_roles"]),"estimate":model["estimate"][None],
        "optimizer_success_by_case":model["success"][None]}
    model=dict(model,methods=fit["methods"],parameter_policy=np.asarray(["conditional_source_density_joint_objective" if conditional else "shared_measurement_density_and_source_references"]),method_applicable=np.asarray([True]))
else:
    fit=dict(np.load(folder/"adfecg_fit.npz",allow_pickle=False))
    assert str(model["model_kind"])==MODEL_KIND and str(model["source_operator"])==SOURCE_OPERATOR
    assert str(model["source_core_policy"])==CORE_POLICY
    list(map(lambda key:np.testing.assert_array_equal(fit[key],model[key]),["model_kind","source_operator","source_core_policy"]))
    bundle=dict(np.load(folder/"adfecg_features.npz",allow_pickle=False))
np.testing.assert_array_equal(fit["qualified"],reference.loc[fit["trace_id"],"qualified"].to_numpy(),
    err_msg="Reference QC differs from the saved fit; refit before matched evaluation")
fit_order=pd.Index(fit["methods"]).get_indexer(model["methods"])
assert np.all(fit_order>=0)
if not source_family:
    list(map(lambda key:np.testing.assert_array_equal(fit[key][fit_order],model[key]),POOL_PARAMETER_KEYS))
case_success=fit["optimizer_success_by_case"]
if not source_family:
    assert str(bundle["source_operator"])==SOURCE_OPERATOR
np.testing.assert_array_equal(bundle["period"],model["period"])
np.testing.assert_array_equal(bundle["trace_id"],fit["trace_id"])
observed=frame.groupby("trace_id").size().reindex(bundle["trace_id"])
np.testing.assert_array_equal(bundle["count"],observed)
np.testing.assert_array_equal(bundle["time"],np.sort(frame.x.unique()))
if source_family:
    if conditional:
        balanced=dict(np.load(folder/"adfecg_features.npz",allow_pickle=False))
        value=predict_latent_representation_density(bundle,balanced,model,device=args.device)
        value["original_period"]=np.full_like(value["estimate"],np.nan)
    else:
        value=predict_general_source_roles(bundle,model)
    fresh={key:value[key][None] for key in ["estimate","source_period","original_period","group_mean"]}
    fresh["population_location"]=np.full_like(fresh["group_mean"],np.nan)
else:
    fresh=prediction_matrix(bundle,model,np.arange(model["anchor_coefficient"].shape[1]))
readback_error=float(np.max(abs(fresh["estimate"]-fit["estimate"][fit_order])))
assert np.isfinite(fresh["estimate"]).all() and readback_error<=1e-6
print("ADFECG_FRESH_PREDICTION_READBACK",readback_error,flush=True)
order=pd.Index(context["trace_id"]).get_indexer(bundle["trace_id"])
query=context["outer_query"][:,order]
truth=reference.loc[bundle["trace_id"]].reset_index()
case,cell=np.nonzero(query)
method_count=len(model["methods"])
estimate=fresh["estimate"][:,1:][:,case,cell]
run=bundle["run_index"][cell]
feature_success=np.ones(len(cell),dtype=bool) if source_family else bundle["pool_period_fit_success"][run]
predictions=pd.DataFrame({"method":np.repeat(model["methods"],len(cell)),"case":np.tile(case,method_count),
    "trace_id":np.tile(bundle["trace_id"][cell],method_count),"estimate_period_s":estimate.ravel(),
    "source_period_s":fresh["source_period"][:,1:][:,case,cell].ravel(),
    "original_period_s":fresh["original_period"][:,1:][:,case,cell].ravel(),
    "finite_cohort_mean_s":fresh["group_mean"][:,1:][:,case,run].ravel(),
    "population_location_s":fresh["population_location"][:,1:][:,case,run].ravel(),
    "fit_status":np.where(case_success[fit_order][:,case+1]&feature_success[None],"converged","not_converged").ravel(),
    "parameter_policy":np.repeat(model["parameter_policy"],len(cell)),
    "applicable":np.repeat(model["method_applicable"],len(cell))})
period=bundle["period"]
target=truth.reference_period_s.to_numpy()
training=context["outer_train_run"][:,pd.Index(context["run_name"]).get_indexer(bundle["run_names"])]
qualified=truth.qualified.to_numpy(dtype=bool)
train=training[:,bundle["run_index"]]&qualified[None]
np.testing.assert_array_equal(fit["training_mask"][1:],train)
if conditional:
    expected_sources=(train.astype(float)@bundle["membership"].T>0.).sum(1)
    np.testing.assert_array_equal(model["reference_source_count"][1:],expected_sources)
    assert int(model["reference_source_count"][0])==int(np.sum(bundle["membership"]@qualified.astype(float)>0.))
weight=train/train.sum(axis=1)[:,None]
prior_mean=weight@np.where(qualified,target,0.)
if source_family:
    prior=softmax(model["foreground_log_prior"][1:] if conditional else foreground_reference_log_prior(bundle,model,np.arange(1,len(model["success"]))),axis=-1)
else:
    unified_index=pd.Index(model["methods"]).get_indexer(["unified"]).item()
    assert unified_index>=0
    prior=softmax(model["foreground_log_prior"][unified_index,1:],axis=-1)
power=np.clip(bundle["identity_base_features"][:,:,0],0.,1.-1e-8)
prior_weighted=power[None]*prior[:,None]
prior_weighted/=np.maximum(prior_weighted.sum(axis=2)[:,:,None],1e-300)
prior_estimate=prior_weighted@period
dominant=period[power.argmax(axis=1)]
baselines=pd.DataFrame({"method":np.repeat(["training_period_mean","dominant_H1","prior_weighted_H1"],len(cell)),
    "case":np.tile(case,3),"trace_id":np.tile(bundle["trace_id"][cell],3),
    "estimate_period_s":np.concatenate([prior_mean[case],dominant[cell],prior_estimate[case,cell]]),
    "fit_status":"not_fitted","applicable":True})
if source_family:
    population_mean=model["foreground_reference_mean"][1:]*float(model["period_scale"])
    baselines=pd.concat([baselines,pd.DataFrame({"method":"training_unit_period_mean","case":case,
        "trace_id":bundle["trace_id"][cell],"estimate_period_s":population_mean[case],"fit_status":"not_fitted","applicable":True})],ignore_index=True)
geometry=observations(frame)
signal=detrend(geometry["signal"],geometry["time"],geometry["mask"])
assert np.all(geometry["membership"].sum(axis=1)==4)
windows=signal.reshape(-1,4,signal.shape[1])


def unmix_window(value):
    estimator=FastICA(n_components=4,whiten="unit-variance",random_state=CONFIG["seed"],max_iter=500,tol=1e-4)
    separated=estimator.fit_transform(value.T).T
    return separated,int(estimator.n_iter_)


unmixed=list(map(unmix_window,windows))
components=np.stack(list(map(lambda item:item[0],unmixed))).reshape(signal.shape)
iterations=np.asarray(list(map(lambda item:item[1],unmixed)))
components=detrend(components,geometry["time"],geometry["mask"])
components/=np.sqrt(np.sum(components**2,axis=1)/geometry["count"])[:,None]
component_profile=harmonic_profile(components,geometry,period,float(model["period_scale"]),harmonics=1)
component_power=component_profile["energy_one"]/geometry["count"][:,None]
component_power=component_power.reshape(-1,4,len(period))
source_weight=component_power[None]*prior[:,None,None]
selected=source_weight.sum(axis=-1).argmax(axis=-1)
chosen=np.take_along_axis(source_weight,selected[:,:,None,None],axis=2)[:,:,0]
chosen/=np.maximum(chosen.sum(axis=-1)[:,:,None],1e-300)
ica_window=chosen@period
ica_cell=ica_window[:,bundle["run_index"]]
ica=pd.DataFrame({"method":"ICA_prior_weighted_H1","case":case,"trace_id":bundle["trace_id"][cell],
    "estimate_period_s":ica_cell[case,cell],
    "fit_status":np.where(iterations[bundle["run_index"]][cell]>=500,"not_converged","converged"),"applicable":True})
predictions=pd.concat([predictions,baselines,ica],ignore_index=True).merge(reference.reset_index(),on="trace_id",validate="many_to_one")
predictions["fit_status"]=pd.Categorical(predictions.fit_status,categories=["not_fitted","converged","not_converged"],ordered=True)
predictions["estimate_available"]=np.isfinite(predictions.estimate_period_s)
predictions["fit_not_converged"]=predictions.fit_status.eq("not_converged")
predictions["raw_mean_rr_error_ms"]=1000*abs(predictions.estimate_period_s-predictions.reference_period_s)
predictions["median_rr_error_ms"]=1000*abs(predictions.estimate_period_s-predictions.reference_median_rr_s)
predictions["absolute_error_ms"]=pd.Series(np.where(predictions.estimate_available,predictions.raw_mean_rr_error_ms,np.inf)).where(predictions.qualified&predictions.applicable)
predictions["relative_error"]=pd.Series(np.where(predictions.estimate_available,abs(predictions.estimate_period_s/predictions.reference_period_s-1),np.inf)).where(predictions.qualified&predictions.applicable)
predictions.to_csv(folder/"adfecg_cells.csv",index=False,encoding="utf-8")
subjects=predictions.groupby(["method","subject"],sort=False).agg(n_input=("trace_id","size"),
    n_reference=("qualified","sum"),n_estimated=("estimate_available","sum"),
    fit_status=("fit_status","max"),applicable=("applicable","all"),n_fit_not_converged=("fit_not_converged","sum"),
    period_mae_ms=("absolute_error_ms","mean"),relative_mae=("relative_error","mean")).reset_index()
subjects.to_csv(folder/"adfecg_subjects.csv",index=False,encoding="utf-8")
summary=subjects.groupby("method",sort=False).agg(subjects=("subject","size"),
    n_input=("n_input","sum"),n_reference=("n_reference","sum"),n_estimated=("n_estimated","sum"),
    fit_status=("fit_status","max"),applicable=("applicable","all"),n_fit_not_converged=("n_fit_not_converged","sum"),
    period_mae_ms=("period_mae_ms","mean"),relative_mae=("relative_mae","mean")).reset_index()
summary.to_csv(folder/"adfecg_summary.csv",index=False,encoding="utf-8")
win=predictions.groupby(["method","subject","window"],sort=False).agg(estimate_period_s=("estimate_period_s","mean"),
    reference_period_s=("reference_period_s","first"),qualified=("qualified","all"),
    reference_qc=("reference_qc","first"),fit_status=("fit_status","max"),applicable=("applicable","all")).reset_index().sort_values(["method","subject","window"])
win["estimated_change_s"]=win.groupby(["method","subject"]).estimate_period_s.diff()
win["reference_change_s"]=win.groupby(["method","subject"]).reference_period_s.diff()
win["change_reference_usable"]=win.qualified&win.groupby(["method","subject"]).qualified.shift(fill_value=False)&win.groupby(["method","subject"]).window.diff().eq(1)
win["absolute_change_error_ms"]=pd.Series(np.where(np.isfinite(win.estimated_change_s),1000*abs(win.estimated_change_s-win.reference_change_s),np.inf),index=win.index).where(win.change_reference_usable&win.applicable)
win.to_csv(folder/"adfecg_window_changes.csv",index=False,encoding="utf-8")
changes=win.groupby(["method","subject"],sort=False).agg(n_input_windows=("window","size"),
    n_reference_windows=("qualified","sum"),n_change_pairs=("change_reference_usable","sum"),
    fit_status=("fit_status","max"),applicable=("applicable","all"),change_mae_ms=("absolute_change_error_ms","mean")).reset_index()
changes.groupby("method",sort=False).agg(subjects=("subject","size"),n_input_windows=("n_input_windows","sum"),
    n_reference_windows=("n_reference_windows","sum"),n_change_pairs=("n_change_pairs","sum"),
    fit_status=("fit_status","max"),applicable=("applicable","all"),change_mae_ms=("change_mae_ms","mean")).reset_index().to_csv(folder/"adfecg_change_summary.csv",index=False,encoding="utf-8")
pd.DataFrame({"run":bundle["run_names"],"ICA_iterations":iterations,"ICA_reached_limit":iterations>=500}).to_csv(folder/"adfecg_ICA_status.csv",index=False,encoding="utf-8")
print("ADFECG_SUBJECT_HELD_OUT",summary.to_string(index=False),flush=True)
