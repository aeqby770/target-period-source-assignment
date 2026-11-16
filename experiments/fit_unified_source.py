"""Fit reference-supervised source-period models and replay No-only predictions."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1],"output":"results/0.3.1",
    "reference_inputs":"results/0.3.0"},"threads":os.cpu_count(),"maxiter":900,
    "schema":"source_assignment_unified_0.3.1",
    "readout_kind":"information_conditioned_reference_W2",
    "model_kind":"information_conditioned_reference_W2",
    "source_operator":"actual_sampling_reference_calibration_density_mixture",
    "source_core_policy":"lawful_reference_calibrations_one_final_period_distribution",
    "feature_fields":["period","period_scale","trace_id","run","condition","count","time","mask",
        "run_names","run_index","membership","average","identity_base_features","source_base","source_rhs",
        "joint_gram","layout","source_coherent_power","source_model_probability","source_representation",
        "source_order_policy","mixing_quality"],
    "parent_parameters":["core_coefficient","role_coefficient","background_coefficient","role_covariance",
        "foreground_reference_mean","foreground_reference_sd","reference_source_count"]}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
import sys,argparse
import numpy as np
import pandas as pd
root=CONFIG["paths"]["root"]
sys.path.insert(0,str(root/"software"))
from source_assignment_unified import (observations,_role_geometry,posterior_clock_role_features,
    general_background_reference_features,general_reference_role_features,fit_general_source_roles,foreground_reference_log_prior,
    fit_latent_representation_density,predict_latent_representation_density,source_reference_weights,period_reference_mass,
    quality_design,_fit_score)
from source_assignment_core import generic_coherent_fwl_power

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset",choices=["native","adfecg"],default="native")
parser.add_argument("--stage",choices=["all","background","background-fit","features","source","fit","replay"],default="all")
parser.add_argument("--calibrations",type=Path,help="Explicit archive of lawful fitted reference calibrations; --source-stage reuse validates its source/split/design metadata")
parser.add_argument("--calibration-recipe",type=Path,help="JSON calibration recipes learned on the supplied new No/reference sources")
parser.add_argument("--research-conditional",action="store_true",help="Run the retained historical conditional-mixture research pipeline")
parser.add_argument("--source-stage",choices=["fit","reuse"],default="fit",
    help="Fit reference calibration recipes, or explicitly reuse parameters after source/split/design validation")
parser.add_argument("--input",type=Path,help="Parquet: trace_id,run,condition,x,signal_y")
parser.add_argument("--references",type=Path,help="CSV with one row per observed trace")
parser.add_argument("--reference-column",default="reference_period")
parser.add_argument("--qualification-column",default="qualified")
parser.add_argument("--reference-source-column",help="Physical source ID shared by its observed measurements")
parser.add_argument("--background-reference-column",default="background_period")
parser.add_argument("--split-column",default="split_id")
parser.add_argument("--period-min",type=float)
parser.add_argument("--period-max",type=float)
parser.add_argument("--time-unit")
parser.add_argument("--candidates",type=int,default=73)
parser.add_argument("--name",help="Output/cache prefix; custom inputs default to custom")
parser.add_argument("--output-dir",type=Path)
parser.add_argument("--reference-views",type=Path)
parser.add_argument("--view-context",type=Path)
parser.add_argument("--source-components",type=Path,help="Factorized measured-source calibration components")
parser.add_argument("--reference-waveforms",type=Path,help="Measured foreground Parquet: source ID, x, foreground_y")
parser.add_argument("--reference-landmarks",type=Path,help="Training landmarks CSV: source ID, time in the No physical coordinate")
parser.add_argument("--reference-intervals",type=Path,help="Optional boundary-owned intervals CSV: source ID, interval")
parser.add_argument("--reference-annotations",type=Path,help="EDF+ ZIP with QRS annotations; explicit window mapping uses --record-metadata")
parser.add_argument("--record-metadata",type=Path,help="CSV: independent-unit column and source_file for EDF+ annotations")
parser.add_argument("--window-duration",type=float,help="EDF+ window duration in --time-unit; window indices are read from references")
parser.add_argument("--source-grouping",choices=["run","trace_id"],default="run",help="Measurements sharing one reference landmark stream")

parser.add_argument("--refresh-balanced",action="store_true",help="Rebuild the balanced source features while reusing already-built matching raw features")
parser.add_argument("--positive-views",type=Path,help="Additional waveform views paired with their union context")
parser.add_argument("--positive-context",type=Path)
parser.add_argument("--full-only",action="store_true")
parser.add_argument("--case-indices",type=int,nargs="+")
parser.add_argument("--model",type=Path,help="Output model, or model to read for replay")
parser.add_argument("--resume",type=Path,help="Explicit conditional-mixture checkpoint to continue")
parser.add_argument("--maxiter",type=int,default=CONFIG["maxiter"])
parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
args=parser.parse_args()
if args.research_conditional:
    CONFIG.update(readout_kind="conditional_latent_representation_density",model_kind="conditional_latent_representation_density",
        source_operator="observed_and_amplitude_balanced_probability_mixture",
        source_core_policy="reference_encoders_joint_individual_cohort_CF")
if (args.reference_views is None)!=(args.view_context is None):
    parser.error("--reference-views and --view-context are supplied together")
if (args.positive_views is None)!=(args.positive_context is None):
    parser.error("--positive-views and --positive-context are supplied together")
if args.input is not None and args.positive_views is not None and args.reference_views is None:
    parser.error("--positive-views for a custom input requires base --reference-views and --view-context")
if args.full_only and args.case_indices is not None:
    parser.error("--full-only and --case-indices select the same cases; supply one")
if args.resume is not None and args.stage not in {"all","fit"}:
    parser.error("--resume is used by --stage all or fit")
if args.source_components is not None and args.stage not in {"background","background-fit","features"}:
    parser.error("Factorized-source training requires the matching source-emission fitter; prepare features before fitting")
if args.refresh_balanced and (args.stage!="features" or args.source_components is None):
    parser.error("--refresh-balanced is used with --stage features and --source-components")
datasets={"native":{"input":"data/sota_dev/recomputed_clean/lemna_task_period_acf_input.parquet",
    "context":"data/sota_dev/recomputed_clean/lemna_task_period_supervised_context.npz","bounds":(12.,48.),"unit":"h"},
    "adfecg":{"input":"data/external/adfecgdb/inputs.parquet",
    "context":"data/external/adfecgdb/supervised_context.npz","bounds":(.25,1.5),"unit":"s",
    "reference_payload":{"references":"data/external/adfecgdb/references.csv","waveforms":"data/external/adfecgdb/foreground_reference.parquet",
        "annotations":"data/external/adfecgdb/adfecgdb-1.0.0.zip","record_metadata":"data/external/adfecgdb/record_metadata.csv",
        "source_metadata":"data/external/adfecgdb/source.json","reference_column":"reference_period_s","source_column":"run","unit_column":"subject"}}}
settings=datasets[args.dataset].copy()
reference_payload=settings.get("reference_payload") if args.input is None else None
if reference_payload is not None and not args.research_conditional:
    import json
    args.references=args.references or Path(reference_payload["references"])
    args.reference_waveforms=args.reference_waveforms or Path(reference_payload["waveforms"])
    if args.reference_landmarks is None and args.reference_annotations is None:
        args.reference_annotations=Path(reference_payload["annotations"])
    args.record_metadata=args.record_metadata or Path(reference_payload["record_metadata"])
    source_metadata=json.loads((root/reference_payload["source_metadata"]).read_text(encoding="utf-8-sig"))
    args.window_duration=args.window_duration or float(source_metadata["window_seconds"])
    args.reference_column=reference_payload["reference_column"] if args.reference_column=="reference_period" else args.reference_column
    args.reference_source_column=args.reference_source_column or reference_payload["source_column"]
    args.split_column=reference_payload["unit_column"] if args.split_column=="split_id" else args.split_column
    args.period_min=settings["bounds"][0] if args.period_min is None else args.period_min
    args.period_max=settings["bounds"][1] if args.period_max is None else args.period_max
    args.time_unit=args.time_unit or settings["unit"]
folder=root/(args.output_dir or CONFIG["paths"]["output"])
folder.mkdir(parents=True,exist_ok=True)
tag=args.name or ("custom" if args.input is not None else args.dataset)
assert Path(tag).name==tag
reference_folder=root/CONFIG["paths"]["reference_inputs"]
shared_payload=args.reference_landmarks is not None or args.reference_annotations is not None
shared_replay=False
if args.stage=="replay" and args.model is not None:
    with np.load(root/args.model,allow_pickle=False) as stored:
        shared_replay=str(stored.get("measurement_kind",""))=="shared_landmark_renewal"
if shared_payload or shared_replay:
    from source_assignment_distribution import (shared_landmark_observations,fit_shared_landmark_renewal,
        save_shared_landmark_renewal_model,load_shared_landmark_renewal_model,predict_shared_landmark_renewal)
    import json
    if args.stage not in {"all","features","source","fit","replay"}:
        parser.error("Shared landmark payload uses --stage all/features/source/fit/replay")
    input_path=root/(args.input or settings["input"])
    unit=args.time_unit or settings["unit"]
    frame=pd.read_parquet(input_path,columns=["trace_id","run","condition","x","signal_y"])
    data=shared_landmark_observations(frame,source_grouping=args.source_grouping,time_unit=unit)
    model_path=root/args.model if args.model is not None else folder/(tag+"_model.npz")
    if shared_replay:
        model=load_shared_landmark_renewal_model(model_path)
        selected=np.asarray(args.case_indices,dtype=int) if args.case_indices is not None else model["case_index"]
        selected=selected[:1] if args.full_only else selected
        result=predict_shared_landmark_renewal(data,model,case_indices=selected)
        source_order=pd.Index(model["source_id"]).get_indexer(result["source_id"])
        positions=pd.Index(model["case_index"]).get_indexer(selected)
        difference=float(np.max(abs(result["source_period"]-model["source_period"][positions][:,source_order])))
        print("SHARED_SOURCE_REPLAY",len(result["source_id"]),len(data["signal"]),selected.tolist(),difference,flush=True)
        raise SystemExit(0)
    if args.references is None or args.reference_waveforms is None:
        parser.error("Landmark fitting requires --references and --reference-waveforms")
    if (args.reference_landmarks is None)==(args.reference_annotations is None):
        parser.error("Supply one of --reference-landmarks or --reference-annotations")
    if args.period_min is None or args.period_max is None or not args.time_unit:
        parser.error("Landmark fitting requires explicit --period-min/--period-max/--time-unit")
    period=np.linspace(args.period_min,args.period_max,args.candidates)
    assert 0.<period[0]<period[-1] and len(period)>=3
    references=pd.read_csv(root/args.references,encoding="utf-8-sig")
    source_column=args.reference_source_column or args.source_grouping
    qualified=references[args.qualification_column].isin([True,1,"true","True"])
    references=references.assign(_source_id=references[source_column].astype(str),_qualified=qualified)
    fields=[args.reference_column,"_qualified",args.split_column]
    assert references.groupby("_source_id")[fields].nunique(dropna=False).le(1).all().all()
    bysource=references.drop_duplicates("_source_id").set_index("_source_id").loc[data["source_names"]]
    reference=bysource[args.reference_column].to_numpy(dtype=float)
    qualified=bysource["_qualified"].to_numpy(dtype=bool)&np.isfinite(reference)
    independent=bysource[args.split_column].to_numpy(dtype=str)
    units=np.unique(independent[qualified])
    heldout=units if len(units)>1 else units[:0]
    training=np.vstack([np.ones(len(reference),dtype=bool),independent[None]!=heldout[:,None]])
    selected=np.asarray(args.case_indices,dtype=int) if args.case_indices is not None else np.arange(len(training))
    selected=selected[:1] if args.full_only else selected
    waveform=pd.read_parquet(root/args.reference_waveforms)
    foreground=waveform.pivot(index=source_column,columns="x",values="foreground_y").reindex(index=data["source_names"],columns=data["time"]).to_numpy(dtype=float)
    if args.reference_landmarks is not None:
        landmarks=pd.read_csv(root/args.reference_landmarks,encoding="utf-8-sig")
        groups=landmarks.groupby(source_column,sort=False)["time"]
        event_by_source={str(key):np.sort(value.to_numpy(dtype=float)) for key,value in groups}
        events=list(map(lambda source:event_by_source.get(str(source),np.asarray([],dtype=float)),data["source_names"]))
        if args.reference_intervals is not None:
            intervals=pd.read_csv(root/args.reference_intervals,encoding="utf-8-sig")
            interval_groups=intervals.groupby(source_column,sort=False)["interval"]
            interval_by_source={str(key):value.to_numpy(dtype=float) for key,value in interval_groups}
            rr=list(map(lambda source:interval_by_source.get(str(source),np.asarray([],dtype=float)),data["source_names"]))
        else:
            rr=list(map(np.diff,events))
        annotation_policy="explicit_reference_landmark_table_same_No_time_coordinate"
    else:
        import zipfile,re
        if args.record_metadata is None or args.window_duration is None:
            parser.error("EDF+ annotations require --record-metadata and --window-duration")
        assert args.reference_intervals is None and np.isclose(data["time"][0],0.)
        recording=pd.read_csv(root/args.record_metadata,encoding="utf-8-sig").set_index(args.split_column)
        duration=float(args.window_duration);seconds={"h":3600.,"min":60.,"s":1.,"ms":.001}
        with zipfile.ZipFile(root/args.reference_annotations) as archive:
            prefix=archive.namelist()[0].split("/")[0]+"/"
            def read_annotation(name):
                content=archive.read(prefix+str(recording.loc[name,"source_file"]))
                header=int(content[184:192]);channels=int(content[252:256]);records=int(content[236:244])
                samples=np.frombuffer(content[256+216*channels:256+224*channels],dtype="S8").astype(int)
                digital=np.frombuffer(content[header:],dtype="<i2").reshape(records,samples.sum())
                annotation=digital[:,-samples[-1]:].astype("<i2").tobytes()
                return np.asarray(re.findall(rb"([+-]\d+(?:\.\d*)?)\x14QRS\x14\x00",annotation),dtype=float)/seconds[unit]
            annotation=dict(zip(np.unique(independent),map(read_annotation,np.unique(independent))))
        def source_annotation(index):
            times=annotation[independent[index]];window=int(bysource.window.iloc[index]);start=window*duration
            events=times[(times>=start)&(times<start+duration)]-start
            interval=np.diff(times)[np.floor(((times[:-1]+times[1:])/2.)/duration).astype(int)==window]
            return events,interval
        annotation_pairs=list(map(source_annotation,np.arange(len(reference))))
        events=list(map(lambda value:value[0],annotation_pairs));rr=list(map(lambda value:value[1],annotation_pairs))
        annotation_policy="EDFplus_QRS; local_window_x0; intervals_owned_by_midpoint_including_cross_boundary"
    event_count=np.asarray(list(map(len,events)));interval_count=np.asarray(list(map(len,rr)))
    assert np.all(event_count[qualified]>=2) and np.all(interval_count[qualified]>0)
    mean_rr=np.asarray(list(map(lambda x:float(np.mean(x)) if len(x) else np.nan,rr)))
    np.testing.assert_allclose(mean_rr[qualified],reference[qualified],rtol=0.,atol=max(1e-12,(period[-1]-period[0])*1e-10))
    event_indptr=np.r_[0,np.cumsum(event_count)];interval_indptr=np.r_[0,np.cumsum(interval_count)]
    payload={"source_id":data["source_names"],"reference":reference,"qualified":qualified,"independent_unit":independent,
        "training_source":training,"case_index":selected,"reference_waveform":foreground,"time":data["time"],"period":period,
        "landmark_time":np.concatenate(events),"landmark_indptr":event_indptr,
        "reference_interval":np.concatenate(rr),"reference_interval_indptr":interval_indptr,
        "source_index":data["source_index"],"trace_id":data["metadata"].index.to_numpy(dtype=str),"time_unit":np.asarray(unit),
        "input_path":np.asarray(str(input_path)),"references_path":np.asarray(str(root/args.references)),
        "reference_waveforms_path":np.asarray(str(root/args.reference_waveforms)),"landmark_policy":np.asarray(annotation_policy),
        "source_grouping":np.asarray(args.source_grouping),"reference_information":np.asarray("training_period_waveform_and_landmarks; No_only_query")}
    payload_path=folder/(tag+"_shared_source_training_payload.npz")
    np.savez_compressed(payload_path,**payload)
    count=(training&qualified[None]).sum(1)
    groups=(np.asarray([np.unique(independent[row]).size for row in training&qualified[None]]))
    print("SHARED_SOURCE_INPUT",len(data["signal"]),len(reference),int(qualified.sum()),len(units),count.tolist(),groups.tolist(),str(payload_path),flush=True)
    if args.stage=="features":
        raise SystemExit(0)
    detector_path=model_path.with_name(model_path.stem+"_detector.joblib")
    if args.source_stage=="reuse":
        if args.calibrations is None:
            parser.error("Shared-source reuse requires explicit --calibrations")
        model=load_shared_landmark_renewal_model(root/args.calibrations)
        assert str(model["time_unit"])==unit
        np.testing.assert_array_equal(model["source_id"],data["source_names"])
        np.testing.assert_allclose(model["period"],period,rtol=0.,atol=0.)
        positions=pd.Index(model["case_index"]).get_indexer(selected);assert np.all(positions>=0)
        np.testing.assert_array_equal(model["training_source"][positions],training[selected])
        np.testing.assert_array_equal(model["qualified"],qualified)
        np.testing.assert_allclose(model["reference"],reference,rtol=0.,atol=0.,equal_nan=True)
        np.testing.assert_array_equal(model["training_independent_unit"],independent)
        case_fields={"case_index","learned_temporal_lags","learned_QRS_width","foreground_event_template","positive_event_tolerance",
            "class_positive_prior","class_weight_positive","class_weight_negative","training_timestamp_count","train_reference_source_count",
            "train_reference_unit_count","reference_source_count","training_source","training_reference_access_mask","training_source_weight","renewal_RR_variance",
            "renewal_timing_variance","renewal_detector_precision","renewal_detector_recall","foreground_log_prior","foreground_reference_mean",
            "foreground_reference_sd","renewal_temperature","fit_seconds","source_probability","source_period","source_log_likelihood",
            "interval_mean","interval_median","interval_ridge"}
        model={key:value[positions] if key in case_fields else value for key,value in model.items()}
        model["calibration_import_path"]=np.asarray(str(root/args.calibrations))
    else:
        if args.calibrations is not None:
            parser.error("--calibrations is explicit reuse; omit it for new source learning")
        model=fit_shared_landmark_renewal(data,reference,qualified,training,independent,foreground,events,period,
            reference_intervals=rr,time_unit=unit,detector_path=detector_path,case_indices=selected)
    model.update({"input_path":payload["input_path"],"references_path":payload["references_path"],
        "reference_waveforms_path":payload["reference_waveforms_path"],"landmark_policy":payload["landmark_policy"],
        "training_payload_path":np.asarray(str(payload_path)),"source_refs_reference_source_id":data["source_names"],
        "source_refs_independent_unit":independent,"source_refs_qualified":qualified,"source_refs_reference":reference,
        "source_refs_training":training[selected],"reference":reference,"qualified":qualified,
        "methods":np.asarray(["unified"]),"method_indices":np.asarray([0]),"preferred_method":np.asarray(0)})
    save_shared_landmark_renewal_model(model,model_path)
    print("SHARED_SOURCE_MODEL",selected.tolist(),model["train_reference_source_count"].tolist(),str(model_path),flush=True)
    raise SystemExit(0)

if not args.research_conditional and args.source_components is None and args.stage in {"all","features","source","fit","replay"}:
    from source_assignment_unified import (reference_source_observations,predict_reference_calibration_mixture,
        fit_reference_calibration_density,observed_source_information)
    import json
    input_path=root/(args.input or settings["input"])
    frame=pd.read_parquet(input_path)
    data=observations(frame);identity=data["metadata"].index.to_numpy(dtype=str)
    metadata=data["metadata"];unit=args.time_unit or settings["unit"]
    model_path=root/args.model if args.model is not None else folder/("unified_model.npz" if args.input is None and args.dataset=="native" else tag+"_model.npz")
    if args.stage=="replay":
        if args.model is None:
            parser.error("Reference-calibration replay requires explicit --model")
        model=dict(np.load(model_path,allow_pickle=False))
        assert str(model["readout_kind"])=="information_conditioned_reference_W2"
        assert str(model["time_unit"])==unit
        selected=np.asarray(args.case_indices,dtype=int) if args.case_indices is not None else model["case_index"]
        selected=selected[:1] if args.full_only else selected
        rows=pd.Index(model["case_index"]).get_indexer(selected);assert np.all(rows>=0)
        result=predict_reference_calibration_mixture(frame,model,cases=rows,device=args.device,include_diagnostics=True)
        source_order=pd.Index(model["trace_id"]).get_indexer(identity);assert np.all(source_order>=0)
        difference={key:float(np.max(abs(result[key]-model[key][rows][:,source_order]))) for key in ["estimate","posterior"]}
        np.testing.assert_allclose(result["posterior"].sum(-1),1.,rtol=0.,atol=1e-12)
        print("REFERENCE_CALIBRATION_REPLAY",selected.tolist(),difference,flush=True)
        raise SystemExit(0)
    if args.input is not None:
        if args.references is None or args.period_min is None or args.period_max is None or not args.time_unit:
            parser.error("New input fitting requires --references/--period-min/--period-max/--time-unit")
        references=pd.read_csv(root/args.references,encoding="utf-8-sig",float_precision="round_trip").set_index("trace_id").loc[identity]
        reference=references[args.reference_column].to_numpy(dtype=float)
        qualified=references[args.qualification_column].isin([True,1,"true","True"]).to_numpy()&np.isfinite(reference)
        independent=references[args.split_column].to_numpy(dtype=str)
        source_column=args.reference_source_column or ("reference_source_id" if "reference_source_id" in references else None)
        source_id=references[source_column].to_numpy(dtype=str) if source_column is not None else identity
        group=pd.DataFrame({"run":metadata.run.to_numpy(dtype=str),"unit":independent}).drop_duplicates().set_index("run")
        assert group.index.is_unique
        group_unit=group.loc[data["runs"],"unit"].to_numpy(dtype=str)
        units=np.unique(independent[qualified]);heldout=units if len(units)>1 else units[:0]
        training=np.vstack([np.ones(len(data["runs"]),dtype=bool),group_unit[None]!=heldout[:,None]])
        query=np.vstack([np.zeros(len(identity),dtype=bool),independent[None]==heldout[:,None]])
        background_reference=references[args.background_reference_column].to_numpy(dtype=float) if args.background_reference_column in references else None
        bounds=np.asarray([args.period_min,args.period_max],dtype=float)
        reference_path=root/args.references
    else:
        context=dict(np.load(root/settings["context"],allow_pickle=False))
        order=pd.Index(context["trace_id"]).get_indexer(identity);assert np.all(order>=0)
        reference=context["target"][order];qualified=context["qualified"][order]
        independent=metadata.run.to_numpy(dtype=str) if "independent_unit" not in context else context["independent_unit"][order].astype(str)
        source_id=identity if "reference_source_id" not in context else context["reference_source_id"][order].astype(str)
        group_order=pd.Index(context.get("run_names",context.get("run_name"))).get_indexer(data["runs"])
        query=np.vstack([np.zeros(len(identity),dtype=bool),context["outer_query"][:,order]])
        training=((~query).astype(float)@data["membership"].T)>0.
        background_reference=None;bounds=np.asarray(settings["bounds"] if args.period_min is None else [args.period_min,args.period_max],dtype=float)
        reference_path=root/settings["context"]
    period=np.linspace(*bounds,args.candidates)
    selected=np.asarray(args.case_indices,dtype=int) if args.case_indices is not None else np.arange(len(training))
    selected=selected[:1] if args.full_only else selected
    source_member=(np.unique(source_id)[:,None]==source_id[None]).astype(float)
    train_rows=training[:,data["run_index"]]&qualified[None]
    source_count=((train_rows.astype(float)@source_member.T)>0.).sum(1)
    payload={"trace_id":identity,"run_names":data["runs"],"source_refs_reference_source_id":source_id,
        "source_refs_independent_unit":independent,"source_refs_reference":reference,"source_refs_qualified":qualified,
        "training_run":training,"source_refs_query_mask":query,"period":period,"time_unit":np.asarray(unit)}
    np.savez_compressed(folder/(tag+"_reference_calibration_training_payload.npz"),**payload)
    print("REFERENCE_CALIBRATION_INPUT",len(identity),int(qualified.sum()),source_count[selected].tolist(),selected.tolist(),flush=True)
    if args.stage=="features":
        raise SystemExit(0)
    if args.source_stage=="reuse":
        if args.calibrations is None:
            parser.error("--source-stage reuse requires explicit --calibrations")
        if args.calibration_recipe is not None:
            parser.error("Reuse reads the archived calibration recipe; omit --calibration-recipe")
        model=dict(np.load(root/args.calibrations,allow_pickle=False))
        assert str(model["readout_kind"])=="information_conditioned_reference_W2"
        assert str(model["time_unit"])==unit
        np.testing.assert_array_equal(model["trace_id"],identity)
        np.testing.assert_allclose(model["period"],period,rtol=0.,atol=0.)
        np.testing.assert_array_equal(model["source_refs_reference_source_id"],source_id)
        np.testing.assert_array_equal(model["source_refs_independent_unit"],independent)
        np.testing.assert_array_equal(model["source_refs_qualified"],qualified)
        np.testing.assert_allclose(model["source_refs_reference"],reference,rtol=0.,atol=0.,equal_nan=True)
        rows=pd.Index(model["case_index"]).get_indexer(selected);assert np.all(rows>=0)
        np.testing.assert_array_equal(model["training_run"][rows],training[selected])
        np.testing.assert_array_equal(model["reference_source_count"][rows],source_count[selected])
        np.testing.assert_array_equal(model["source_refs_query_mask"][rows],query[selected])
        assert not np.any(model["reference_supervision_mask"][rows]&query[selected,None])
        if "background_view_supervision_mask" in model:
            np.testing.assert_array_equal(model["background_origin_trace_id"],identity)
            np.testing.assert_array_equal(model["background_reference_independent_unit"],independent)
            background_mask=model["background_view_supervision_mask"][rows]
            overlap=(background_mask&query[selected,None,None]).sum(-1)
            assert not np.any(overlap),"Archived BG supervision overlaps the requested query records"
            unit_code,units=pd.factorize(independent,sort=True)
            unit_member=(np.arange(len(units))[:,None]==unit_code[None]).astype(float)
            query_units=(query[selected].astype(float)@unit_member.T)>0.
            background_units=(background_mask.astype(float)@unit_member.T)>0.
            assert not np.any(background_units&query_units[:,None,None]),"Archived BG supervision overlaps query units"
            np.testing.assert_array_equal(model["background_view_reference_count"][rows],background_mask.sum(-1))
            np.testing.assert_array_equal(model["validation_BG_query_record_overlap_count"][rows],overlap)
            print("REFERENCE_CALIBRATION_BG_REUSE",selected.tolist(),"query_overlap",int(overlap.max()),flush=True)
        geometry=_role_geometry(data["signal"],data,period,float(model["period_scale"]),normalize_reference_peer=True)
        information=observed_source_information({"period":period,"joint_gram":geometry["joint"],"layout":geometry["layout"]})
        np.testing.assert_allclose(model["source_information"],information,rtol=1e-12,atol=1e-10)
        for position in rows:
            allowed=model["reference_supervision_mask"][position].any(0)
            endpoint=np.asarray([information[allowed].min(),information[allowed].max()])
            expected=endpoint[:1] if model["reference_information_knots"].shape[1]==1 else endpoint
            np.testing.assert_allclose(model["reference_information_knots"][position],expected,rtol=1e-12,atol=1e-10)
        case_fields={"case_index","success","reference_information_knots","reference_source_count","effective_training_units",
            "reference_supervision_mask","reference_supervision_weight","training_run","source_refs_query_mask","outer_protocol",
            "calibration_parameter_provenance","posterior","estimate","source_period","group_mean","deviation",
            "cohort_probability","reference_calibration_weight","reference_calibration_probability","reference_calibration_mean",
            "source_assignment_probability","raw_individual_probability","calibrated_mean_target"}
        case_fields.update({"background_manifest_case_index","background_reference_source_count","background_reference_unit_count",
            "background_reference_supervision_mask","background_native_supervision_mask","background_view_supervision_mask",
            "background_view_reference_count","background_view_weight","background_supervision_union_source_count",
            "background_native_regularization_source_count","background_calibration_parameter_provenance",
            "background_calibration_parameter_row_index","validation_BG_query_record_overlap_count",
            "validation_BG_query_unit_overlap_count","validation_BG_treatment_condition_checked",
            "validation_BG_treatment_condition_overlap_count"})
        case_fields.update(key for key,value in model.items() if key.startswith("calibration_") and np.ndim(value)>0 and key not in {
            "calibration_name","calibration_recipe_json","calibration_representation_options_json","calibration_fundamental_support",
            "calibration_source_features","calibration_measurement_kind"})
        model={key:value[rows] if key in case_fields else value for key,value in model.items()}
        model["calibration_import_path"]=np.asarray(str(root/args.calibrations))
        result=predict_reference_calibration_mixture(frame,model,device=args.device,include_diagnostics=True)
        model.update(result)
    else:
        if args.calibrations is not None:
            parser.error("--calibrations is explicit reuse; new calibration learning uses --calibration-recipe")
        recipe=json.loads((root/args.calibration_recipe).read_text(encoding="utf-8-sig")) if args.calibration_recipe is not None else None
        model=fit_reference_calibration_density(frame,reference,qualified,training,independent,period,
            reference_source_id=source_id,background_reference=background_reference,calibration_recipe=recipe,
            case_indices=selected,maxiter=args.maxiter,device=args.device,time_unit=unit)
        model["source_refs_query_mask"]=query[selected]
    model.update({"input_path":np.asarray(str(input_path)),"references_path":np.asarray(str(reference_path)),
        "training_information":np.asarray(str(model.get("background_training_information",
            model.get("training_information","declared_No_period_references_source_IDs_and_independent_unit_splits")))),
        "model_role":np.asarray("recommended_reference_calibrated_source_period_model")})
    assert not any("fit_length" in key for key in model)
    temporary=model_path.with_suffix(".pending.npz");np.savez_compressed(temporary,**model);temporary.replace(model_path)
    print("REFERENCE_CALIBRATION_MODEL",selected.tolist(),model["reference_source_count"].tolist(),str(model_path),flush=True)
    raise SystemExit(0)
if args.stage=="background-fit":
    target_path=folder/(tag+"_source_background_features.npz")
    background=dict(np.load(target_path,allow_pickle=False))
    selected=np.asarray(args.case_indices,dtype=int) if args.case_indices is not None else np.arange(len(background["training_run"]))
    selected=selected[:1] if args.full_only else selected
    assert np.all((selected>=0)&(selected<len(background["training_run"])))
    parent=dict(np.load(folder/(tag+"_observed_source_density_reference_model.npz"),allow_pickle=False))
    np.testing.assert_array_equal(parent["period"],background["period"])
    np.testing.assert_array_equal(parent["training_run"],background["training_run"])
    left,right=np.triu_indices(13)
    columns=np.r_[np.isin(np.arange(13),[4,5,6,7,8]),np.isin(left,[4,5,6,7,8])&np.isin(right,[4,5,6,7,8])]
    columns=np.tile(columns,2)
    design=quality_design(background["features"],background["mixing_quality"])
    training_weight=background["training_weight"][selected]
    original_weight=training_weight*background["is_original"][None]
    original_weight/=original_weight.sum(1)[:,None]
    fits=list(map(lambda position:_fit_score(design,background["background_reference"],background["qualified"],
        training_weight[position:position+1]>0.,background["period"],maxiter=args.maxiter,
        training_weight=training_weight[position:position+1],
        effective_sample_size=background["background_reference_count"][selected[position:position+1]],
        prior_scale_weight=original_weight[position:position+1],
        initial_coefficient=parent["background_coefficient"][selected[position:position+1]][:,columns],
        quality=background["mixing_quality"],interaction_units=background["effective_training_units"][selected[position:position+1]]),
        np.arange(len(selected))))
    coefficient=np.zeros((len(selected),len(columns)))
    coefficient[:,columns]=np.concatenate(list(map(lambda value:value[0],fits)))
    background.update({"background_fit_case_index":selected,"background_coefficient":coefficient,
        "background_fit_success":np.asarray(list(map(lambda value:value[1]["success"],fits))),
        "background_fit_iterations":np.asarray(list(map(lambda value:value[1]["iterations"],fits))),
        "background_fit_gradient":np.asarray(list(map(lambda value:value[1]["gradient"],fits))),
        "background_fit_policy":np.asarray("all_measured_source_views_equal_unit_origin_view_source_weighted_proper_CE"),
        "background_conditioning_prior":np.asarray("distinct_BG_source_N_and_biological_unit_G"),
        "background_density_role":np.asarray("conditional_reference_density_not_a_query_independent_physical_prior")})
    temporary=target_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,**background)
    temporary.replace(target_path)
    print("SOURCE_BACKGROUND_FIT",tag,selected.tolist(),background["background_fit_success"].tolist(),
        background["background_fit_iterations"].tolist(),str(target_path),flush=True)
    raise SystemExit(0)
if args.stage=="background":
    component_path=root/(args.source_components or folder/(tag+"_positive_training_source_components.npz"))
    view_path=root/(args.reference_views or folder/(tag+"_positive_training_views.parquet"))
    context_path=root/(args.view_context or folder/(tag+"_positive_training_context.npz"))
    components=dict(np.load(component_path,allow_pickle=False))
    view_context=dict(np.load(context_path,allow_pickle=False))
    bounds=view_context["period_bounds"]
    assert args.period_min is None or np.isclose(args.period_min,bounds[0])
    assert args.period_max is None or np.isclose(args.period_max,bounds[1])
    assert args.time_unit is None or args.time_unit==str(view_context["unit"])
    period=np.linspace(float(bounds[0]),float(bounds[1]),args.candidates)
    np.testing.assert_array_equal(components["candidate_view_id"],view_context["candidate_view_id"])
    np.testing.assert_array_equal(components["trace_id"],view_context["candidate_origin_trace_id"])
    assert str(components["unit"])==str(view_context["unit"])
    tags=np.unique(view_context["view_id"])
    background=general_background_reference_features(view_path,view_context,period,tags)
    context_order=pd.Index(view_context["trace_id"]).get_indexer(background["trace_id"])
    original_order=pd.Index(components["trace_id"]).get_indexer(background["origin_trace_id"])
    view_order=pd.Index(components["candidate_view_id"]).get_indexer(background["view_id"])
    assert np.all(context_order>=0) and np.all(original_order>=0) and np.all(view_order>=0)
    b=components["background"];common=components["common_background"];mask=components["mask"]
    own=np.sum(b*b*mask,axis=1)/components["count"]
    shared=np.sum(common*common*mask,axis=1)/components["count"]
    cross=np.sum(b*common*mask,axis=1)/components["count"]
    gain=components["background_gain"];common_gain=components["common_background_gain"]
    power=gain**2*own[None]+common_gain**2*shared[None]+2*gain*common_gain*cross[None]
    background["foreground_qualified"]=background["qualified"].copy()
    background["qualified"]=np.isfinite(background["background_reference"])&(power[view_order,original_order]>np.finfo(float).eps)
    background["origin_index"]=original_order
    background["independent_unit"]=components["independent_unit"][original_order]
    background["view_weight"]=view_context["view_weight"][context_order]
    background["background_reference_kind"]=view_context["background_reference_kind"][context_order]
    background["background_source_id"]=np.char.add("BG:",components["reference_source_id"][original_order].astype(str))
    run_code,run_names=pd.factorize(components["run"],sort=True)
    run_order=pd.Index(view_context["run_name"]).get_indexer(run_names)
    assert np.all(run_order>=0)
    training=np.concatenate([np.ones((1,len(run_names)),dtype=bool),view_context["outer_train_run"][:,run_order]])
    original_rows=np.flatnonzero(background["is_original"])
    aligned=pd.Index(background["origin_trace_id"][original_rows]).get_indexer(components["trace_id"])
    assert np.all(aligned>=0)
    original_rows=original_rows[aligned]
    source_code,source_names=pd.factorize(components["reference_source_id"],sort=True)
    source_member=(np.arange(len(source_names))[:,None]==source_code[None]).astype(float)
    bg_qualified=background["qualified"][original_rows]
    source_number=source_member@bg_qualified.astype(float)
    source_period=np.divide(source_member@np.where(bg_qualified,background["background_reference"][original_rows],0.),source_number,
        out=np.zeros(len(source_names)),where=source_number>0.)
    unit_code,unit_names=pd.factorize(components["independent_unit"],sort=True)
    unit_member=(np.arange(len(unit_names))[:,None]==unit_code[None]).astype(float)
    source_unit=(source_member@unit_member.T)>0.
    assert np.all(source_unit.sum(1)==1)
    source_unit_code=source_unit.argmax(1)
    eligible=((training[:,run_code]&bg_qualified[None]).astype(float)@source_member.T)>0.
    per_unit=eligible.astype(float)@source_unit
    unit_count=(per_unit>0.).sum(1)
    source_weight=eligible/np.maximum(per_unit[:,source_unit_code],1.)/unit_count[:,None]
    source_mass=period_reference_mass(period,source_period,source_number>0.)
    weak_prior=(unit_count[:,None]*(source_weight@source_mass)+1./len(period))/(unit_count[:,None]+1.)
    row_groups=np.char.add(np.char.add(background["view_id"].astype(str),":"),background["origin_run"].astype(str))
    row_code,row_names=pd.factorize(row_groups,sort=True)
    row_member=(np.arange(len(row_names))[:,None]==row_code[None]).astype(float)
    row_origin=np.rint((row_member/row_member.sum(1)[:,None])@run_code[original_order]).astype(int)
    np.testing.assert_array_equal(row_origin[row_code],run_code[original_order])
    bg_bundle={"run_index":row_code,"membership":row_member,"average":row_member/row_member.sum(1)[:,None]}
    weights=source_reference_weights(bg_bundle,background,training[:,row_origin])
    background.update({"period":period,"period_scale":np.asarray(np.sqrt(period[0]*period[-1])),"unit":components["unit"],
        "origin_run_names":np.asarray(run_names,dtype=str),"training_run":training,"background_reference_count":eligible.sum(1),
        "effective_training_units":unit_count,"background_source_names":np.char.add("BG:",np.asarray(source_names,dtype=str)),
        "background_source_period":source_period,"background_source_qualified":source_number>0.,
        "background_source_training":eligible,"background_source_weight":source_weight,
        "weak_background_log_prior":np.log(weak_prior),"weak_prior_pseudo_units":np.asarray(1.),
        "training_weight":weights["training_weight"],"group_weight":weights["group_weight"],
        "view_run_names":np.asarray(row_names,dtype=str),"view_run_index":row_code,
        "background_is_proxy":np.ones(len(background["trace_id"]),dtype=bool),
        "background_reference_available":background["qualified"].copy(),
        "background_availability_policy":np.asarray("finite_in_bounds_H1_proxy_from_nonzero_known_BG_component_not_rhythm_QC"),
        "reference_policy":np.asarray("measured_or_estimated_BG_waveform_H1_proxy_all_declared_views"),
        "source_components":np.asarray(str(component_path))})
    assert np.allclose(weak_prior.sum(1),1.) and np.allclose(background["training_weight"].sum(1),1.)
    assert not np.any(training[1:,run_code]&view_context["origin_query"][:,pd.Index(view_context["trace_id"]).get_indexer(components["trace_id"])])
    target_path=folder/(tag+"_source_background_features.npz")
    temporary=target_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,**background)
    temporary.replace(target_path)
    print("SOURCE_BACKGROUND_READY",tag,len(background["trace_id"]),len(tags),eligible.sum(1).tolist(),unit_count.tolist(),str(target_path),flush=True)
    raise SystemExit(0)
frame=None
if args.input is not None:
    if args.references is None or args.period_min is None or args.period_max is None or not args.time_unit:
        parser.error("--input requires --references, --period-min, --period-max and --time-unit")
    settings.update(input=args.input,bounds=(args.period_min,args.period_max),unit=args.time_unit)
    frame=pd.read_parquet(root/args.input)
    metadata=frame[["trace_id","run","condition"]].drop_duplicates().sort_values("trace_id")
    references=pd.read_csv(root/args.references,encoding="utf-8-sig",float_precision="round_trip").set_index("trace_id")
    assert references.index.is_unique and metadata.trace_id.is_unique
    order=references.index.get_indexer(metadata.trace_id);assert np.all(order>=0)
    references=references.iloc[order]
    labels=references[args.reference_column].to_numpy(dtype=float)
    qualified=references[args.qualification_column].isin([True,1,"true","True"]).to_numpy()&np.isfinite(labels)
    split=references[args.split_column].to_numpy(dtype=str) if args.split_column in references else metadata.run.to_numpy(dtype=str)
    source_column=args.reference_source_column or ("reference_source_id" if "reference_source_id" in references else None)
    source_id=references[source_column].to_numpy(dtype=str) if source_column is not None else metadata.trace_id.to_numpy(dtype=str)
    groups=metadata.assign(split=split)[["run","split"]].drop_duplicates().sort_values("run")
    assert groups.run.is_unique
    eligible=metadata.assign(qualified=qualified).groupby("run").qualified.any().reindex(groups.run).to_numpy()
    units=np.unique(groups.split.to_numpy(dtype=str)[eligible])
    heldout=units if len(units)>1 else units[:0]
    context={"trace_id":metadata.trace_id.to_numpy(dtype=str),"run_name":groups.run.to_numpy(dtype=str),
        "target":labels,"qualified":qualified,"outer_query":split[None]==heldout[:,None],
        "outer_protocol":np.repeat("leave_"+args.split_column,len(heldout)),"outer_condition":heldout,
        "outer_control_run":np.repeat("",len(heldout)),"independent_unit":split,
        "reference_source_id":source_id,
        "background_reference":references[args.background_reference_column].to_numpy(dtype=float) if args.background_reference_column in references else np.full(len(labels),np.nan)}
else:
    if args.time_unit is not None and args.time_unit!=settings["unit"]:
        parser.error("Built-in timestamps use "+settings["unit"]+"; converted inputs use --input")
    if (args.period_min is None)!=(args.period_max is None):
        parser.error("--period-min and --period-max are supplied together")
    if args.period_min is not None:
        settings["bounds"]=(args.period_min,args.period_max)
    context=dict(np.load(root/settings["context"],allow_pickle=False))
assert 0<settings["bounds"][0]<settings["bounds"][1] and args.candidates>=3
period=np.linspace(*settings["bounds"],args.candidates)
feature_paths={"raw":folder/(tag+"_observed_features.npz"),"balanced":folder/(tag+"_features.npz")}
view_paths={"raw":folder/(tag+"_observed_reference_features.npz"),"balanced":folder/(tag+"_reference_features.npz")}
parent_paths={"raw":folder/(tag+"_observed_source_density_reference_model.npz"),
    "balanced":folder/(tag+"_source_density_reference_model.npz")}
default_model=tag+("_model.npz" if args.dataset=="adfecg" else "_representation_mixture_model.npz")
model_path=root/args.model if args.model is not None else folder/default_model
background_path=folder/(tag+"_background_features.npz")
has_views=args.input is None or args.reference_views is not None
wave_path=root/(args.reference_views or reference_folder/(args.dataset+"_training_views.parquet"))
wave_context_path=root/(args.view_context or reference_folder/(args.dataset+"_training_context.npz"))
factorized=args.source_components is not None
if factorized:
    wave_path=root/(args.reference_views or folder/(tag+"_positive_training_views.parquet"))
    wave_context_path=root/(args.view_context or folder/(tag+"_positive_training_context.npz"))
    background_path=folder/(tag+"_source_background_features.npz")
    view_paths={"raw":folder/(tag+"_source_observed_reference_features.npz"),"balanced":folder/(tag+"_source_reference_features.npz")}
has_positive=not factorized and (args.positive_views is not None or (args.input is None and args.dataset=="native"))
positive_path=root/(args.positive_views or root/CONFIG["paths"]["output"]/(args.dataset+"_positive_training_views.parquet"))
positive_context_path=root/(args.positive_context or root/CONFIG["paths"]["output"]/(args.dataset+"_positive_training_context.npz"))
positive_cache=folder/(tag+"_representation_mixture_positive_features.npz")
build_features=args.stage in {"all","features"}
background=None
source_context=dict(np.load(wave_context_path,allow_pickle=False)) if has_views else None
if has_views:
    np.testing.assert_allclose(source_context["period_bounds"],settings["bounds"])
    assert str(source_context["unit"])==settings["unit"]
if build_features:
    if frame is None:
        frame=pd.read_parquet(root/settings["input"])
    if has_views:
        if factorized:
            background=dict(np.load(background_path,allow_pickle=False))
            np.testing.assert_array_equal(background["period"],period)
            assert set(background["view_id"])==set(source_context["view_id"])
            assert set(background["trace_id"])==set(source_context["trace_id"])
        else:
            tags=np.unique(source_context["view_id"])
            tags=tags[~np.char.startswith(tags.astype(str),"phase")]
            background=general_background_reference_features(wave_path,source_context,period,tags)
        background["origin_index"]=pd.Index(np.sort(frame.trace_id.unique())).get_indexer(background["origin_trace_id"])
        assert np.all(background["origin_index"]>=0)
        if not factorized:
            np.savez_compressed(background_path,**background)
else:
    background=dict(np.load(background_path,allow_pickle=False)) if has_views else None

def representation(name):
    raw=name=="raw"
    if build_features and not (args.refresh_balanced and raw):
        measured=posterior_clock_role_features(frame,period_min=period.min(),period_max=period.max(),
            candidates=len(period),source_balancing=not raw,noise_cache=background if raw else None)
        fields=CONFIG["feature_fields"]+(["source_observation_energy","source_pair_scale","source_peer_weight_square"] if factorized else [])
        bundle={key:measured[key] for key in fields}
        if raw and background is not None:
            original=np.flatnonzero(background["is_original"])
            order=pd.Index(background["trace_id"][original]).get_indexer(bundle["trace_id"]);assert np.all(order>=0)
            bundle["mixing_quality"]=background["mixing_quality"][original[order]]
        np.savez_compressed(feature_paths[name],**bundle)
    else:
        bundle=dict(np.load(feature_paths[name],allow_pickle=False))
    np.testing.assert_array_equal(bundle["period"],period)
    expected="observed_physical_peer" if raw else "frequency_marginal_retained_peer"
    assert str(bundle["source_representation"])==expected
    return bundle

raw,balanced=map(representation,["raw","balanced"])
np.testing.assert_array_equal(raw["trace_id"],balanced["trace_id"])
np.testing.assert_array_equal(raw["run_names"],balanced["run_names"])
np.testing.assert_allclose(raw["mixing_quality"],balanced["mixing_quality"],rtol=0.,atol=1e-10)
balanced=dict(balanced,mixing_quality=raw["mixing_quality"])
trace_order=pd.Index(context["trace_id"]).get_indexer(raw["trace_id"])
run_order=pd.Index(context["run_name"]).get_indexer(raw["run_names"])
assert np.all(trace_order>=0) and np.all(run_order>=0)
reference=context["target"][trace_order];qualified=context["qualified"][trace_order]
assert np.all((reference[qualified]>=period.min())&(reference[qualified]<=period.max()))
query=context["outer_query"][:,trace_order]
training=np.concatenate([np.ones((1,len(raw["run_names"])),dtype=bool),
    ~((query.astype(float)@raw["membership"].T)>0.)])
assert not np.any(training[1:,raw["run_index"]]&query)
independent=context["independent_unit"][trace_order] if "independent_unit" in context else raw["run"]
source_id=context["reference_source_id"][trace_order] if "reference_source_id" in context else raw["trace_id"]
background_reference=context["background_reference"][trace_order] if "background_reference" in context else np.full(len(reference),np.nan)
if args.input is None and args.dataset=="native":
    bg=pd.read_csv(reference_folder/"native_background_references.csv",encoding="utf-8-sig").set_index("trace_id")
    background_reference=np.where(qualified,bg.loc[raw["trace_id"],"background_reference_h"].to_numpy(),np.nan)
    independent=raw["run"];source_id=raw["trace_id"]
if args.input is None and args.dataset=="adfecg":
    ids=pd.read_csv(root/"data/external/adfecgdb/references.csv",encoding="utf-8-sig").set_index("trace_id").loc[raw["trace_id"]]
    independent=ids.subject.to_numpy(dtype=str);source_id=raw["run"]
assert pd.DataFrame({"run":raw["run"],"unit":independent}).drop_duplicates().run.is_unique
source_code,source_names=pd.factorize(source_id,sort=True)
source_member=(np.arange(len(source_names))[:,None]==source_code[None]).astype(float)
source_count=((training[:,raw["run_index"]]&qualified[None]).astype(float)@source_member.T>0.).sum(1)
replay_model=dict(np.load(model_path,allow_pickle=False)) if args.stage=="replay" else None
selected=np.asarray([0]) if args.full_only else np.asarray(args.case_indices) if args.case_indices is not None else replay_model.get("case_index",np.arange(len(replay_model["gate_coefficient"]))) if replay_model is not None else np.arange(len(training))
assert len(np.unique(selected))==len(selected) and np.all((selected>=0)&(selected<len(training)))
if args.stage!="replay":
    np.savez_compressed(folder/(tag+"_source_context.npz"),trace_id=raw["trace_id"],run_names=raw["run_names"],
        reference=reference,qualified=qualified,reference_source_id=source_id,independent_unit=independent,
        training_run=training,reference_source_count=source_count,outer_query=query,period_bounds=np.asarray(settings["bounds"]),
        time_unit=np.asarray(settings["unit"]),input_path=np.asarray(str(root/settings["input"])),
        references_path=np.asarray(str(root/args.references)) if args.references is not None else np.asarray(str(root/settings["context"])))
print("SOURCE_INPUT",tag,len(reference),int(qualified.sum()),int(source_count[0]),len(raw["run_names"]),len(np.unique(independent)),selected.tolist(),flush=True)

def parent(name,bundle):
    path=parent_paths[name]
    if args.source_stage=="fit" and args.stage!="replay":
        data=observations(frame if frame is not None else pd.read_parquet(root/settings["input"]))
        geometry=_role_geometry(data["signal"],data,period,float(bundle["period_scale"]),
            peer_reference=data["signal"] if name=="raw" else None,normalize_reference_peer=name=="raw")
        np.testing.assert_array_equal(data["metadata"].index.to_numpy(),bundle["trace_id"])
        measured=dict(bundle,base_features=bundle["source_base"] if name=="raw" else bundle["identity_base_features"],
            original_unit=geometry["unit"],original_peer=geometry["peer"],original_rhs=geometry["rhs"],
            harmonic_design=geometry["design"],original_coherent_power=bundle["source_coherent_power"] if name=="raw" else
            generic_coherent_fwl_power(data["time"],data["mask"],geometry["unit"],geometry["peer"],period,half_gram=False))
        result=fit_general_source_roles(measured,reference,qualified,training[selected],background_reference,
            maxiter=args.maxiter,device=args.device,independent_unit=independent,background_views=background,
            conditioned=True,reference_source_id=source_id)
        result.pop("posterior",None)
        result.update({"trace_id":raw["trace_id"],"period":period,"training_run":training[selected],"case_index":selected,
            "reference":reference,"qualified":qualified,"independent_unit":independent,"reference_source_id":source_id,
            "time_unit":np.asarray(settings["unit"]),"period_bounds":np.asarray(settings["bounds"]),
            "source_stage":np.asarray("reference"),"source_order_policy":bundle["source_order_policy"]})
        np.savez_compressed(path,**result)
    else:
        result=dict(np.load(path,allow_pickle=False))
    np.testing.assert_array_equal(result["trace_id"],raw["trace_id"])
    np.testing.assert_array_equal(result["period"],period)
    parent_cases=result.get("case_index",np.arange(len(result["training_run"])))
    np.testing.assert_array_equal(result["training_run"],training[parent_cases])
    np.testing.assert_array_equal(result["reference_source_count"],source_count[parent_cases])
    if "reference" in result:
        np.testing.assert_allclose(result["reference"],reference,rtol=0.,atol=0.,equal_nan=True)
    if "qualified" in result:
        np.testing.assert_array_equal(result["qualified"],qualified)
    if "reference_source_id" in result:
        np.testing.assert_array_equal(result["reference_source_id"],source_id)
    positions=pd.Index(parent_cases).get_indexer(selected);assert np.all(positions>=0)
    print("REFERENCE_PARENT",name,str(path),selected.tolist(),result["reference_source_count"][positions].tolist(),flush=True)
    return {key:result[key][positions] for key in CONFIG["parent_parameters"]}

if args.stage=="features":
    parents=None
else:
    raw_parent=parent("raw",raw);balanced_parent=parent("balanced",balanced)
if args.stage=="source":
    raise SystemExit(0)
model_path.parent.mkdir(parents=True,exist_ok=True)
if args.stage=="replay":
    model=replay_model
    assert str(model["readout_kind"])==CONFIG["readout_kind"]
    assert str(model["regularization_center"])=="reference_source_encoders"
    assert str(model["time_unit"])==settings["unit"]
    np.testing.assert_array_equal(model["trace_id"],raw["trace_id"])
    np.testing.assert_array_equal(model["period"],period)
    model_cases=model.get("case_index",np.arange(len(model["gate_coefficient"])))
    rows=pd.Index(model_cases).get_indexer(selected);assert np.all(rows>=0)
    np.testing.assert_array_equal(model["training_run"][rows],training[selected])
    np.testing.assert_array_equal(model["reference_source_count"][rows],source_count[selected])
    expected_prior=foreground_reference_log_prior(raw,raw_parent,np.arange(len(selected)))
    np.testing.assert_allclose(model["foreground_log_prior"][rows],expected_prior,rtol=0.,atol=1e-12)
    np.testing.assert_allclose(model["raw_background_coefficient"][rows],raw_parent["background_coefficient"],rtol=0.,atol=0.)
    np.testing.assert_allclose(model["balanced_background_coefficient"][rows],balanced_parent["background_coefficient"],rtol=0.,atol=0.)
    np.testing.assert_allclose(model["raw_role_covariance"][rows],raw_parent["role_covariance"],rtol=0.,atol=0.)
    np.testing.assert_allclose(model["balanced_role_covariance"][rows],balanced_parent["role_covariance"],rtol=0.,atol=0.)
    fresh=predict_latent_representation_density(raw,balanced,model,cases=rows,device=args.device,include_log_probability=True)
    checked=["estimate","source_period","group_mean","posterior","balanced_probability","deviation","log_posterior"]
    difference={key:float(np.max(abs(fresh[key]-model[key][rows]))) for key in checked}
    assert max(difference.values())<1e-7
    np.testing.assert_allclose(fresh["posterior"].sum(-1),1.,rtol=0.,atol=1e-12)
    np.testing.assert_allclose(fresh["estimate"]@raw["average"].T,fresh["group_mean"],rtol=0.,atol=1e-11)
    print("UNIFIED_REPLAY",tag,selected.tolist(),difference,flush=True)
    raise SystemExit(0)

def views(name,bundle):
    if not has_views:
        return bundle,None
    if build_features and not (args.refresh_balanced and name=="raw"):
        result,ctx=general_reference_role_features(wave_path,source_context,period,posterior_clock=True,
            source_balancing=name!="raw",noise_cache=background if name=="raw" else None)
        if name=="raw":
            order=pd.Index(background["trace_id"]).get_indexer(result["trace_id"]);matched=order>=0
            result["mixing_quality"][matched]=background["mixing_quality"][order[matched]]
        result["source_order_policy"]=bundle["source_order_policy"]
        if factorized:
            result["source_components"]=np.asarray(str(root/args.source_components))
            result["reference_view_operator"]=np.asarray("measured_source_timewarp_shared_donor_known_gain")
            source_order=pd.Index(source_context["trace_id"]).get_indexer(result["trace_id"])
            assert np.all(source_order>=0)
            ctx["reference_source_id"]=source_context["reference_source_id"][source_order]
        np.savez_compressed(view_paths[name],**result,**{"ctx_"+key:value for key,value in ctx.items()})
    else:
        stored=dict(np.load(view_paths[name],allow_pickle=False))
        result={key:value for key,value in stored.items() if not key.startswith("ctx_")}
        ctx={key[4:]:value for key,value in stored.items() if key.startswith("ctx_")}
    return result,ctx

view_raw,view_context=views("raw",raw);view_balanced,balanced_context=views("balanced",balanced)
if has_positive:
    union=dict(np.load(positive_context_path,allow_pickle=False))
    np.testing.assert_allclose(union["period_bounds"],settings["bounds"]);assert str(union["unit"])==settings["unit"]
    if build_features:
        row_keys=["trace_id","target","qualified","origin_run","origin_trace_id","independent_unit","view_weight","is_original","view_id"]
        missing=~np.isin(union["trace_id"],view_raw["trace_id"])
        extra_context=dict(union,**{key:union[key][missing] for key in row_keys})
        extra_raw,_=general_reference_role_features(positive_path,extra_context,period,posterior_clock=True,source_balancing=False)
        extra_balanced,_=general_reference_role_features(positive_path,extra_context,period,posterior_clock=True,source_balancing=True)
        np.testing.assert_array_equal(extra_raw["trace_id"],extra_balanced["trace_id"])
        extra={key:extra_raw[key] for key in ["trace_id","run","condition","count","mixing_quality","joint_gram","layout"]}
        extra.update({name+"_"+key:value[key] for name,value in [("raw",extra_raw),("balanced",extra_balanced)]
            for key in ["source_base","source_rhs","source_coherent_power"]})
        np.savez_compressed(positive_cache,**extra)
    else:
        extra=dict(np.load(positive_cache,allow_pickle=False))
    def append_positive(name,old):
        all_ids=np.concatenate([old["trace_id"],extra["trace_id"]])
        order=pd.Index(all_ids).get_indexer(union["trace_id"])
        assert np.all(order>=0) and len(np.unique(order))==len(all_ids)==len(union["trace_id"])
        result={key:np.concatenate([old[key],extra[key]])[order] for key in ["trace_id","run","condition","count","mixing_quality"]}
        result.update({key:np.concatenate([old[key],extra[name+"_"+key]])[order] for key in ["source_base","source_rhs","source_coherent_power"]})
        result["joint_gram"]=np.concatenate([old["joint_gram"],extra["joint_gram"]])
        result["layout"]=np.concatenate([old["layout"],extra["layout"]+len(old["joint_gram"])])[order]
        code,runs=pd.factorize(result["run"],sort=True);member=(np.arange(len(runs))[:,None]==code[None]).astype(float)
        result.update(run_index=code,run_names=np.asarray(runs,dtype=str),membership=member,average=member/member.sum(1)[:,None],
            period=period,period_scale=raw["period_scale"])
        return result
    view_raw=append_positive("raw",view_raw);view_balanced=append_positive("balanced",view_balanced)
    view_context={key:union[key] for key in ["target","qualified","origin_run","origin_trace_id","independent_unit","view_weight","is_original","view_id"]}
    groups=pd.Index(view_raw["run_names"]);left=union["pair_left_run"].ravel();right=union["pair_right_run"].ravel()
    view_context["pair_left"]=groups.get_indexer(union["run_name"][np.maximum(left,0)])
    view_context["pair_right"]=groups.get_indexer(union["run_name"][np.maximum(right,0)])
    view_context["pair_eligible"]=union["pair_eligible"].ravel()&(left>=0)&(right>=0)&(view_context["pair_left"]>=0)&(view_context["pair_right"]>=0)
    view_context["pair_left"]=np.maximum(view_context["pair_left"],0);view_context["pair_right"]=np.maximum(view_context["pair_right"],0)
    view_context["pair_effect"]=np.where(view_context["pair_eligible"],union["pair_effect"].ravel()/float(raw["period_scale"]),0.)
if not has_views:
    view_context={"target":reference,"qualified":qualified,"origin_run":raw["run"],"origin_trace_id":raw["trace_id"],
        "independent_unit":independent,"view_weight":np.ones(len(reference)),"is_original":np.ones(len(reference),dtype=bool),
        "view_id":np.repeat("original",len(reference)),"pair_left":np.asarray([],dtype=int),"pair_right":np.asarray([],dtype=int),
        "pair_eligible":np.asarray([],dtype=bool),"pair_effect":np.asarray([],dtype=float)}
np.testing.assert_array_equal(view_raw["trace_id"],view_balanced["trace_id"])
np.testing.assert_allclose(view_raw["mixing_quality"],view_balanced["mixing_quality"],rtol=0.,atol=1e-10)
view_balanced=dict(view_balanced,mixing_quality=view_raw["mixing_quality"])
origin=pd.Index(raw["trace_id"]).get_indexer(view_context["origin_trace_id"]);assert np.all(origin>=0)
view_context["independent_unit"]=independent[origin]
if not factorized:
    view_context["reference_source_id"]=source_id[origin]
origin_run=pd.Index(raw["run_names"]).get_indexer(view_context["origin_run"]);assert np.all(origin_run>=0)
group_origin=np.rint(view_raw["average"]@origin_run).astype(int)
np.testing.assert_array_equal(group_origin[view_raw["run_index"]],origin_run)
view_training=training[selected][:,group_origin]
original=np.flatnonzero(view_context["is_original"])
order=pd.Index(view_context["origin_trace_id"][original]).get_indexer(raw["trace_id"]);assert np.all(order>=0)
original=original[order];assert len(np.unique(original))==len(raw["trace_id"])
np.testing.assert_array_equal(view_context["qualified"][original],qualified)
np.testing.assert_allclose(view_context["target"][original],reference,rtol=0.,atol=0.,equal_nan=True)
group_map=np.rint(raw["average"]@view_raw["run_index"][original]).astype(int)
if args.stage=="features":
    print("UNIFIED_FEATURES",tag,len(raw["trace_id"]),len(view_raw["trace_id"]),flush=True)
    raise SystemExit(0)
case_query=np.concatenate([np.zeros((1,len(reference)),dtype=bool),query])[selected]
protocol=np.concatenate([np.asarray(["full_fit"]),context["outer_protocol"]])[selected]
condition=np.concatenate([np.asarray(["all"]),context["outer_condition"]])[selected]

def save_checkpoint(path,values):
    temporary=path.with_name(path.stem+".pending.npz")
    np.savez_compressed(temporary,**values)
    temporary.replace(path)


def checkpoint(index,current):
    count=index+1;case_ids=selected[:count]
    common={"schema":np.asarray(CONFIG["schema"]),"time_unit":np.asarray(settings["unit"]),
        "model_kind":np.asarray(CONFIG["model_kind"]),"source_operator":np.asarray(CONFIG["source_operator"]),
        "source_core_policy":np.asarray(CONFIG["source_core_policy"]),
        "period_bounds":np.asarray(settings["bounds"]),"training_scope":np.asarray("original_source_units_and_declared_CF"),
        "raw_reference_model":np.asarray(str(parent_paths["raw"])),"balanced_reference_model":np.asarray(str(parent_paths["balanced"])),
        "input_path":np.asarray(str(root/settings["input"])),"source_case_index":case_ids,
        "source_refs_reference_source_id":source_id,"source_refs_independent_unit":independent,
        "source_refs_qualified":qualified,"source_refs_reference":reference,
        "source_refs_training_run":training[case_ids],"source_refs_query_mask":case_query[:count],
        "source_view_ids":np.unique(view_context["view_id"]),
        "zero_cf_pairs":np.asarray(np.sum(view_context["pair_eligible"]&(view_context["pair_effect"]==0.))),
        "positive_cf_pairs":np.asarray(np.sum(view_context["pair_eligible"]&(view_context["pair_effect"]!=0.)))}
    all_model=dict(current,**common,case_index=case_ids,trace_id=view_raw["trace_id"],run_names=view_raw["run_names"],
        run_index=view_raw["run_index"],training_run=view_training[:count],qualified=view_context["qualified"],
        reference=view_context["target"])
    all_model.update({"ctx_"+key:value for key,value in view_context.items()})
    save_checkpoint(folder/(tag+"_representation_mixture_views.npz"),all_model)
    native=dict(current)
    native.update({key:current[key][:,original] for key in ["estimate","source_period","posterior","balanced_probability","deviation","log_posterior"]})
    native["group_mean"]=current["group_mean"][:,group_map]
    native.update(common,case_index=case_ids,trace_id=raw["trace_id"],run_names=raw["run_names"],run_index=raw["run_index"],
        training_run=training[case_ids],qualified=qualified,reference=reference,reference_source_id=source_id,independent_unit=independent,
        query_mask=case_query[:count],outer_protocol=protocol[:count],outer_condition=condition[:count])
    np.testing.assert_allclose(native["estimate"]@raw["average"].T,native["group_mean"],rtol=0.,atol=1e-11)
    save_checkpoint(model_path,native)
    print("UNIFIED_CHECKPOINT",case_ids.tolist(),str(model_path),flush=True)

initial=dict(np.load(root/args.resume,allow_pickle=False)) if args.resume is not None else None
model=fit_latent_representation_density(view_raw,view_balanced,raw_parent,balanced_parent,
    view_context["target"],view_context["qualified"],view_training,view_context["independent_unit"],view_context,
    maxiter=args.maxiter,device=args.device,case_callback=checkpoint,initial_model=initial,case_indices=selected,
    reference_source_id=view_context["reference_source_id"])
print("UNIFIED_MODEL",str(model_path),CONFIG["readout_kind"],selected.tolist(),model["reference_source_count"].tolist(),flush=True)

