"""Evaluate every current fitted removal on complete external test inputs."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1],"output":"results/0.3.0",
    "matched_mean_parameters":"results/0.3.1/native_matched_mean_parameters.npz"},
    "threads":os.cpu_count(),"plot":{},"zero_tolerance":1e-12,"individual_tolerance_h":2.,
    "shorter_counts":{104:95,145:120},"missing_patterns":5,"missing_stride":11}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
import sys,argparse,json,hashlib,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update(CONFIG["plot"])
root=CONFIG["paths"]["root"]
sys.path.insert(0,str(root/"software"))
from source_assignment_unified import feature_bundle,prediction_matrix,MODEL_KIND,SOURCE_OPERATOR,CORE_POLICY,POOL_PARAMETER_KEYS,general_source_role_features,predict_general_source_roles,predict_latent_representation_density,posterior_clock_role_features,observations,measured_mixing_quality,reference_source_observations,predict_reference_source_w2,predict_reference_calibration_mixture,_polynomial
from source_assignment_core import interpolate_mean
from period_support import project_periods_with_mean
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset",required=True,choices=["controlled","single","dual","scn","coupled"])
parser.add_argument("--stage",choices=["all","features","quality","evaluate","readback"],default="all")
parser.add_argument("--phase",choices=["Pre","TTX","Wash"],default="Pre")
parser.add_argument("--length",type=int,choices=[104,145],default=145)
parser.add_argument("--acquisition",choices=["native","shorter","missing"],default="native")
parser.add_argument("--output-dir",type=Path,help="Existing fitted-model directory; write its external evaluations here")
parser.add_argument("--role-model",action="store_true",help="Evaluate the current shared source-role development model")
parser.add_argument("--role-augmented",action="store_true",help="Evaluate its joint counterfactual-view fit")
parser.add_argument("--posterior-clock",action="store_true",help="Use frequency-marginal source measurements and the measured peer")
parser.add_argument("--reference-role",action="store_true",help="Use the shared source law fitted on original observations")
parser.add_argument("--reuse-clock-features",action="store_true")
parser.add_argument("--source-role-model",type=Path,help="Shared source-role weights on actual observation geometry")
parser.add_argument("--comparison-model",type=Path,action="append",default=[])
parser.add_argument("--evaluation-label",default="")
parser.add_argument("--device",choices=["cpu","cuda"],default="cuda",help="Device for conditional source-density inference")
args=parser.parse_args()
args.role_model=args.role_model or args.role_augmented or args.posterior_clock or args.reference_role
assert args.acquisition=="native" or args.dataset=="controlled", "Changed acquisitions are defined only for controlled constant-period truth"
folder=args.output_dir or (args.source_role_model.parent if args.source_role_model is not None else root/CONFIG["paths"]["output"])
folder.mkdir(parents=True,exist_ok=True)
specifications={"controlled":{"folder":"data/sota_dev/final_model_controlled","input":"inputs.parquet","truth":"truth_cells.csv"},
    "single":{"folder":"data/sota_dev/recomputed_clean","input":"lemna2021_single_reporter_input.parquet","truth":"lemna2021_single_reporter_references.csv"},
    "dual":{"folder":"data/external/lemna_watanabe2021","input":"dual_reporter_observations.parquet","truth":"author_period_references.csv"},
    "scn":{"folder":"data/external/shan2020_semisynthetic","input":"input_"+args.phase+"_l"+str(args.length)+".parquet","truth":"mixture_records.csv"},
    "coupled":{"folder":"data/external/wang2026_mechanism","input":"inputs.parquet","truth":"truth_cells.csv"}}
baseline_sources={
    "controlled":{"classical":"data/sota_dev/final_model_controlled/classical_predictions.csv",
        "legacy":"data/sota_dev/final_model_controlled/predictions_assignment_common_baseline.csv",
        "frequency":"data/sota_dev/final_model_controlled/predictions_assignment_frequency_geometry.csv"},
    "single":{"classical":"data/external/lemna_watanabe2021/final_0_2_4_classical/classical_predictions.csv",
        "legacy":"data/sota_dev/recomputed_clean/lemna2021_source_assignment_common_baseline_predictions.csv",
        "frequency":"data/sota_dev/recomputed_clean/lemna2021_source_assignment_frequency_geometry_predictions.csv"},
    "scn":{"classical":"data/external/shan2020_semisynthetic/classical_predictions.csv",
        "legacy":"data/external/shan2020_semisynthetic/final_0_2_4/"+args.phase+"_l"+str(args.length)+"_predictions.csv",
        "frequency":"data/external/shan2020_semisynthetic/frequency_0_2_4/"+args.phase+"_l"+str(args.length)+"_predictions.csv"},
    "coupled":{"classical":"data/external/wang2026_mechanism/classical_predictions.csv",
        "legacy":"data/external/wang2026_mechanism/predictions_final_0_2_4.csv",
        "frequency":"data/external/wang2026_mechanism/predictions_frequency_0_2_4.csv"}}
spec=specifications[args.dataset]
input_folder=root/spec["folder"]
frame=pd.read_parquet(input_folder/spec["input"])
truth=pd.read_csv(input_folder/spec["truth"],encoding="utf-8")
if args.dataset=="dual":
    frame=frame[["trace_id","run","time_h","no"]].rename(columns={"time_h":"x","no":"signal_y"}).assign(condition="observational")
    truth=truth.rename(columns={"target_period_h":"reference_period_h","target_reference_usable":"qualified"})
    truth["qualified"]=truth.qualified.fillna(False).astype(bool)
frame=frame.sort_values(["trace_id","x"]).reset_index(drop=True)
original_count=frame.groupby("trace_id").size()
if args.acquisition!="native":
    assert frame.trace_id.nunique()==2304 and truth.trace_id.nunique()==2304
    sample_index=frame.groupby("trace_id").cumcount()
    count=frame.trace_id.map(original_count)
    pair_index=pd.Series(pd.factorize(truth.pair_id.str.replace(r"_l\d+(?=_|$)","",regex=True),sort=True)[0],index=truth.trace_id)
    pattern=frame.trace_id.map(pair_index)%CONFIG["missing_patterns"]
    shorter=sample_index<count.map(CONFIG["shorter_counts"])
    missing=~((sample_index>0)&(sample_index<count-1)&(sample_index%CONFIG["missing_stride"]==pattern))
    frame=frame.loc[{"shorter":shorter,"missing":missing}[args.acquisition]].copy()
    assert frame.trace_id.nunique()==2304
if args.source_role_model is not None:
    role_path=args.source_role_model
    role_model=dict(np.load(role_path,allow_pickle=False))
    role_paths=[role_path]+args.comparison_model
    role_models=[role_model]+list(map(lambda path:dict(np.load(path,allow_pickle=False)),args.comparison_model))
    reference_mixture=str(role_model.get("readout_kind",""))=="information_conditioned_reference_W2"
    reference_w2=str(role_model.get("readout_kind","")) in {"reference_source_bounded_W2","information_conditioned_reference_W2"}
    conditional=str(role_model.get("readout_kind",""))=="conditional_latent_representation_density"
    source_balancing=not reference_w2 and not conditional and str(role_model["source_representation"])!="observed_physical_peer"
    paired_core=any(str(other.get("core_measurements","single"))=="paired" for other in role_models)
    comparison_names=list(map(lambda pair:str(pair[0].get("control_name","matched_density" if str(pair[0].get("role_kind","gaussian"))=="density" else pair[1].stem)),zip(role_models[1:],role_paths[1:])))
    names=np.asarray(["unified" if reference_w2 else "source_roles"]+[name.removeprefix("task_source_assignment_") for name in comparison_names])
    assert len(np.unique(names))==len(names)
    list(map(lambda other:np.testing.assert_array_equal(other["period"],role_model["period"]),role_models))
    assert all(str(other.get("readout_kind","")) in {"reference_source_bounded_W2","information_conditioned_reference_W2"} for other in role_models) if reference_w2 else all(str(other["source_representation"])==str(role_model["source_representation"]) for other in role_models)
    model=dict(role_model,methods=names,method_indices=np.arange(len(names)),
        parameter_policy=np.asarray([str(other["parameter_policy"]) for other in role_models]) if reference_w2 else np.repeat("matched_information_source_model",len(names)),method_applicable=np.ones(len(names),dtype=bool))
else:
    reference_w2=False
    reference_mixture=False
    model=dict(np.load(folder/"unified_model.npz",allow_pickle=False))
    assert str(model["model_kind"])==MODEL_KIND and str(model["source_operator"])==SOURCE_OPERATOR
    assert str(model["source_core_policy"])==CORE_POLICY
if args.stage=="readback" and not reference_w2:
    parser.error("--stage readback is defined for reference_source_bounded_W2")
tag=args.dataset if args.dataset!="scn" else "scn_"+args.phase+"_l"+str(args.length)
tag+= "" if args.acquisition=="native" else "__"+args.acquisition
input_tag=("role_input_"+tag) if args.role_model else tag
if reference_w2:
    input_tag="reference_"+input_tag
elif args.source_role_model is not None and not source_balancing:
    input_tag="observed_"+input_tag
tag=(("role_augmented_" if args.role_augmented else "role_")+tag) if args.role_model else tag
tag=tag.replace("role_","clock_reference_" if args.reference_role else "clock_",1) if args.posterior_clock else tag
tag=args.evaluation_label+tag
if reference_w2:
    kernel_files=["source_amplitude_normalization.py","source_assignment_core.py","source_assignment_unified.py"]
    kernel_fingerprint=hashlib.sha256(b"".join(map(lambda name:name.encode("utf-8")+b"\0"+
        (root/"software"/name).read_bytes(),kernel_files))).hexdigest()
    signal_digest=hashlib.sha256(pd.util.hash_pandas_object(
        frame[["trace_id","run","condition","x","signal_y"]],index=False).to_numpy().tobytes()).hexdigest()
    def calibration_observations(options,support):
        started=time.perf_counter()
        specification=json.dumps({"representation_options":options,"fundamental_support":bool(support)},
            sort_keys=True,separators=(",",":"))
        fingerprint=hashlib.sha256(model["period"].tobytes()+specification.encode("utf-8")+signal_digest.encode("ascii")+kernel_fingerprint.encode("ascii")).hexdigest()[:16]
        cache=folder/(input_tag+"_"+fingerprint+"_features.npz")
        reused=args.stage=="evaluate" or cache.is_file()
        if reused:
            with np.load(cache,allow_pickle=False) as saved:
                measured={key:saved[key] for key in saved.files if key not in {"raw_features","core_features"}}
            assert str(measured["cache_format"])=="reference_calibration_No_features_0.3.1"
            assert str(measured["input_signal_sha256"])==signal_digest
            assert str(measured["source_kernel_sha256"])==kernel_fingerprint
            assert str(measured["calibration_specification_json"])==specification
            np.testing.assert_array_equal(measured["period"],model["period"])
        else:
            measured=reference_source_observations(frame,model["period"],options,
                fundamental_support=bool(support),device=args.device)
            measured.update(cache_format=np.asarray("reference_calibration_No_features_0.3.1"),
                input_signal_sha256=np.asarray(signal_digest),source_kernel_sha256=np.asarray(kernel_fingerprint),
                calibration_specification_json=np.asarray(specification))
        raw=_polynomial(measured["base_features"])
        core_base=measured["base_features"].copy()
        core_base[:,:,11]=measured["coherent_power"]
        core=_polynomial(core_base)
        if not reused:
            np.testing.assert_array_equal(raw,measured["raw_features"])
            np.testing.assert_array_equal(core,measured["core_features"])
        measured.update(raw_features=raw,core_features=core,
            cache_storage=np.asarray("base13_coherent_power_exact_polynomial104"))
        if not reused:
            stored={key:np.asarray(value) for key,value in measured.items() if key not in {"raw_features","core_features"}
                and isinstance(value,(np.ndarray,np.generic,str,int,float,bool))}
            np.savez_compressed(cache,**stored)
        np.testing.assert_array_equal(measured["trace_id"],frame.trace_id.drop_duplicates().to_numpy(dtype=str))
        actual_layout=frame.pivot(index="trace_id",columns="x",values="signal_y").reindex(index=measured["trace_id"])
        np.testing.assert_array_equal(measured["time"],actual_layout.columns.to_numpy(dtype=float))
        np.testing.assert_array_equal(measured["mask"],actual_layout.notna().to_numpy())
        np.testing.assert_array_equal(measured["count"],actual_layout.notna().sum(1).to_numpy())
        print("CALIBRATION_FEATURE_CACHE",cache.name,"reuse" if reused else "cold",
            len(measured["trace_id"]),"records",round(time.perf_counter()-started,3),"seconds",flush=True)
        return measured
if reference_mixture:
    assert str(role_model["schema"])=="source_assignment_unified_0.3.1"
    # Each declared calibration is measured from this complete No-only input.
    calibration_bundles=list(map(lambda r:calibration_observations(
        json.loads(str(role_model["calibration_representation_options_json"][r])),
        bool(role_model["calibration_fundamental_support"][r])),
        np.arange(len(role_model["calibration_representation_options_json"]))))
    bundle=calibration_bundles[0]
elif reference_w2:
    assert str(role_model["schema"])=="source_assignment_unified_0.3.1"
    representation_options=json.loads(str(role_model["representation_options_json"]))
    fundamental_support=bool(role_model.get("fundamental_support",True))
    if args.stage in {"evaluate","quality"}:
        bundle=dict(np.load(folder/(input_tag+"_features.npz"),allow_pickle=False))
        assert json.loads(str(bundle["representation_options_json"]))==representation_options
        assert bool(bundle["fundamental_support_enabled"])==fundamental_support
    else:
        bundle=reference_source_observations(frame,model["period"],representation_options,
            fundamental_support=fundamental_support,device=args.device)
        bundle["representation_options_json"]=np.asarray(json.dumps(representation_options,sort_keys=True))
        bundle["fundamental_support_enabled"]=np.asarray(fundamental_support)
        if args.stage in {"features","all"}:
            np.savez_compressed(folder/(input_tag+"_features.npz"),**bundle)
else:
    if args.stage in {"evaluate","quality"}:
        bundle=dict(np.load(folder/(input_tag+"_features.npz"),allow_pickle=False))
    else:
        if args.source_role_model is not None:
            bundle=posterior_clock_role_features(frame,period_min=float(model["period"].min()),period_max=float(model["period"].max()),candidates=len(model["period"]),source_balancing=source_balancing,paired_core=paired_core)
        else:
            controls=bool(np.any(np.isin(model["method_indices"],[3,11])))
            bundle=feature_bundle(frame,float(model["period"].min()),float(model["period"].max()),len(model["period"]),include_controls=controls)
        if args.role_model and args.dataset in {"controlled","coupled"}:
            np.savez_compressed(folder/(input_tag+"_features.npz"),**bundle)
        if args.stage=="features":
            cached=bundle
            if args.source_role_model is not None:
                keep=["period","period_scale","trace_id","run","condition","count","time","mask","run_names","run_index","membership","average",
                    "identity_base_features","source_base","source_rhs","joint_gram","layout","source_coherent_power","source_model_probability","source_representation","source_order_policy","mixing_quality",
                    "source_observation_energy","source_pair_scale","source_peer_weight_square"]
                if paired_core:
                    keep+=["companion_base_features","companion_coherent_power","companion_rhs"]
                cached={key:bundle[key] for key in keep}
            np.savez_compressed(folder/(input_tag+"_features.npz"),**cached)
        if args.source_role_model is not None and args.stage=="all":
            keep=["period","period_scale","trace_id","run","condition","count","time","mask","run_names","run_index","membership","average",
                "identity_base_features","source_base","source_rhs","joint_gram","layout","source_coherent_power","source_model_probability","source_representation","source_order_policy","mixing_quality",
                "source_observation_energy","source_pair_scale","source_peer_weight_square"]
            if paired_core:
                keep+=["companion_base_features","companion_coherent_power","companion_rhs"]
            np.savez_compressed(folder/(input_tag+"_features.npz"),**{key:bundle[key] for key in keep})
if args.stage=="quality" and not reference_w2:
    data=observations(frame)
    np.testing.assert_array_equal(data["metadata"].index.to_numpy(),bundle["trace_id"])
    bundle["mixing_quality"]=measured_mixing_quality(data,bundle["period"],float(bundle["period_scale"]))
    np.savez_compressed(folder/(input_tag+"_features.npz"),**bundle)
if args.stage in {"features","quality"}:
    print("UNIFIED_EXTERNAL_FEATURES",tag,len(bundle["trace_id"]),flush=True)
    raise SystemExit(0)
observed=frame.groupby("trace_id").agg(n_observed=("x","size"),first_x=("x","min"),last_x=("x","max"))
np.testing.assert_array_equal(bundle["count"],observed.loc[bundle["trace_id"],"n_observed"])
np.testing.assert_array_equal(bundle["time"],np.sort(frame.x.unique()))
if args.source_role_model is None:
    assert str(bundle["source_operator"])==SOURCE_OPERATOR
np.testing.assert_array_equal(bundle["period"],model["period"])
if args.source_role_model is not None:
    if reference_w2:
        deployment_rows=list(map(lambda m:pd.Index(m["case_index"]).get_indexer([0]),role_models))
        assert all(np.all(rows>=0) for rows in deployment_rows)
        predicted=[]
        for other,rows in zip(role_models,deployment_rows):
            if str(other.get("readout_kind",""))=="information_conditioned_reference_W2":
                same_options=reference_mixture and np.array_equal(other["calibration_representation_options_json"],role_model["calibration_representation_options_json"])
                same_support=reference_mixture and np.array_equal(other["calibration_fundamental_support"],role_model["calibration_fundamental_support"])
                value=predict_reference_calibration_mixture(frame,other,cases=rows,device=args.device,
                    include_diagnostics=True,source_bundles=calibration_bundles if same_options and same_support else list(map(
                        lambda r:calibration_observations(json.loads(str(other["calibration_representation_options_json"][r])),
                            bool(other["calibration_fundamental_support"][r])),
                        np.arange(len(other["calibration_representation_options_json"])))))
            else:
                other_options=json.loads(str(other["representation_options_json"]))
                support=bool(other.get("fundamental_support",True))
                measured=bundle if not reference_mixture and other_options==representation_options and support==fundamental_support else reference_source_observations(
                    frame,other["period"],other_options,fundamental_support=support,device=args.device)
                value=predict_reference_source_w2(measured,other,cases=rows,device=args.device)
            value["original_period"]=value.get("original_period",np.full_like(value["estimate"],np.nan))
            predicted.append(value)
        if args.stage=="readback":
            value=predicted[0]
            checks={"probability":float(np.max(abs(value["posterior"].sum(-1)-1.))),
                "expectation_h":float(np.max(abs(value["posterior"]@bundle["period"]-value["estimate"]))),
                "mean_h":float(np.max(abs(value["estimate"]@bundle["average"].T-value["group_mean"]))),
                "target_h":float(np.max(abs(value["group_mean"]-value["calibrated_mean_target"])))}
            assert checks["probability"]<1e-10 and max(v for k,v in checks.items() if k!="probability")<1e-6
            print("EXTERNAL_REFERENCE_SOURCE_API_READBACK",json.dumps({"dataset":args.dataset,"records":len(bundle["trace_id"]),
                "method":"task_source_assignment_unified","model":str(role_path),"case_index":0,
                "evaluation_tables":"not_run","W2_relations":checks}),flush=True)
            raise SystemExit(0)
    elif conditional:
        companion_tag=input_tag.removeprefix("observed_")
        balanced=dict(np.load(folder/(companion_tag+"_features.npz"),allow_pickle=False)) if args.stage=="evaluate" else posterior_clock_role_features(frame,period_min=float(model["period"].min()),period_max=float(model["period"].max()),candidates=len(model["period"]),source_balancing=True)
        np.testing.assert_array_equal(balanced["trace_id"],bundle["trace_id"])
        np.testing.assert_array_equal(balanced["count"],bundle["count"])
        np.testing.assert_array_equal(balanced["time"],bundle["time"])
        np.testing.assert_array_equal(balanced["mask"],bundle["mask"])
        if args.stage!="evaluate":
            keep=["period","period_scale","trace_id","run","condition","count","time","mask","run_names","run_index","membership","average",
                "identity_base_features","source_base","source_rhs","joint_gram","layout","source_coherent_power","source_model_probability","source_representation","source_order_policy","mixing_quality",
                "source_observation_energy","source_pair_scale","source_peer_weight_square"]
            np.savez_compressed(folder/(companion_tag+"_features.npz"),**{key:balanced[key] for key in keep})
        predicted=list(map(lambda other:predict_latent_representation_density(bundle,balanced,other,cases=[0],device=args.device),role_models))
        list(map(lambda value:value.update({"original_period":np.full_like(value["estimate"],np.nan)}),predicted))
    else:
        assert str(bundle["source_representation"])==str(role_model["source_representation"])
        evidence_cache={}
        predicted=list(map(lambda other:predict_general_source_roles(bundle,other,cases=[0],evidence_cache=evidence_cache),role_models))
    fresh={key:np.stack(list(map(lambda value:value[key],predicted))) for key in predicted[0]}
    fresh["population_location"]=np.full_like(fresh["group_mean"],np.nan)
elif args.role_model:
    role_path=folder/("native_reference_role_model.npz" if args.reference_role else "native_role_augmented_probe.npz" if args.role_augmented else "native_role_model_probe.npz")
    role_model=dict(np.load(role_path,allow_pickle=False))
    if args.posterior_clock:
        clock_tag=tag.replace("clock_reference_","clock_",1)
        if args.reuse_clock_features:
            bundle=dict(np.load(folder/(clock_tag+"_features.npz"),allow_pickle=False))
        else:
            bundle=posterior_clock_role_features(frame,bundle)
            if args.dataset!="scn":
                np.savez_compressed(folder/(clock_tag+"_features.npz"),**bundle)
    else:
        bundle=general_source_role_features(frame,bundle)
    role_prediction=predict_general_source_roles(bundle,role_model,cases=[0])
    fresh={key:value[None] for key,value in role_prediction.items()}
    fresh["population_location"]=np.full_like(fresh["group_mean"],np.nan)
    model=dict(model,methods=np.asarray(["general_source_roles"]),method_indices=np.asarray([0]),parameter_policy=np.asarray(["joint_source_views_gain_pairs_monotone_mean" if args.role_augmented else "one_role_law_uniform_period_training_monotone_mean"]),method_applicable=np.asarray([True]))
else:
    fresh=prediction_matrix(bundle,model)
assert fresh["estimate"].shape[2]==len(bundle["trace_id"]) and np.isfinite(fresh["estimate"]).all()
methods=np.char.add("task_source_assignment_",model["methods"])
names=bundle["trace_id"]
if args.source_role_model is not None:
    full_success=np.broadcast_to(np.asarray([bool(other["success"][rows[0]]) for other,rows in zip(role_models,deployment_rows)] if reference_w2 else list(map(lambda other:bool(other["success"][0]),role_models)))[:,None],(len(role_models),len(names)))
    args.role_model=True
else:
    fitted=dict(np.load(folder/"native_fit.npz",allow_pickle=False))
    list(map(lambda key:np.testing.assert_array_equal(fitted[key],model[key]),["model_kind","source_operator","source_core_policy"]))
if args.role_model and args.source_role_model is None:
    full_success=role_model["success"][0]&bundle["pool_period_fit_success"][bundle["run_index"]][None]
elif args.source_role_model is None:
    fit_order=pd.Index(fitted["methods"]).get_indexer(model["methods"])
    assert np.all(fit_order>=0)
    list(map(lambda key:np.testing.assert_array_equal(fitted[key][fit_order],model[key]),POOL_PARAMETER_KEYS))
    full_success=fitted["optimizer_success_by_case"][fit_order,0,None]&bundle["pool_period_fit_success"][bundle["run_index"]][None]
metadata=pd.DataFrame({"trace_id":names,"run":bundle["run"],"condition":bundle["condition"],
    "n_observed":bundle["count"],"native_n_observed":original_count.loc[names].to_numpy(),
    "first_x":observed.loc[names,"first_x"].to_numpy(),"last_x":observed.loc[names,"last_x"].to_numpy(),
    "acquisition":args.acquisition})
metadata["mask_pattern"]=np.nan
if args.acquisition=="missing":
    metadata["mask_pattern"]=metadata.trace_id.map(pair_index)%CONFIG["missing_patterns"]
predictions=pd.DataFrame({"method":np.repeat(methods,len(names)),"trace_id":np.tile(names,len(methods)),
    "estimate_h":fresh["estimate"][:,0].ravel(),"source_period_h":fresh["source_period"][:,0].ravel(),
    "original_period_h":fresh["original_period"][:,0].ravel(),
    "experiment_mean_h":fresh["group_mean"][:,0][:,bundle["run_index"]].ravel(),
    "finite_cohort_mean_h":fresh["group_mean"][:,0][:,bundle["run_index"]].ravel(),
    "population_location_h":fresh["population_location"][:,0][:,bundle["run_index"]].ravel(),
    "comparison_group":"current_model","prediction_source":np.repeat(list(map(lambda path:Path(os.path.relpath(path,root)).as_posix(),role_paths)),len(names)) if args.source_role_model is not None else Path(os.path.relpath(role_path if args.role_model else folder/"unified_model.npz",root)).as_posix(),
    "training_information":np.repeat(np.where(((model["method_indices"]==8)|(args.role_model and not args.role_augmented))&(args.source_role_model is None),"original_foreground_period_references","original_source_references_under_training_mask" if reference_w2 else "allowed_source_reference_views"),len(names)),"method_status":"ok",
    "fit_status":np.where(full_success,"completed","not_converged").ravel(),
    "parameter_policy":np.repeat(model["parameter_policy"],len(names)),
    "applicable":np.repeat(model["method_applicable"],len(names))})
# Own-model shape/finite checks stay above every appended comparator.
h1= pd.DataFrame({"method":"unlabelled_H1","trace_id":names,
    "estimate_h":bundle["period"][bundle["identity_base_features"][:,:,0].argmax(axis=1)],"comparison_group":"unlabelled_point",
    "prediction_source":"current_actual_timestamps_12_48_H1","training_information":"none",
    "method_status":"ok","fit_status":"not_fitted","applicable":True})
baseline_methods=["MESA","MFourFit","native_H1_continuous","native_H2"]
baseline_index=pd.MultiIndex.from_product([baseline_methods+["legacy_0_2_4","legacy_frequency"],names],names=["method","trace_id"])
if args.dataset!="dual" and args.acquisition=="native":
    source=baseline_sources[args.dataset]
    classical=pd.read_csv(root/source["classical"],encoding="utf-8")
    classical=classical.loc[classical.trace_id.isin(names)&classical.readout_training.eq("none")].copy()
    assert classical.input_protocol.eq("all_native_hourly_No_no_target_reference").all()
    assert not classical.duplicated(["method","trace_id"]).any()
    assert classical.groupby("method").trace_id.nunique().eq(len(names)).all()
    saved_h1=classical.loc[classical.method.eq("native_H1")].set_index("trace_id").estimate_h.reindex(names).to_numpy()
    np.testing.assert_array_equal(saved_h1,h1.estimate_h.to_numpy())
    classical=classical.loc[classical.method.isin(baseline_methods)].assign(
        comparison_group="unlabelled_point",prediction_source=source["classical"],training_information="none",fit_status="not_fitted",applicable=True)
    legacy=pd.read_csv(root/source["legacy"],encoding="utf-8")
    frequency=pd.read_csv(root/source["frequency"],encoding="utf-8")
    assert legacy.trace_id.is_unique and frequency.trace_id.is_unique
    assert set(legacy.trace_id)==set(names) and set(frequency.trace_id)==set(names)
    assert legacy.method.eq("task_source_assignment_common_baseline").all()
    assert frequency.method.eq("task_source_assignment_frequency_geometry").all()
    legacy=legacy.assign(method="legacy_0_2_4",comparison_group="legacy_original_protocol",prediction_source=source["legacy"],
        training_information="original_0.2.4_labels_maps_and_exclusions",fit_status="legacy_frozen",applicable=True)
    frequency=frequency.assign(method="legacy_frequency",comparison_group="legacy_original_protocol",prediction_source=source["frequency"],
        training_information="original_0.2.4_labels_maps_and_exclusions",fit_status="legacy_frozen",applicable=True)
    baseline=pd.concat([classical,legacy,frequency],ignore_index=True).set_index(["method","trace_id"]).reindex(baseline_index).reset_index()
    assert baseline.prediction_source.notna().all()
else:
    comparator_folder=root/"results/0.3.1"/("classical_dual" if args.dataset=="dual" else "classical_controlled_"+args.acquisition)
    classical=pd.read_csv(comparator_folder/"classical_predictions.csv",encoding="utf-8")
    assert classical.input_protocol.eq("all_observed_No_no_target_reference").all()
    assert classical.readout_training.eq("none").all() and classical.acquisition.eq(args.acquisition).all()
    assert not classical.duplicated(["method","trace_id"]).any()
    assert classical.groupby("method").trace_id.nunique().eq(len(names)).all()
    np.testing.assert_array_equal(np.sort(classical.trace_id.unique()),np.sort(names))
    official_input=pd.read_csv(root/"build/classical_transfer"/(args.dataset+"_"+args.acquisition)/"native_input.tsv",
        sep="\t",header=None,names=["trace_id","time_relative_h","signal_y"],float_precision="round_trip")
    measured_input=frame.assign(time_relative_h=frame.x-frame.groupby("run").x.transform("min")).sort_values(["trace_id","time_relative_h"])
    pd.testing.assert_frame_equal(official_input,measured_input[["trace_id","time_relative_h","signal_y"]].reset_index(drop=True),
        check_dtype=False,check_exact=True)
    classical=classical.loc[classical.method.isin(baseline_methods)].assign(
        comparison_group="unlabelled_point",prediction_source=Path(os.path.relpath(comparator_folder/"classical_predictions.csv",root)).as_posix(),
        training_information="none",fit_status="not_fitted",applicable=True)
    unavailable=pd.MultiIndex.from_product([["legacy_0_2_4","legacy_frequency"],names],names=["method","trace_id"]).to_frame(index=False).assign(
        estimate_h=np.nan,comparison_group="unavailable_fixed_comparator",prediction_source="no_same_input_saved_prediction",
        training_information="unavailable",method_status="not_recomputed_for_"+args.acquisition+"_"+args.dataset,
        fit_status="not_evaluated",applicable=False)
    baseline=pd.concat([classical,unavailable],ignore_index=True).set_index(["method","trace_id"]).reindex(baseline_index).reset_index()
comparison_columns=["method","trace_id","estimate_h","comparison_group","prediction_source","training_information","method_status","fit_status","applicable"]
baseline["method"]=baseline.method.replace(dict(zip(baseline_methods,
    ["native_MESA","native_MFourFit","native_H1_continuous","native_H2"])))
if reference_w2:
    matched_path=root/CONFIG["paths"]["matched_mean_parameters"]
    matched_bank=dict(np.load(matched_path,allow_pickle=False))
    np.testing.assert_array_equal(matched_bank["period"],bundle["period"])
    raw_method=np.asarray(list(map(lambda name:{"MESA":"native_MESA","MFourFit":"native_MFourFit",
        "unlabelled_H1":"unlabelled_H1"}[str(name)],matched_bank["native_methods"])))
    raw=pd.concat([h1[["method","trace_id","estimate_h"]],baseline[["method","trace_id","estimate_h"]]],ignore_index=True)
    raw_table=raw.pivot(index="trace_id",columns="method",values="estimate_h").reindex(index=names,columns=raw_method)
    matched_points=raw_table.to_numpy(dtype=float).T
    member=bundle["membership"];average=bundle["average"];run=bundle["run_index"]
    available=np.isfinite(matched_points)
    complete=np.einsum("gn,mn->mg",member,available.astype(float),optimize=True)==member.sum(1)[None]
    point_mean=np.einsum("mn,gn->mg",np.nan_to_num(matched_points,nan=0.),average,optimize=True)
    # This parameter bank stores physical-hour knots, as written by the Native evaluator.
    calibrated=interpolate_mean(point_mean[:,None],matched_bank["calibration_prediction"][:,0:1],
        matched_bank["calibration_mean"][:,0:1],matched_bank["calibration_experiments"][:,0:1],identity_tails=True)[:,0]
    clean_deviation=np.nan_to_num(matched_points-point_mean[:,run],nan=0.)
    target_mean=np.clip(calibrated,matched_bank["period"].min(),matched_bank["period"].max())
    unbounded=target_mean[:,run]+matched_bank["within_scale"][:,0,None]*clean_deviation
    bounded=project_periods_with_mean(unbounded,target_mean,member,
        matched_bank["period"].min(),matched_bank["period"].max())
    matched_estimate=np.where(complete[:,run]&available,bounded,np.nan)
    np.testing.assert_allclose(np.einsum("mn,gn->mg",bounded,average,optimize=True),target_mean,rtol=0.,atol=1e-6)
    matched=pd.DataFrame({"method":np.repeat(matched_bank["methods"],len(names)),"trace_id":np.tile(names,len(raw_method)),
        "estimate_h":matched_estimate.ravel(),"experiment_mean_h":np.where(complete[:,run],target_mean[:,run],np.nan).ravel(),
        "finite_cohort_mean_h":np.where(complete[:,run],target_mean[:,run],np.nan).ravel(),"raw_point_method":np.repeat(raw_method,len(names)),
        "comparison_group":"current_target_calibrated_point",
        "prediction_source":Path(os.path.relpath(matched_path,root)).as_posix(),
        "training_information":"native_full_fit_source_period_references_only",
        "training_case":0,"calibration_coordinate":"physical_period_h",
        "parameter_policy":"frozen_native_F0_PAV_identity_tails_within_scale_bounded_mean_projection",
        "method_status":np.where(complete[:,run],"ok","raw_point_cohort_incomplete").ravel(),
        "fit_status":"completed","applicable":True})
    predictions=pd.concat([predictions,h1,baseline[comparison_columns],matched],ignore_index=True)
else:
    predictions=pd.concat([predictions,h1,baseline[comparison_columns]],ignore_index=True)
predictions=predictions.merge(metadata,on="trace_id",validate="many_to_one")
methods=predictions.method.drop_duplicates().to_numpy(dtype=str)
truth=truth.loc[truth.trace_id.isin(names)].copy()
assert len(truth)==len(names) and truth.trace_id.is_unique
cells=predictions.merge(truth.loc[:,~truth.columns.isin(["run","condition"])],on="trace_id",validate="many_to_one")
assert len(cells)==len(names)*len(methods) and not cells.duplicated(["method","trace_id"]).any()
cells["estimate_available"]=np.isfinite(cells.estimate_h)
cells["fit_not_converged"]=cells.fit_status.eq("not_converged")
coverage=cells.groupby("method",sort=False).agg(n_input=("trace_id","size"),n_estimated=("estimate_available","sum"),
    n_observed_min=("n_observed","min"),n_observed_max=("n_observed","max"),applicable=("applicable","all"),
    comparison_group=("comparison_group","first"),prediction_source=("prediction_source","first"),
    n_fit_not_converged=("fit_not_converged","sum"),
    fit_status=("fit_status",lambda value:";".join(sorted(value.unique()))),method_status=("method_status",lambda value:";".join(sorted(value.unique())))).reset_index()
if args.dataset in {"controlled","coupled"}:
    cells["absolute_period_error_h"]=pd.Series(np.where(cells.estimate_available,abs(cells.estimate_h-cells.true_period_h),np.inf)).where(cells.applicable)
    keys=["method","run","condition","pair_id","family","scenario","length","replicate"]
    experiments=cells.groupby(keys,sort=False).agg(n_input=("trace_id","size"),n_estimated=("estimate_available","sum"),estimate_mean_h=("estimate_h","mean"),
        n_observed_min=("n_observed","min"),n_observed_max=("n_observed","max"),applicable=("applicable","all"),
        reference_mean_h=("true_period_h","mean"),period_mae_h=("absolute_period_error_h","mean")).reset_index()
    experiments["estimate_mean_h"]=experiments.estimate_mean_h.where(experiments.n_input.eq(experiments.n_estimated))
    pair_keys=["method","pair_id","family","scenario","length","replicate"]
    treated=experiments.loc[experiments.condition.eq("treated")].drop(columns="condition")
    control=experiments.loc[experiments.condition.eq("control")].drop(columns="condition")
    pairs=treated.merge(control,on=pair_keys,suffixes=("_treated","_control"),validate="one_to_one")
    pairs["reference_effect_h"]=pairs.reference_mean_h_treated-pairs.reference_mean_h_control
    pairs["estimate_effect_h"]=pairs.estimate_mean_h_treated-pairs.estimate_mean_h_control
    pairs["absolute_effect_error_h"]=pd.Series(np.where(np.isfinite(pairs.estimate_effect_h),abs(pairs.estimate_effect_h-pairs.reference_effect_h),np.inf)).where(pairs.applicable_treated&pairs.applicable_control)
    pairs["nonzero"]=abs(pairs.reference_effect_h)>CONFIG["zero_tolerance"]
    pairs["correct_sign"]=pairs.nonzero&(np.sign(pairs.estimate_effect_h)==np.sign(pairs.reference_effect_h))
    pairs["recovered"]=pairs.correct_sign&(pairs.absolute_effect_error_h<abs(pairs.reference_effect_h))
    pairs["nonzero_error_h"]=pairs.absolute_effect_error_h.where(pairs.nonzero)
    pairs["zero_error_h"]=pairs.absolute_effect_error_h.where(~pairs.nonzero)
    summary=pairs.groupby(["method","family","length"],sort=False).agg(n_pairs=("pair_id","size"),
        response_mae_h=("absolute_effect_error_h","mean"),maximum_error_h=("absolute_effect_error_h","max"),
        n_nonzero=("nonzero","sum"),correct_signs=("correct_sign","sum"),recoveries=("recovered","sum"),
        nonzero_mae_h=("nonzero_error_h","mean"),zero_mae_h=("zero_error_h","mean")).reset_index()
    individual=cells.groupby(["method","length"],sort=False).agg(n=("trace_id","size"),n_estimated=("estimate_available","sum"),
        n_observed_min=("n_observed","min"),n_observed_max=("n_observed","max"),period_mae_h=("absolute_period_error_h","mean")).reset_index()
    experiments.to_csv(folder/(tag+"_experiments.csv"),index=False,encoding="utf-8")
    pairs.to_csv(folder/(tag+"_pairs.csv"),index=False,encoding="utf-8")
    individual.to_csv(folder/(tag+"_individual.csv"),index=False,encoding="utf-8")
elif args.dataset in {"single","dual"}:
    cells["absolute_reference_error_h"]=pd.Series(np.where(cells.estimate_available,abs(cells.estimate_h-cells.reference_period_h),np.inf)).where(cells.qualified&cells.applicable)
    cells["bad2h"]=(cells.absolute_reference_error_h>CONFIG["individual_tolerance_h"])&cells.qualified
    summary=cells.groupby("method",sort=False).agg(n_reference=("qualified","sum"),
        period_mae_h=("absolute_reference_error_h","mean"),errors_above_2h=("bad2h","sum")).reset_index()
    summary["errors_above_2h"]=summary.errors_above_2h.where(summary.method.map(coverage.set_index("method").applicable))
else:
    keys=["method","genotype","slice","phase","channel","length","alpha_index"]
    experiments=cells.groupby(keys,sort=False).agg(n_input=("trace_id","size"),n_estimated=("estimate_available","sum"),estimate_mean_h=("estimate_h","mean"),
        n_observed_min=("n_observed","min"),n_observed_max=("n_observed","max"),applicable=("applicable","all")).reset_index()
    experiments["estimate_mean_h"]=experiments.estimate_mean_h.where(experiments.n_input.eq(experiments.n_estimated))
    pair_keys=["method","genotype","slice","phase","channel","length"]
    wide=experiments.pivot(index=pair_keys,columns="alpha_index",values="estimate_mean_h")
    pairs=(wide[1]-wide[0]).rename("false_response_h").reset_index()
    pairs=pairs.merge(coverage[["method","applicable"]],on="method",validate="many_to_one")
    pairs["absolute_false_response_h"]=pd.Series(np.where(np.isfinite(pairs.false_response_h),abs(pairs.false_response_h),np.inf)).where(pairs.applicable)
    summary=pairs.groupby(["method","length"],sort=False).agg(n_pairs=("slice","size"),
        false_response_mae_h=("absolute_false_response_h","mean"),false_response_max_h=("absolute_false_response_h","max")).reset_index()
    experiments.to_csv(folder/(tag+"_experiments.csv"),index=False,encoding="utf-8")
    pairs.to_csv(folder/(tag+"_pairs.csv"),index=False,encoding="utf-8")
summary=summary.merge(coverage,on="method",validate="many_to_one")
cells.to_csv(folder/(tag+"_cells.csv"),index=False,encoding="utf-8")
summary.to_csv(folder/(tag+"_summary.csv"),index=False,encoding="utf-8")
print("UNIFIED_EXTERNAL_COMPLETE",tag,len(names),len(methods),flush=True)
print(summary.to_string(index=False),flush=True)
