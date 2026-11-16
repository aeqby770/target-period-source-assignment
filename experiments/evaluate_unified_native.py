"""Fresh native predictions and unchanged original endpoint aggregation."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1],"results":"results/0.3.1",
    "input":"data/sota_dev/recomputed_clean/lemna_task_period_acf_input.parquet",
    "context":"data/sota_dev/recomputed_clean/lemna_task_period_supervised_context.npz",
    "point_predictions":"data/sota_dev/recomputed_clean/lemna_task_period_classical_predictions.npz",
    "legacy_default":"data/sota_dev/recomputed_clean/lemna_task_period_actual_source_assignment_common_baseline_fit.npz",
    "legacy_frequency":"data/sota_dev/recomputed_clean/lemna_task_period_actual_source_assignment_frequency_geometry_fit.npz",
    "official_input":"data/sota_dev/competitors/official_lemna_dual_input.tsv"},
    "threads":os.cpu_count(),"plot":{}}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
import sys,ast,json,argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update(CONFIG["plot"])
root=CONFIG["paths"]["root"]
sys.path.insert(0,str(root/"software"))
from source_assignment_unified import feature_bundle,prediction_matrix,MODEL_KIND,SOURCE_OPERATOR,CORE_POLICY,predict_general_source_roles,predict_latent_representation_density,reference_source_observations,predict_reference_source_w2,predict_reference_calibration_mixture
from source_assignment_core import fit_monotone_mean,interpolate_mean
from period_support import project_periods_with_mean
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model",type=Path)
parser.add_argument("--output-dir",type=Path)
parser.add_argument("--feature-cache",type=Path)
parser.add_argument("--comparison-model",type=Path,action="append",default=[])
parser.add_argument("--stage",choices=["evaluate","readback"],default="evaluate",help="Readback checks model API only; full Native endpoints require all 30 cases")
parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
parser.add_argument("--refresh-model-cache",action="store_true",help="Refresh only ancillary prediction arrays from this same fresh API evaluation")
args=parser.parse_args()
folder=args.output_dir or root/CONFIG["paths"]["results"]
folder.mkdir(parents=True,exist_ok=True)
model_path=args.model or folder/"unified_model.npz"
data=dict(np.load(root/CONFIG["paths"]["context"],allow_pickle=False))
model=dict(np.load(model_path,allow_pickle=False))
frame=pd.read_parquet(root/CONFIG["paths"]["input"])
conditional=str(model.get("readout_kind",""))=="conditional_latent_representation_density"
reference_mixture=str(model.get("readout_kind",""))=="information_conditioned_reference_W2"
reference_w2=str(model.get("readout_kind","")) in {"reference_source_bounded_W2","information_conditioned_reference_W2"}
if args.stage=="readback" and not reference_w2:
    parser.error("--stage readback is defined for reference_source_bounded_W2")
if reference_w2:
    assert str(model["schema"])=="source_assignment_unified_0.3.1"
    available_cases=np.asarray(model["case_index"],dtype=int)
    assert len(np.unique(available_cases))==len(available_cases)
    complete_cases=np.array_equal(np.sort(available_cases),np.arange(30))
    if args.stage=="evaluate" and not complete_cases:
        print("NATIVE_EVALUATION_NOT_RUN",json.dumps({"status":"not_run","reason":"Full Native endpoints require case_index 0..29",
            "available_case_index":available_cases.tolist()}),flush=True)
        raise SystemExit(2)
    case_rows=pd.Index(available_cases).get_indexer(np.arange(30)) if complete_cases else np.arange(len(available_cases))
    if reference_mixture:
        assert args.feature_cache is None,"Reference mixtures require their explicit per-calibration No observations."
        calibration_bundles=list(map(lambda r:reference_source_observations(frame,model["period"],
            json.loads(str(model["calibration_representation_options_json"][r])),
            fundamental_support=bool(model["calibration_fundamental_support"][r]),device=args.device),
            np.arange(len(model["calibration_representation_options_json"]))))
        bundle=calibration_bundles[0]
        value=predict_reference_calibration_mixture(frame,model,cases=case_rows,device=args.device,
            include_diagnostics=True,source_bundles=calibration_bundles)
    else:
        options=json.loads(str(model["representation_options_json"]))
        support=bool(model.get("fundamental_support",True))
        if args.feature_cache is None:
            bundle=reference_source_observations(frame,model["period"],options,fundamental_support=support,device=args.device)
        else:
            bundle=dict(np.load(args.feature_cache,allow_pickle=False))
            assert json.loads(str(bundle["representation_options_json"]))==options
            assert bool(bundle["fundamental_support_enabled"])==support
        value=predict_reference_source_w2(bundle,model,cases=case_rows,device=args.device)
    actual=frame.pivot(index="trace_id",columns="x",values="signal_y").sort_index(axis=1)
    np.testing.assert_array_equal(bundle["trace_id"],actual.index.to_numpy(dtype=str))
    np.testing.assert_array_equal(bundle["time"],actual.columns.to_numpy(dtype=float))
    np.testing.assert_array_equal(bundle["mask"],actual.notna().to_numpy())
    np.testing.assert_array_equal(bundle["count"],actual.notna().sum(1).to_numpy())
    np.testing.assert_array_equal(bundle["period"],model["period"])
    np.testing.assert_array_equal(bundle["trace_id"],model["trace_id"])
    readback={key:float(np.max(abs(value[key]-model[key][case_rows]))) for key in ["estimate","source_period","group_mean","deviation"]}
    relations={"probability":float(np.max(abs(value["posterior"].sum(-1)-1.))),
        "expectation_h":float(np.max(abs(value["posterior"]@bundle["period"]-value["estimate"]))),
        "mean_h":float(np.max(abs(value["estimate"]@bundle["average"].T-value["group_mean"]))),
        "target_h":float(np.max(abs(value["group_mean"]-value["calibrated_mean_target"])))}
    assert max(readback.values())<1e-6 and relations["probability"]<1e-10 and max(v for k,v in relations.items() if k!="probability")<1e-6
    models=[model]+list(map(lambda p:dict(np.load(p,allow_pickle=False)),args.comparison_model))
    names=np.asarray(["unified"]+[str(m.get("control_name",p.stem)) for m,p in zip(models[1:],args.comparison_model)])
    assert len(np.unique(names))==len(names)
    values=[value]
    for other in models[1:]:
        assert str(other["schema"])=="source_assignment_unified_0.3.1" and str(other["readout_kind"]) in {"reference_source_bounded_W2","information_conditioned_reference_W2"}
        rows=pd.Index(other["case_index"]).get_indexer(available_cases[case_rows])
        assert np.all(rows>=0)
        np.testing.assert_array_equal(other["period"],bundle["period"])
        np.testing.assert_array_equal(other["trace_id"],bundle["trace_id"])
        np.testing.assert_array_equal(other["training_run"][rows],model["training_run"][case_rows])
        if str(other["readout_kind"])=="information_conditioned_reference_W2":
            same_options=reference_mixture and np.array_equal(other["calibration_representation_options_json"],model["calibration_representation_options_json"])
            same_support=reference_mixture and np.array_equal(other["calibration_fundamental_support"],model["calibration_fundamental_support"])
            values.append(predict_reference_calibration_mixture(frame,other,cases=rows,device=args.device,
                include_diagnostics=True,source_bundles=calibration_bundles if same_options and same_support else None))
        else:
            other_options=json.loads(str(other["representation_options_json"]))
            measured=bundle if not reference_mixture and other_options==options and bool(other.get("fundamental_support",True))==support else reference_source_observations(
                frame,other["period"],other_options,fundamental_support=bool(other.get("fundamental_support",True)),device=args.device)
            values.append(predict_reference_source_w2(measured,other,cases=rows,device=args.device))
    if args.refresh_model_cache:
        assert np.array_equal(available_cases[case_rows],np.arange(30)), "Cache refresh requires the complete original 30 cases."
        output_fields=["posterior","estimate","source_period","group_mean","deviation","cohort_probability","source_information",            "reference_calibration_weight","reference_calibration_probability","reference_calibration_mean",            "source_assignment_probability","raw_individual_probability","calibrated_mean_target"]
        for path,current,observed in zip([model_path]+args.comparison_model,models,values):
            parameters={key:current[key].copy() for key in current if key.startswith("calibration_") or key in ["reference_information_knots","training_run","case_index","reference_supervision_mask"]}
            rows=pd.Index(current["case_index"]).get_indexer(np.arange(30))
            for key in output_fields:
                if key not in observed:
                    continue
                fresh_array=np.asarray(observed[key])
                if key=="source_information":
                    current[key]=fresh_array
                else:
                    stored=np.asarray(current.get(key,fresh_array)).copy()
                    stored[rows]=fresh_array
                    current[key]=stored
            current["prediction_cache_policy"]=np.asarray("fresh_public_reference_API_same_parameter_arrays_final_No_producer")
            for key,old in parameters.items():
                np.testing.assert_array_equal(current[key],old)
            temporary=path.with_name(path.stem+"_prediction_cache.tmp.npz")
            np.savez_compressed(temporary,**current);temporary.replace(path)
            print("NATIVE_ANCILLARY_CACHE_REFRESHED",path,"parameters_unchanged",True,flush=True)
        alias=folder/"native_reference_calibration_model.npz"
        if reference_mixture and alias!=model_path:
            copy=dict(np.load(alias,allow_pickle=False))
            np.testing.assert_array_equal(copy["calibration_core_coefficient"],model["calibration_core_coefficient"])
            np.testing.assert_array_equal(copy["calibration_role_coefficient"],model["calibration_role_coefficient"])
            copy.update({key:model[key] for key in output_fields if key in model})
            copy["prediction_cache_policy"]=model["prediction_cache_policy"]
            temporary=alias.with_name(alias.stem+"_prediction_cache.tmp.npz");np.savez_compressed(temporary,**copy);temporary.replace(alias)
    if args.stage=="readback":
        reports=[]
        for name,other,observed_value in zip(names,models,values):
            rows=pd.Index(other["case_index"]).get_indexer(available_cases[case_rows])
            errors={key:float(np.max(abs(observed_value[key]-other[key][rows]))) for key in ["estimate","source_period","group_mean","deviation"]}
            probability_error=float(np.max(abs(observed_value["posterior"].sum(-1)-1.)))
            expectation_error=float(np.max(abs(observed_value["posterior"]@bundle["period"]-observed_value["estimate"])))
            mean_error=float(np.max(abs(observed_value["estimate"]@bundle["average"].T-observed_value["group_mean"])))
            target_error=float(np.max(abs(observed_value["group_mean"]-observed_value["calibrated_mean_target"])))
            assert max(errors.values())<1e-6 and probability_error<1e-10 and max(expectation_error,mean_error,target_error)<1e-6
            reports.append({"method":"task_source_assignment_"+str(name),"case_index":available_cases[case_rows].tolist(),
                "readback_error_h":errors,"probability_error":probability_error,"expectation_error_h":expectation_error,
                "mean_error_h":mean_error,"W2_target_error_h":target_error})
        print("NATIVE_REFERENCE_SOURCE_API_READBACK",json.dumps({"records":len(bundle["trace_id"]),
            "full_30_validation":"not_run","results":reports}),flush=True)
        raise SystemExit(0)
    fresh={key:np.stack([v[key] for v in values]) for key in ["estimate","source_period","group_mean"]}
    fresh["original_period"]=np.stack([v.get("original_period",np.full_like(v["estimate"],np.nan)) for v in values])
    fresh["population_location"]=np.full_like(fresh["group_mean"],np.nan)
    fresh["mean_probability"]=np.full((*fresh["group_mean"].shape,len(bundle["period"])),np.nan)
    fresh["methods"]=names
    trace_order=pd.Index(data["trace_id"]).get_indexer(bundle["trace_id"])
    training=model["training_run"][case_rows]
    fitted={"trace_id":bundle["trace_id"],"run_names":bundle["run_names"],"training_run":training,
        "training_mask":training[:,bundle["run_index"]]&data["qualified"][trace_order][None],
        "estimate":np.stack([m["estimate"][pd.Index(m["case_index"]).get_indexer(np.arange(30))] for m in models]),
        "optimizer_success_by_method":np.asarray([m["success"].all() for m in models]),
        "optimizer_success_by_case":np.stack([m["success"][pd.Index(m["case_index"]).get_indexer(np.arange(30))] for m in models]),
        "training_scope":np.asarray("pooled")}
    model=dict(model,methods=names,parameter_policy=np.asarray([str(m["parameter_policy"]) for m in models]),
        method_applicable=np.ones(len(models),dtype=bool),preferred_variant=np.asarray(0),preferred_method=np.asarray("unified"))
elif conditional or str(model.get("schema",""))=="source_assignment_unified_0.3.1":
    cache=args.feature_cache or folder/("native_observed_features.npz" if conditional or str(model["source_representation"])=="observed_physical_peer" else "native_features.npz")
    bundle=dict(np.load(cache,allow_pickle=False))
    assert conditional or str(bundle["source_representation"])==str(model["source_representation"])
    np.testing.assert_array_equal(bundle["trace_id"],model["trace_id"])
    observed=frame.groupby("trace_id").size().reindex(bundle["trace_id"])
    np.testing.assert_array_equal(bundle["count"],observed.to_numpy())
    np.testing.assert_array_equal(bundle["time"],np.sort(frame.x.unique()))
    if conditional:
        balanced=dict(np.load(folder/"native_features.npz",allow_pickle=False))
        value=predict_latent_representation_density(bundle,balanced,model)
        value["original_period"]=np.full_like(value["estimate"],np.nan)
    else:
        value=predict_general_source_roles(bundle,model)
    fresh={key:value[key][None] for key in ["estimate","group_mean","source_period","original_period"]}
    fresh["population_location"]=np.full_like(fresh["group_mean"],np.nan)
    fresh["mean_probability"]=np.full((*fresh["group_mean"].shape,len(bundle["period"])),np.nan)
    fresh["methods"]=np.asarray(["source_roles"])
    trace_order=pd.Index(data["trace_id"]).get_indexer(bundle["trace_id"])
    fitted={"trace_id":bundle["trace_id"],"run_names":bundle["run_names"],"training_run":model["training_run"],
        "training_mask":model["training_run"][:,bundle["run_index"]]&data["qualified"][trace_order][None],
        "estimate":model["estimate"][None],"optimizer_success_by_method":np.asarray([model["success"].all()]),
        "optimizer_success_by_case":model["success"][None],"training_scope":np.asarray("pooled")}
    if conditional:
        np.testing.assert_array_equal(model["reference_source_count"],fitted["training_mask"].sum(1))
    model=dict(model,methods=fresh["methods"],parameter_policy=np.asarray(["conditional_source_density_joint_objective" if conditional else "shared_measurement_density_and_source_references"]),
        method_applicable=np.asarray([True]),preferred_variant=np.asarray(0),preferred_method=np.asarray("source_roles"),
        model_kind=np.asarray("conditional_latent_representation_density" if conditional else str(model["model_kind"])),
        source_operator=np.asarray("observed_and_amplitude_balanced_probability_mixture" if conditional else str(model["source_operator"])),
        source_core_policy=np.asarray("conditional_source_density" if conditional else "observed_measurement_source_roles"))
else:
    fitted=dict(np.load(folder/"native_fit.npz",allow_pickle=False))
    assert str(model["model_kind"])==str(fitted["model_kind"])==MODEL_KIND
    assert str(model["source_operator"])==str(fitted["source_operator"])==SOURCE_OPERATOR
    assert str(model["source_core_policy"])==str(fitted["source_core_policy"])==CORE_POLICY
    bundle=feature_bundle(frame,include_controls=len(model["methods"])>1)
    fresh=prediction_matrix(bundle,model,np.arange(model["anchor_coefficient"].shape[1]))
training_order=pd.Index(fitted["run_names"]).get_indexer(data["run_name"])
data["outer_train_run"]=fitted["training_run"][1:,training_order]
assert not np.any(fitted["training_mask"][1:][:,pd.Index(fitted["trace_id"]).get_indexer(data["trace_id"])]&data["outer_query"])
original_order=pd.Index(fitted["trace_id"]).get_indexer(bundle["trace_id"])
assert np.all(original_order>=0)
readback=float(np.max(np.abs(fresh["estimate"]-fitted["estimate"][:,:,original_order])))
assert readback<1e-6,readback
order=pd.Index(bundle["trace_id"]).get_indexer(data["trace_id"])
estimate=fresh["estimate"][:,1:,order].transpose(1,2,0)
methods=np.char.add("task_source_assignment_",fresh["methods"])
unconverged_methods=methods[~fitted["optimizer_success_by_method"]]
model_methods=methods.copy()
method_info=pd.DataFrame({"method":methods,"comparison_group":"current_model",
    "prediction_source":[Path(os.path.relpath(p,root)).as_posix() for p in [model_path]+args.comparison_model] if reference_w2 else Path(os.path.relpath(model_path,root)).as_posix(),
    "training_information":"original_source_references_under_training_mask" if reference_w2 else "current_shared_training_views_and_exclusions",
    "fit_policy":model["parameter_policy"],
    "fit_status":np.where(fitted["optimizer_success_by_method"],"completed","not_converged"),
    "applicable":model["method_applicable"]})
# The API export is the unchanged measured input, not a calibrated period output.
official=pd.read_csv(root/CONFIG["paths"]["official_input"],sep="\t",header=None,
    names=["trace_id","x","signal_y"],float_precision="round_trip")
pd.testing.assert_frame_equal(frame[["trace_id","x","signal_y"]].sort_values(["trace_id","x"]).reset_index(drop=True),
    official.sort_values(["trace_id","x"]).reset_index(drop=True),check_dtype=False,check_exact=True)
point=np.load(root/CONFIG["paths"]["point_predictions"],allow_pickle=False)
point_order=pd.Index(point["trace_id"]).get_indexer(data["trace_id"])
assert np.all(point_order>=0) and len(point_order)==len(point["trace_id"])
assert "saved_No_only_point_predictions" in str(point["information"])
assert not np.char.startswith(point["native_methods"],"target_mean__").any()
point_index=pd.Index(point["methods"]).get_indexer(point["native_methods"])
point_estimate=point["estimate"][:,point_order][:,:,point_index]
np.testing.assert_array_equal(point_estimate,np.broadcast_to(point_estimate[:1],point_estimate.shape))
h1=bundle["period"][bundle["identity_base_features"][:,:,0].argmax(axis=1)][order]
legacy=np.load(root/CONFIG["paths"]["legacy_default"],allow_pickle=False)
frequency=np.load(root/CONFIG["paths"]["legacy_frequency"],allow_pickle=False)
legacy_order=pd.Index(legacy["trace_id"]).get_indexer(data["trace_id"])
frequency_order=pd.Index(frequency["trace_id"]).get_indexer(data["trace_id"])
assert np.all(legacy_order>=0) and np.all(frequency_order>=0)
assert str(legacy["software_version"])==str(frequency["software_version"])=="0.2.4"
np.testing.assert_array_equal(legacy["inference_mask"][:,legacy_order],data["outer_query"])
np.testing.assert_array_equal(frequency["inference_mask"][:,frequency_order],data["outer_query"])
baseline_methods=np.asarray(["unlabelled_H1"]+np.char.add("native_",point["native_methods"]).tolist()+["legacy_0_2_4","legacy_frequency"])
baseline_estimate=np.concatenate([np.broadcast_to(h1[None,:,None],(len(estimate),len(h1),1)),point_estimate,
    legacy["estimate"][:,legacy_order],frequency["estimate"][:,frequency_order]],axis=-1)
estimate=np.concatenate([estimate,baseline_estimate],axis=-1)
methods=np.concatenate([methods,baseline_methods])
method_info=pd.concat([method_info,pd.DataFrame({"method":baseline_methods,
    "comparison_group":["unlabelled_or_frozen_point"]*(1+len(point_index))+["legacy_original_protocol"]*2,
    "prediction_source":["current_native_timestamps_12_48_H1"]+[CONFIG["paths"]["point_predictions"]]*len(point_index)+[CONFIG["paths"]["legacy_default"],CONFIG["paths"]["legacy_frequency"]],
    "training_information":["none"]+["saved_point_method_no_current_optical_labels"]*len(point_index)+["original_0.2.4_labels_maps_and_exclusions"]*2,
    "fit_status":["not_fitted"]*(1+len(point_index))+["legacy_frozen"]*2,"applicable":True})],ignore_index=True)
observed_count=pd.Series(bundle["count"],index=bundle["trace_id"])
matched_names=np.asarray(["matched_mean_MESA","matched_mean_MFourFit","matched_mean_H1"])
matched_index=pd.Index(point["native_methods"]).get_indexer(["MESA","MFourFit"])
assert np.all(matched_index>=0)
matched_points=np.stack([point_estimate[0,:,matched_index[0]],point_estimate[0,:,matched_index[1]],h1])
member=data["run_membership"];average=member/member.sum(1)[:,None];run=data["run_index"]
available=np.isfinite(matched_points)
complete=np.einsum("gn,mn->mg",member,available.astype(float),optimize=True)==member.sum(1)[None]
point_mean=np.einsum("mn,gn->mg",np.nan_to_num(matched_points,nan=0.),average,optimize=True)
label=np.where(data["qualified"],data["target"],0.)
reference_mean=(member@label)/np.maximum(member@data["qualified"],1.)
training=np.concatenate([np.ones((1,len(member)),dtype=bool),data["outer_train_run"]],axis=0)
allowed=training[None]&complete[:,None]&((member@data["qualified"])>0)[None,None]
parent_means=np.broadcast_to(point_mean[:,None],allowed.shape)
_,cal_x,cal_y,cal_n=fit_monotone_mean(parent_means,reference_mean,allowed)
assert (cal_n>0).all()
calibrated=interpolate_mean(parent_means,cal_x,cal_y,cal_n,identity_tails=True)
deviation=matched_points-point_mean[:,run]
target_deviation=label-reference_mean[run]
train=allowed[:,:,run]&data["qualified"][None,None]
n=train.sum(-1);weight=train/n[:,:,None]
variance=np.maximum(np.einsum("mfn,n->mf",weight,label**2,optimize=True)-np.einsum("mfn,n->mf",weight,label,optimize=True)**2,
    np.diff(bundle["period"])[0]**2/12)
clean_deviation=np.nan_to_num(deviation,nan=0.)
within=np.maximum((np.einsum("mfn,mn,n->mf",weight,clean_deviation,target_deviation,optimize=True)+variance/n)/
    (np.einsum("mfn,mn->mf",weight,clean_deviation**2,optimize=True)+variance/n),0.)
mean=np.clip(calibrated,bundle["period"].min(),bundle["period"].max())
unbounded=mean[:,:,run]+within[:,:,None]*clean_deviation[:,None]
bounded=project_periods_with_mean(unbounded.reshape(-1,len(run)),mean.reshape(-1,len(member)),member,
    bundle["period"].min(),bundle["period"].max()).reshape(unbounded.shape)
matched_estimate=np.where(complete[:,None,run]&available[:,None],bounded,np.nan)
estimate=np.concatenate([estimate,matched_estimate[:,1:].transpose(1,2,0)],axis=-1)
methods=np.concatenate([methods,matched_names])
baseline_methods=np.concatenate([baseline_methods,matched_names])
method_info=pd.concat([method_info,pd.DataFrame({"method":matched_names,"comparison_group":"current_target_calibrated_point",
    "prediction_source":[CONFIG["paths"]["point_predictions"],CONFIG["paths"]["point_predictions"],"current_actual_timestamps_12_48_H1"],
    "training_information":"current_"+str(fitted.get("training_scope","legacy_layout_control"))+"_training_target_period_labels_only",
    "fit_policy":"equal_experiment_PAV_identity_tails_closed_form_centered_deviation_support_projection",
    "fit_status":"completed","applicable":True})],ignore_index=True)
np.savez_compressed(folder/"native_matched_mean_parameters.npz",methods=matched_names,trace_id=data["trace_id"],
    calibration_prediction=cal_x,calibration_mean=cal_y,calibration_experiments=cal_n,within_scale=within,
    training_run=allowed,training_mask=train,complete_experiment=complete,
    native_methods=np.asarray(["MESA","MFourFit","unlabelled_H1"]),period=bundle["period"],
    training_scope=fitted.get("training_scope",np.asarray("legacy_layout_control")))
case_success=fitted["optimizer_success_by_case"]
case_status=np.where(case_success[:,1:],"completed","not_converged")
baseline_status=np.broadcast_to(method_info.set_index("method").loc[baseline_methods,"fit_status"].to_numpy()[:,None],(len(baseline_methods),len(data["outer_protocol"])))
case_status=np.concatenate([case_status,baseline_status],axis=0)
case_fit_info=pd.DataFrame({"method":np.repeat(methods,len(data["outer_protocol"])),
    "case":np.tile(np.arange(len(data["outer_protocol"])),len(methods)),"fit_status":case_status.ravel()})
protocol_fit_info=case_fit_info.assign(protocol=data["outer_protocol"][case_fit_info.case],
    failed=case_fit_info.fit_status.eq("not_converged")).groupby(["protocol","method"],sort=False).agg(
        n_fit_cases=("case","size"),n_failed_fit_cases=("failed","sum")).reset_index()
protocol_fit_info=protocol_fit_info.merge(method_info[["method","fit_status"]].rename(columns={"fit_status":"original_fit_status"}),on="method",validate="many_to_one")
protocol_fit_info["fit_status"]=np.where(protocol_fit_info.method.isin(np.r_[model_methods,matched_names]),
    np.where(protocol_fit_info.n_failed_fit_cases.eq(0),"completed","not_converged"),protocol_fit_info.original_fit_status)
protocol_fit_info["n_fit_cases"]=np.where(protocol_fit_info.method.isin(np.r_[model_methods,matched_names]),protocol_fit_info.n_fit_cases,0)
protocol_fit_info=protocol_fit_info.drop(columns="original_fit_status")
method_info=method_info.rename(columns={"fit_status":"overall_fit_status"})
application={"paths":{"lemna_reference":root/"data/external/lemna_watanabe2023/dual_reporter_reference_cells.csv"},
    "task_period":{"control_condition":"control"},"risk_tolerance_h":2.}
stem=folder/"native"
layer_method,layer_case,layer_run=np.indices(fresh["group_mean"][:,1:].shape)
cohort_layer=pd.DataFrame({"case":layer_case.ravel(),"method":model_methods[layer_method.ravel()],
    "run":bundle["run_names"][layer_run.ravel()],
    "finite_cohort_mean_h":fresh["group_mean"][:,1:].ravel(),
    "population_location_h":fresh["population_location"][:,1:].ravel()})
trace_run=frame[["trace_id","run"]].drop_duplicates().set_index("trace_id").run
tree=ast.parse((root/"software/sota_marked_application.py").read_text(encoding="utf-8"))
evaluation=next(node.value.value for node in tree.body if isinstance(node,ast.Assign)
    and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="task_transfer_report_stage")
evaluation=evaluation.replace('assert np.isfinite(predictions.estimate_h).all()',
    'predictions["estimate_available"]=np.isfinite(predictions.estimate_h)\npredictions=predictions.merge(method_info,on="method",validate="many_to_one")\npredictions["n_observed"]=predictions.trace_id.map(observed_count)')
evaluation=evaluation.replace('predictions.to_csv(',
    'predictions["run"]=predictions.trace_id.map(trace_run)\npredictions=predictions.merge(cohort_layer,on=["case","method","run"],how="left",validate="many_to_one")\nexperiment_layer=predictions[["protocol","comparison","method","run","case","finite_cohort_mean_h","population_location_h"]].drop_duplicates().groupby(["protocol","comparison","method","run"],sort=False)[["finite_cohort_mean_h","population_location_h"]].mean().reset_index()\npredictions.to_csv(')
evaluation=evaluation.replace('estimate_h=("estimate_h","mean"),n_models=("case","size")).reset_index()',
    'estimate_h=("estimate_h","mean"),n_models=("case","size"),n_estimated_models=("estimate_h","count")).reset_index()\ncells["estimate_h"]=cells.estimate_h.where(cells.n_estimated_models.eq(cells.n_models))\ncells=cells.merge(method_info,on="method",validate="many_to_one")\ncells["n_observed"]=cells.trace_id.map(observed_count)')
evaluation=evaluation.replace('cells["absolute_difference_h"] = (cells.estimate_h-cells.reference_period_h).abs().where(cells.reference_usable)',
    'cells["available_absolute_difference_h"]=(cells.estimate_h-cells.reference_period_h).abs().where(cells.reference_usable)\ncells["absolute_difference_h"]=pd.Series(np.where(np.isfinite(cells.estimate_h),abs(cells.estimate_h-cells.reference_period_h),np.inf)).where(cells.reference_usable)')
evaluation=evaluation.replace('mae_h=("absolute_difference_h","mean")).reset_index()',
    'mae_h=("absolute_difference_h","mean"),valid_mae_h=("available_absolute_difference_h","mean"),n_reference_estimated=("estimate_h","count")).reset_index()')
evaluation=evaluation.replace('individual.to_csv(',
    'coverage=cells.loc[cells.protocol.eq("leave_experiment")|cells.query_role.eq("treatment")].groupby(["protocol","method"]).agg(n_input=("trace_id","size"),n_estimated=("estimate_h","count")).reset_index()\nindividual=individual.merge(coverage,on=["protocol","method"],validate="one_to_one").merge(method_info,on="method",validate="many_to_one")\nindividual.to_csv(')
evaluation=evaluation.replace('n_input=("trace_id","size"),estimate_mean_h=("estimate_h","mean")).reset_index()',
    'n_input=("trace_id","size"),n_estimated=("estimate_h","count"),estimate_mean_h=("estimate_h","mean")).reset_index()\nexperiments["estimate_mean_h"]=experiments.estimate_mean_h.where(experiments.n_estimated.eq(experiments.n_input))')
evaluation=evaluation.replace('n_experiments=("run","size"),n_input=("n_input","sum"),',
    'n_experiments=("run","size"),n_complete_experiments=("estimate_mean_h","count"),n_input=("n_input","sum"),')
evaluation=evaluation.replace('control = conditions.loc[',
    'conditions["estimate_mean_h"]=conditions.estimate_mean_h.where(conditions.n_complete_experiments.eq(conditions.n_experiments))\ncontrol = conditions.loc[')
evaluation=evaluation.replace('effects["absolute_effect_error_h"] = effects.effect_error_h.abs()',
    'effects["absolute_effect_error_h"]=np.where(np.isfinite(effects.estimate_effect_h),effects.effect_error_h.abs(),np.inf)')
evaluation=evaluation.replace('effect_summary["fit_status"] = np.where(effect_summary.method.isin(unconverged_methods),"not_converged","completed")',
    'effect_summary=effect_summary.merge(method_info,on="method",validate="many_to_one")\nsame_length_summary=same_length_summary.merge(method_info,on="method",validate="many_to_one")')
evaluation=evaluation.replace('data["outer_train_run"].sum(axis=1)','fitted["training_run"][1:].sum(axis=1)').replace(
    'data["case_train"][:len(data["outer_protocol"])].sum(axis=1)','fitted["training_mask"][1:].sum(axis=1)')
evaluation=evaluation.replace('predictions=predictions.merge(method_info,on="method",validate="many_to_one")',
    'predictions=predictions.merge(method_info,on="method",validate="many_to_one").merge(case_fit_info,on=["case","method"],validate="many_to_one")\npredictions["fit_not_converged"]=predictions.fit_status.eq("not_converged")')
evaluation=evaluation.replace('n_estimated_models=("estimate_h","count")',
    'n_estimated_models=("estimate_h","count"),n_failed_models=("fit_not_converged","sum")')
evaluation=evaluation.replace('cells=cells.merge(method_info,on="method",validate="many_to_one")',
    'cells=cells.merge(method_info,on="method",validate="many_to_one")\ncells["fit_status"]=np.where(cells.method.isin(model_methods),np.where(cells.n_failed_models.eq(0),"completed","not_converged"),cells.overall_fit_status)\ncells["fit_not_converged"]=cells.fit_status.eq("not_converged")')
evaluation=evaluation.replace('individual.to_csv(',
    'individual=individual.merge(protocol_fit_info,on=["protocol","method"],validate="one_to_one")\nindividual.to_csv(')
evaluation=evaluation.replace('individual_condition.to_csv(',
    'individual_condition=individual_condition.merge(protocol_fit_info,on=["protocol","method"],validate="many_to_one")\nindividual_condition.to_csv(')
evaluation=evaluation.replace('n_estimated=("estimate_h","count"),estimate_mean_h=("estimate_h","mean")',
    'n_estimated=("estimate_h","count"),n_failed_predictions=("fit_not_converged","sum"),estimate_mean_h=("estimate_h","mean")')
evaluation=evaluation.replace('n_complete_experiments=("estimate_mean_h","count"),',
    'n_complete_experiments=("estimate_mean_h","count"),n_failed_predictions=("n_failed_predictions","sum"),')
evaluation=evaluation.replace('keys+["estimate_mean_h","reference_mean_h"]].rename(columns={',
    'keys+["estimate_mean_h","reference_mean_h","n_failed_predictions"]].rename(columns={"n_failed_predictions":"control_failed_predictions",')
evaluation=evaluation.replace('experiments.to_csv(',
    'experiments=experiments.merge(method_info,on="method",validate="many_to_one").merge(experiment_layer,on=["protocol","comparison","method","run"],how="left",validate="many_to_one")\nexperiments["fit_status"]=np.where(experiments.method.isin(model_methods),np.where((experiments.n_failed_predictions+experiments.control_failed_predictions).eq(0),"completed","not_converged"),experiments.overall_fit_status)\nexperiments.to_csv(')
evaluation=evaluation.replace('effects.to_csv(',
    'effects=effects.merge(method_info,on="method",validate="many_to_one")\neffects["fit_status"]=np.where(effects.method.isin(model_methods),np.where((effects.n_failed_predictions+effects.control_failed_predictions).eq(0),"completed","not_converged"),effects.overall_fit_status)\neffects.to_csv(')
evaluation=evaluation.replace('effect_summary=effect_summary.merge(method_info,on="method",validate="many_to_one")',
    'effect_summary=effect_summary.merge(method_info,on="method",validate="many_to_one").merge(protocol_fit_info,on=["protocol","method"],validate="many_to_one")')
evaluation=evaluation.replace('same_length_summary=same_length_summary.merge(method_info,on="method",validate="many_to_one")',
    'same_length_summary=same_length_summary.merge(method_info,on="method",validate="many_to_one").merge(protocol_fit_info,on=["protocol","method"],validate="many_to_one")')
exec(compile(evaluation[evaluation.index("outer_i, trace_i ="):evaluation.index("complete_stem =")],
    "<unchanged_native_evaluation>","exec"))
np.savez_compressed(folder/"native_predictions.npz",estimate=estimate,methods=methods,
    preferred_variant=model["preferred_variant"],preferred_method=model["preferred_method"],
    trace_id=data["trace_id"],outer_query=data["outer_query"],group_mean=fresh["group_mean"],
    source_period=fresh["source_period"],original_period=fresh["original_period"],model_methods=model_methods,
    population_location=fresh["population_location"],mean_probability=fresh["mean_probability"],
    model_kind=model["model_kind"],source_operator=model["source_operator"],source_core_policy=model["source_core_policy"])
(folder/"native_prediction_readback.json").write_text(json.dumps({"fresh_prediction_vs_fit_max_h":readback,
    "records":len(bundle["trace_id"]),"contexts":len(data["outer_protocol"]),"methods":len(methods),
    "current_model_methods":len(model_methods),"baseline_methods":len(baseline_methods),
    "preferred_variant":int(model["preferred_variant"]),"preferred_method":str(model["preferred_method"]),
    "model_kind":str(model["model_kind"]),"source_operator":str(model["source_operator"]),"source_core_policy":str(model["source_core_policy"])},indent=2)+"\n",encoding="utf-8")
print("UNIFIED_NATIVE_INDIVIDUAL",individual.to_string(index=False),flush=True)
print("UNIFIED_NATIVE_RESPONSE",effect_summary.loc[effect_summary.readout.eq("all_input")].to_string(index=False),flush=True)
