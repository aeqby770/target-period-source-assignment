"""Validate the shared reference-source bounded-W2 entry on actual Native data."""
from pathlib import Path
import os,argparse
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1],
    "model":"results/0.3.1/unified_model.npz","output":"results/0.3.1/kernel_validation.json",
    "native":"data/sota_dev/recomputed_clean/lemna_task_period_acf_input.parquet",
    "context":"data/sota_dev/recomputed_clean/lemna_task_period_supervised_context.npz",
    "controlled":"data/sota_dev/final_model_controlled/inputs.parquet",
    "ecg_context":"data/external/adfecgdb/supervised_context.npz",
    "ecg_references":"data/external/adfecgdb/references.csv"},
    "threads":os.cpu_count(),"plot":{},"unit_factor":3600.,
    "period_tolerance_h":1e-6,"probability_tolerance":1e-10,"statistic_tolerance":1e-7,
    "boundary":{"mask_samples":48,"fit_lengths":[104,145,145],"fit_periods":[20.,22.,26.,28.],"fit_maxiter":900}}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["OPENBLAS_NUM_THREADS"]=str(CONFIG["threads"])
os.environ["MKL_NUM_THREADS"]=str(CONFIG["threads"])
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model",type=Path,default=Path(CONFIG["paths"]["model"]))
parser.add_argument("--feature-cache",type=Path,help="Matching reference-source cache with recorded representation options")
parser.add_argument("--output",type=Path,default=Path(CONFIG["paths"]["output"]))
parser.add_argument("--stage",choices=["full","readback"],default="full")
parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
args=parser.parse_args()
import sys,json,subprocess,datetime
import numpy as np
import pandas as pd
root=CONFIG["paths"]["root"]
sys.path.insert(0,str(root/"software"))
import source_assignment_unified as source
from source_assignment_unified import observations,reference_source_observations,predict_reference_source_w2,predict_reference_calibration_mixture
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
model_path=Path(os.path.abspath(root/args.model))
output=Path(os.path.abspath(root/args.output))
model=dict(np.load(model_path,allow_pickle=False))
assert str(model["schema"])=="source_assignment_unified_0.3.1"
reference_mixture=str(model["readout_kind"])=="information_conditioned_reference_W2"
assert str(model["readout_kind"]) in {"reference_source_bounded_W2","information_conditioned_reference_W2"}
assert str(model["time_unit"])=="h"
case_ids=np.asarray(model["case_index"],dtype=int)
assert len(np.unique(case_ids))==len(case_ids)
complete_cases=np.array_equal(np.sort(case_ids),np.arange(30))
if args.stage=="full" and not complete_cases:
    not_run={"passed":None,"status":"not_run","reason":"Full Native validation requires case_index 0..29",
        "model":str(model_path),"available_case_index":case_ids.tolist(),"full_30_validation":"not_run",
        "thresholds":{"period_h":CONFIG["period_tolerance_h"],"probability":CONFIG["probability_tolerance"],"statistics":CONFIG["statistic_tolerance"]}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(not_run,indent=2)+"\n",encoding="utf-8")
    print("UNIFIED_KERNEL_NOT_RUN",json.dumps(not_run),flush=True)
    raise SystemExit(2)
boundary_checks={}
if args.stage=="full":
    probe_time=np.arange(CONFIG["boundary"]["mask_samples"],dtype=float)
    probe_period=np.linspace(12.,48.,73)
    probe_signal=np.sin(2*np.pi*probe_time[None]/np.asarray([17.,19.,23.])[:,None]+np.asarray([0.,.4,1.])[:,None])
    probe_signal+=.2*np.cos(4*np.pi*probe_time[None]/np.asarray([17.,19.,23.])[:,None])
    probe_mask=np.ones_like(probe_signal,dtype=bool)
    probe_mask[0,1::3]=False;probe_mask[1,2::4]=False;probe_mask[2,3::5]=False
    probe_mask=np.concatenate([probe_mask,np.broadcast_to(probe_mask[:1],probe_mask.shape)])
    probe_signal=np.where(probe_mask,np.tile(probe_signal,(2,1)),0.)
    probe_run=np.repeat(np.arange(2),3);probe_member=(np.arange(2)[:,None]==probe_run[None]).astype(float)
    probe_data={"time":probe_time,"mask":probe_mask,"count":probe_mask.sum(1),"membership":probe_member,
        "average":probe_member/3.,"run_index":probe_run}
    probe_geometry=source._role_geometry(probe_signal,probe_data,probe_period,24.,normalize_reference_peer=True)
    expected_peer_rhs=np.einsum("npti,nt->npi",probe_geometry["design"][probe_geometry["layout"]],probe_geometry["peer"],optimize=True)
    peer_rhs_error=float(np.max(abs(probe_geometry["rhs"][:,:,:,1]-expected_peer_rhs)))
    peer_projection=source.detrend(probe_geometry["peer"],probe_time,probe_mask)
    peer_projection_error=float(np.max(abs(peer_projection-probe_geometry["peer"])))
    probe_feature=source.profile_features(probe_signal,probe_data,probe_period,24.,ridge_fraction=0.)[0]
    changed_feature=source.profile_features(probe_signal+1e6*(~probe_mask),probe_data,probe_period,24.,ridge_fraction=0.)[0]
    unobserved_error=float(np.max(abs(probe_feature-changed_feature)))
    boundary_checks["heterogeneous_mask"]={"records":6,"counts":probe_data["count"].tolist(),
        "peer_RHS_error":peer_rhs_error,"peer_projection_error":peer_projection_error,
        "unobserved_value_feature_error":unobserved_error,
        "passed":bool(max(peer_rhs_error,peer_projection_error)<CONFIG["probability_tolerance"] and unobserved_error==0.)}
    fit_lengths=np.repeat(np.asarray(CONFIG["boundary"]["fit_lengths"]),len(CONFIG["boundary"]["fit_periods"]))
    fit_reference=np.tile(np.asarray(CONFIG["boundary"]["fit_periods"]),len(CONFIG["boundary"]["fit_lengths"]))
    fit_time=np.arange(fit_lengths.max(),dtype=float)
    fit_run=np.repeat(np.arange(len(CONFIG["boundary"]["fit_lengths"])),len(CONFIG["boundary"]["fit_periods"]))
    fit_trace=np.char.add("boundary_",np.char.zfill(np.arange(len(fit_reference)).astype(str),2))
    fit_signal=np.sin(2*np.pi*fit_time[None]/fit_reference[:,None]+np.arange(len(fit_reference))[:,None]/7.)
    fit_signal+=.17*np.cos(4*np.pi*fit_time[None]/fit_reference[:,None])+.002*fit_time[None]
    fit_observed=fit_time[None]<fit_lengths[:,None]
    fit_frame=pd.DataFrame({"trace_id":np.repeat(fit_trace,len(fit_time)),"run":np.repeat(np.char.add("run_",fit_run.astype(str)),len(fit_time)),
        "condition":"reference","x":np.tile(fit_time,len(fit_reference)),"signal_y":fit_signal.ravel()}).loc[fit_observed.ravel()]
    fit_training=np.vstack([np.ones((1,len(CONFIG["boundary"]["fit_lengths"])),dtype=bool),~np.eye(len(CONFIG["boundary"]["fit_lengths"]),dtype=bool)])
    boundary_model=source.fit_reference_calibration_density(fit_frame,fit_reference,np.ones(len(fit_reference),dtype=bool),
        fit_training,np.char.add("unit_",fit_run.astype(str)),probe_period,maxiter=CONFIG["boundary"]["fit_maxiter"],device=args.device)
    boundary_checks["homogeneous_information_training_fold"]={"records":len(fit_reference),"training_contexts":len(fit_training),
        "calibrations":len(boundary_model["calibration_name"]),"reference_counts":boundary_model["calibration_reference_source_count"].tolist(),
        "unit_counts":boundary_model["calibration_reference_unit_count"].tolist(),"optimizer_success":boundary_model["success"].tolist(),
        "finite_predictions":bool(np.isfinite(boundary_model["estimate"]).all()),
        "passed":bool(len(boundary_model["calibration_name"])==1 and np.all(boundary_model["calibration_reference_source_count"]>0)
            and np.all(boundary_model["success"]) and np.isfinite(boundary_model["estimate"]).all())}
    print("UNIFIED_BOUNDARY_VALIDATION",json.dumps(boundary_checks),flush=True)
rows=pd.Index(case_ids).get_indexer(np.arange(30)) if complete_cases else np.arange(len(case_ids))
period=np.asarray(model["period"],dtype=float)
scale=float(model["period_scale"])
options=list(map(lambda value:json.loads(str(value)),model["calibration_representation_options_json"])) if reference_mixture else json.loads(str(model["representation_options_json"]))
fundamental_support=model["calibration_fundamental_support"].astype(bool) if reference_mixture else bool(model.get("fundamental_support",True))
native=pd.read_parquet(root/CONFIG["paths"]["native"])
actual=observations(native)
names=actual["metadata"].index.to_numpy(dtype=str)
assert len(names)==777 and len(actual["runs"])==17
context=dict(np.load(root/CONFIG["paths"]["context"],allow_pickle=False))
co=pd.Index(context["trace_id"]).get_indexer(names)
mo=pd.Index(model["trace_id"]).get_indexer(names)
assert np.all(co>=0) and np.all(mo>=0)
qualified=context["qualified"][co].astype(bool)
reference=context["target"][co]
assert int(qualified.sum())==506
query=context["outer_query"][:,co].astype(bool)
assert query.shape==(29,777)
treated=actual["metadata"].condition.to_numpy()!="control"
heldout_treated=np.any(query[context["outer_protocol"]=="leave_treatment"],axis=0)&treated&qualified
assert int(heldout_treated.sum())==235
cache_path=Path(os.path.abspath(root/args.feature_cache)) if args.feature_cache is not None else None
if reference_mixture:
    assert cache_path is None,"Validate each declared calibration from the actual No input."
    calibration_bundles=list(map(lambda r:reference_source_observations(native,period,options[r],
        fundamental_support=bool(fundamental_support[r]),device=args.device),np.arange(len(options))))
    bundle=calibration_bundles[0]
elif cache_path is None:
    bundle=reference_source_observations(native,period,options,fundamental_support=fundamental_support,device=args.device)
else:
    bundle=dict(np.load(cache_path,allow_pickle=False))
    assert json.loads(str(bundle["representation_options_json"]))==options
    assert bool(bundle["fundamental_support_enabled"])==fundamental_support
np.testing.assert_array_equal(bundle["trace_id"],names)
np.testing.assert_array_equal(bundle["period"],period)
np.testing.assert_allclose(bundle["period_scale"],scale,rtol=0.,atol=1e-12)
np.testing.assert_array_equal(bundle["time"],actual["time"])
np.testing.assert_array_equal(bundle["mask"],actual["mask"])
np.testing.assert_array_equal(bundle["count"],actual["count"])
np.testing.assert_array_equal(bundle["run_index"],actual["run_index"])
np.testing.assert_array_equal(bundle["run_names"],actual["runs"])
model_runs=model.get("run_names",bundle["run_names"])
ro=pd.Index(model_runs).get_indexer(bundle["run_names"])
assert np.all(ro>=0)
training=model["training_run"][rows][:,ro].astype(bool)
expected_training=np.concatenate([np.ones((1,len(bundle["run_names"])),dtype=bool),
    ~((query.astype(float)@bundle["membership"].T)>0.)],axis=0)
np.testing.assert_array_equal(training,expected_training[case_ids[rows]])
train=training[:,bundle["run_index"]]&qualified[None]
assert not np.any(train[case_ids[rows]>0]&query[case_ids[rows][case_ids[rows]>0]-1])
np.testing.assert_array_equal(model["qualified"][mo],qualified)
np.testing.assert_allclose(model["reference"][mo][qualified],reference[qualified],rtol=0.,atol=1e-10)
source_id=model.get("reference_source_id",model["trace_id"])[mo]
source_code,source_names=pd.factorize(source_id,sort=True)
source_member=(np.arange(len(source_names))[:,None]==source_code[None]).astype(float)
n_reference=((train.astype(float)@source_member.T)>0.).sum(1)
np.testing.assert_array_equal(model["reference_source_count"][rows],n_reference)
nq=bundle["membership"]@qualified.astype(float)
groups=(training&(nq>0.)[None]).sum(1)
np.testing.assert_array_equal(model["effective_training_units"][rows],groups)
assert str(model["foreground_prior_kind"])=="uniform"
uniform_log_probability=np.full((len(rows),len(period)),-np.log(len(period)))
if "foreground_log_prior" in model:
    configured=model["foreground_log_prior"][rows]
    assert np.max(np.ptp(configured,axis=-1))<CONFIG["probability_tolerance"]
if reference_mixture:
    calibration_mask=model["reference_supervision_mask"][rows][:,:,mo].astype(bool)
    assert not np.any(calibration_mask&~train[:,None])
    calibration_ids=model["calibration_reference_source_id"][rows][:,:,mo]
    unit_ids=model["source_refs_independent_unit"][mo]
    expert_n=np.asarray(list(map(lambda f:list(map(lambda r:len(np.unique(calibration_ids[f,r,calibration_mask[f,r]])),
        np.arange(calibration_mask.shape[1]))),np.arange(len(rows)))))
    expert_g=np.asarray(list(map(lambda f:list(map(lambda r:len(np.unique(unit_ids[calibration_mask[f,r]])),
        np.arange(calibration_mask.shape[1]))),np.arange(len(rows)))))
    np.testing.assert_array_equal(expert_n,model["calibration_reference_source_count"][rows])
    np.testing.assert_array_equal(expert_g,model["calibration_reference_unit_count"][rows])
    fresh=predict_reference_calibration_mixture(native,model,cases=rows,device=args.device,
        include_diagnostics=True,source_bundles=calibration_bundles)
else:
    fresh=predict_reference_source_w2(bundle,model,cases=rows,device=args.device)
period_keys=["estimate","source_period","deviation"]
replay={key:float(np.max(abs(fresh[key]-model[key][rows][:,mo]))) for key in period_keys}
replay["group_mean"]=float(np.max(abs(fresh["group_mean"]-model["group_mean"][rows][:,ro])))
mass_error=float(np.max(abs(fresh["posterior"].sum(-1)-1.)))
expectation_error=float(np.max(abs(fresh["posterior"]@period-fresh["estimate"])))
source_point_error=float(np.max(abs(fresh["source_period"]-fresh["estimate"])))
mean_error=float(np.max(abs(fresh["estimate"]@bundle["average"].T-fresh["group_mean"])))
deviation_error=float(np.max(abs(fresh["deviation"]-(fresh["estimate"]-fresh["group_mean"][:,bundle["run_index"]]))))
source_mean_error=float(np.max(abs(fresh["source_period"]@bundle["average"].T-fresh["group_mean"]))) if reference_mixture else float(np.max(abs(fresh["cohort_source_period"]@bundle["average"].T-fresh["cohort_source_mean"])))
mixture_checks={}
if reference_mixture:
    weights=fresh["reference_calibration_weight"];component=fresh["reference_calibration_probability"]
    assert np.all(weights>=0.) and np.all(component>=0.)
    mixture_checks={"reference_weight_mass":float(np.max(abs(weights.sum(1)-1.))),
        "reference_probability_mass":float(np.max(abs(component.sum(-1)-1.))),
        "reference_probability_mixture":float(np.max(abs(np.einsum("frn,frnp->fnp",weights,component,optimize=True)-fresh["posterior"])))}
    mass_error=max(mass_error,max(mixture_checks.values()))
target_error=float(np.max(abs(fresh["group_mean"]-fresh["calibrated_mean_target"])))
relations={"probability":mass_error,"expectation_h":expectation_error,"source_point_h":source_point_error,
    "group_mean_h":mean_error,"deviation_h":deviation_error,"source_mean_h":source_mean_error,"W2_target_h":target_error}
assert np.isfinite(fresh["estimate"]).all() and fresh["estimate"].shape==(len(rows),777)
assert np.all(fresh["posterior"]>=0.)
assert fresh["estimate"].min()>=period.min()-1e-10 and fresh["estimate"].max()<=period.max()+1e-10
if args.stage=="readback":
    ok=max(replay.values())<CONFIG["period_tolerance_h"] and mass_error<CONFIG["probability_tolerance"] and max(v for k,v in relations.items() if k!="probability")<CONFIG["period_tolerance_h"]
    print("REFERENCE_SOURCE_KERNEL_API_READBACK",json.dumps({"api_readback_passed":bool(ok),"full_30_validation":"not_run",
        "model":str(model_path),"case_index":case_ids[rows].tolist(),"native_records":777,"qualified":506,
        "unique_heldout_treated":235,"N_by_case":n_reference.tolist(),"G_by_case":groups.tolist(),
        "foreground_prior_kind":"uniform","replay_error_h":replay,"W2_relations":relations}),flush=True)
    raise SystemExit(0 if ok else 1)
factor=CONFIG["unit_factor"]
if reference_mixture:
    second_options=list(map(lambda opt:dict(opt,**({"common_baseline_period":float(opt["common_baseline_period"])*factor}
        if "common_baseline_period" in opt else {})),options))
    seconds_model=dict(model,period=period*factor,period_scale=np.asarray(scale*factor),time_unit=np.asarray("s"),
        calibration_representation_options_json=np.asarray(list(map(json.dumps,second_options))))
    second_frame=native.assign(x=native.x*factor)
    second_bundles=list(map(lambda r:reference_source_observations(second_frame,period*factor,second_options[r],
        fundamental_support=bool(fundamental_support[r]),device=args.device),np.arange(len(options))))
    seconds=second_bundles[0]
else:
    second_options=dict(options)
    if "common_baseline_period" in second_options:
        second_options["common_baseline_period"]=float(second_options["common_baseline_period"])*factor
    seconds_model=dict(model,period=period*factor,period_scale=np.asarray(scale*factor),time_unit=np.asarray("s"),
        representation_options_json=np.asarray(json.dumps(second_options)))
    seconds=reference_source_observations(native.assign(x=native.x*factor),period*factor,second_options,
        fundamental_support=fundamental_support,device=args.device)
np.testing.assert_array_equal(seconds["mask"],bundle["mask"])
statistics=["identity_base_features","raw_features","core_features","base_features","rhs","joint_gram",
    "coherent_power","source_unit","physical_peer","fundamental_support"]
if reference_mixture:
    statistic_error={str(r)+":"+key:float(np.max(abs(second_bundles[r][key]-calibration_bundles[r][key])))
        for r in range(len(options)) for key in statistics}
    converted=predict_reference_calibration_mixture(second_frame,seconds_model,cases=rows,device=args.device,
        include_diagnostics=True,source_bundles=second_bundles)
else:
    statistic_error={key:float(np.max(abs(seconds[key]-bundle[key]))) for key in statistics}
    if "noise_tau" in bundle:
        statistic_error["noise_tau_h"]=float(np.max(abs(seconds["noise_tau"]/factor-bundle["noise_tau"])))
    converted=predict_reference_source_w2(seconds,seconds_model,cases=rows,device=args.device)
unit_keys=period_keys+["group_mean","calibrated_mean_target"]+( [] if reference_mixture else ["cohort_source_period","cohort_source_mean"])
unit_error={key:float(np.max(abs(converted[key]/factor-fresh[key]))) for key in unit_keys}
unit_mass_error=float(np.max(abs(converted["posterior"]-fresh["posterior"])))
def complete_cohort(run_name):
    frame=native.loc[native.run.eq(run_name)]
    if reference_mixture:
        parts=list(map(lambda r:reference_source_observations(frame,period,options[r],
            fundamental_support=bool(fundamental_support[r]),device=args.device),np.arange(len(options))))
        part=parts[0]
    else:
        part=reference_source_observations(frame,period,options,fundamental_support=fundamental_support,device=args.device)
    order=pd.Index(names).get_indexer(part["trace_id"])
    assert len(order)==np.sum(bundle["run"]==run_name) and np.all(order>=0)
    value=predict_reference_calibration_mixture(frame,model,cases=rows,device=args.device,source_bundles=parts) if reference_mixture else predict_reference_source_w2(part,model,cases=rows,device=args.device)
    return {"run":str(run_name),"records":len(order),
        "period_error_h":float(np.max(abs(value["estimate"]-fresh["estimate"][:,order]))),
        "probability_error":float(np.max(abs(value["posterior"]-fresh["posterior"][:,order])))}
batches=list(map(complete_cohort,bundle["run_names"]))
batch_error=max(v["period_error_h"] for v in batches)
batch_mass_error=max(v["probability_error"] for v in batches)
assert sum(v["records"] for v in batches)==777
controlled=observations(pd.read_parquet(root/CONFIG["paths"]["controlled"]))
assert len(controlled["metadata"])==2304
controlled_count,controlled_number=np.unique(controlled["count"],return_counts=True)
ecg=dict(np.load(root/CONFIG["paths"]["ecg_context"],allow_pickle=False))
ecg_reference=pd.read_csv(root/CONFIG["paths"]["ecg_references"],encoding="utf-8-sig")
ecg_sources=ecg_reference.loc[ecg_reference.qualified,["run","reference_period_s"]].drop_duplicates()
assert len(ecg["trace_id"])==600 and int(ecg["qualified"].sum())==592
assert len(ecg_sources)==148 and ecg_sources.run.is_unique and ecg_reference.subject.nunique()==5
output.parent.mkdir(parents=True,exist_ok=True)
cli_output=output.parent/(model_path.stem.removesuffix("_model")+"_validation_cli_predictions.csv")
command=[sys.executable,"-X","utf8",str(root/"software/predict_source_assignment.py"),"--model",str(model_path),
    "--input",str(root/CONFIG["paths"]["native"]),"--output",str(cli_output),"--case","0","--time-unit","h"]
subprocess.run(command,cwd=root,check=True)
cli=pd.read_csv(cli_output,encoding="utf-8").set_index("trace_id").loc[names]
cli_mass=dict(np.load(cli_output.with_name(cli_output.stem+"_source.npz"),allow_pickle=False))
cli_order=pd.Index(cli_mass["trace_id"]).get_indexer(names)
assert len(cli)==777 and cli.index.is_unique and np.all(cli_order>=0)
cli_error=float(np.max(abs(cli.estimate_period.to_numpy()-fresh["estimate"][0])))
cli_mass_error=float(np.max(abs(cli_mass["source_probability"][cli_order]-fresh["posterior"][0])))
numeric_checks={"replay":max(replay.values())<CONFIG["period_tolerance_h"],
    "probability":mass_error<CONFIG["probability_tolerance"],
    "W2_relationships":max(v for k,v in relations.items() if k!="probability")<CONFIG["period_tolerance_h"],
    "hours_seconds_statistics":max(statistic_error.values())<CONFIG["statistic_tolerance"],
    "hours_seconds_period":max(unit_error.values())<CONFIG["period_tolerance_h"],
    "hours_seconds_probability":unit_mass_error<CONFIG["probability_tolerance"],
    "cohort_period":batch_error<CONFIG["period_tolerance_h"],"cohort_probability":batch_mass_error<CONFIG["probability_tolerance"],
    "source_cli_period":cli_error<CONFIG["period_tolerance_h"],"source_cli_probability":cli_mass_error<CONFIG["probability_tolerance"]}
numeric_checks.update({key:bool(value["passed"]) for key,value in boundary_checks.items()})
native_count,native_number=np.unique(bundle["count"],return_counts=True)
checks={"passed":bool(all(numeric_checks.values())),"status":"completed","numerical_checks":{k:bool(v) for k,v in numeric_checks.items()},
    "thresholds":{"period_h":CONFIG["period_tolerance_h"],"probability":CONFIG["probability_tolerance"],"statistics":CONFIG["statistic_tolerance"]},
    "started_utc":started,"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model":str(model_path),"schema":str(model["schema"]),"readout_kind":str(model["readout_kind"]),
    "source_module":str(Path(source.__file__).resolve()),"case_index":case_ids[rows].tolist(),
    "native_records":777,"native_observations":len(native),"native_reference_sources":506,"unique_heldout_treated":235,"cases":30,"natural_runs":17,
    "boundary_validation":boundary_checks,
    "replay_error_h":replay,"W2_relations":relations,
    "p0":{"kind":"uniform","log_probability":float(uniform_log_probability[0,0]),"N_by_case":n_reference.tolist(),
        "G_by_case":groups.tolist(),"artificial_views_do_not_increase_N":True,"trained_query_overlap":0},
    "feature_cache":str(cache_path) if cache_path is not None else "fresh_reference_source_observations",
    "representation_options":options,"fundamental_support":fundamental_support.tolist() if reference_mixture else fundamental_support,
    "reference_calibration_probability_checks":mixture_checks,
    "calibration_N_by_case":expert_n.tolist() if reference_mixture else None,
    "calibration_G_by_case":expert_g.tolist() if reference_mixture else None,
    "hours_seconds":{"statistics_error":statistic_error,"period_error_h":unit_error,"probability_error":unit_mass_error},
    "cohort_batches":batches,"cohort_max_error_h":batch_error,"cohort_probability_error":batch_mass_error,
    "actual_layout_coverage":{"native_counts":dict(zip(native_count.tolist(),native_number.tolist())),
        "controlled_records":2304,"controlled_counts":dict(zip(controlled_count.tolist(),controlled_number.tolist()))},
    "ecg_reference_sources":{"inputs":600,"qualified_channels":592,"distinct_Direct_1_sources":148,"subjects":5,"model_inference":"not_run"},
    "source_cli":{"command":command,"records":777,"estimate_error_h":cli_error,"probability_error":cli_mass_error,
        "output":str(cli_output),"installation_boundary":"source CLI; installed wheel not tested"},
    "scope":"Actual-record implementation, units, complete cohorts, uniform prior and bounded-W2 probability/readout relationships; no accuracy assessment"}
output.write_text(json.dumps(checks,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print("UNIFIED_KERNEL_VALIDATED",json.dumps(checks,ensure_ascii=False),flush=True)
if not checks["passed"]:
    raise SystemExit(1)
