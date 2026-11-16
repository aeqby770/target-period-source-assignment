"""Evaluate matched posterior-clock ablations through the same formal source."""
from pathlib import Path
import os
CONFIG={"paths":{"root":Path(__file__).resolve().parents[1]},"threads":os.cpu_count(),"plot":{},
    "variants":["unit_bic","no_shared_order","no_posterior_energy","no_balance","balanced_anchor"]}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
import sys,subprocess,argparse,json,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"software"))
from source_assignment_distribution import SELECTED_MODEL_STEM,SELECTED_ABLATIONS
root=CONFIG["paths"]["root"]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family",choices=["posterior","frequency","unified"],default="frequency")
parser.add_argument("--model",type=Path)
parser.add_argument("--stage",choices=["readout","models","evaluate","external","all-evaluate"],default="models")
parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
args=parser.parse_args()
if args.family=="unified":
    from scipy.special import softmax
    from source_assignment_unified import observations,raw_no_anchor_density,calibrate_source_density
    folder=root/"results/0.3.1";main_path=args.model or folder/"unified_model.npz"
    main=dict(np.load(main_path,allow_pickle=False));assert str(main["readout_kind"])=="information_conditioned_reference_W2"
    np.testing.assert_array_equal(main["case_index"],np.arange(30))
    if args.stage in {"evaluate","external","all-evaluate"}:
        import time as clock
        catalogue=pd.read_csv(folder/"native_unified_ablation_catalogue.csv",encoding="utf-8")
        comparison=[]
        for path in catalogue.comparison_model_path:
            comparison.extend(["--comparison-model",str(root/path)])
        stages=[]
        if args.stage in {"evaluate","all-evaluate"}:
            stages.append(("native",[sys.executable,"-X","utf8",str(root/"experiments/evaluate_unified_native.py"),"--model",str(main_path),"--device",args.device,"--refresh-model-cache"]+comparison))
        if args.stage in {"external","all-evaluate"}:
            panels=[("controlled",[]),("controlled__shorter",["--acquisition","shorter"]),("controlled__missing",["--acquisition","missing"]),
                ("single",[]),("dual",[]),("coupled",[])]+[("scn_"+phase+"_l"+str(length),["--phase",phase,"--length",str(length)]) for phase in ["Pre","TTX","Wash"] for length in [104,145]]
            for tag,extra in panels:
                dataset="scn" if tag.startswith("scn_") else "controlled" if tag.startswith("controlled") else tag
                command=[sys.executable,"-X","utf8",str(root/"experiments/evaluate_unified_external.py"),"--dataset",dataset,"--stage","all",                    "--source-role-model",str(main_path),"--device",args.device]+extra+comparison
                stages.append((tag,command))
        if args.stage=="all-evaluate":
            stages.extend([("adfecg",[sys.executable,"-X","utf8",str(root/"experiments/evaluate_unified_adfecg.py"),"--model",str(folder/"adfecg_model.npz"),"--output-dir",str(folder)]),
                ("kernel",[sys.executable,"-X","utf8",str(root/"development/validate_unified_kernel.py"),"--model",str(main_path),"--output",str(folder/"kernel_validation.json"),"--device",args.device])])
        records=[]
        for tag,command in stages:
            log=folder/("unified_ablation_"+tag+"_evaluation.log")
            started=clock.perf_counter();print("UNIFIED_FINAL_STAGE_START",tag,"log",log,flush=True)
            with log.open("w",encoding="utf-8") as stream:
                result=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=False)
            seconds=clock.perf_counter()-started
            records.append({"panel":tag,"status":"complete" if result.returncode==0 else "failed","returncode":result.returncode,"elapsed_seconds":seconds,"comparison_count":len(catalogue),"log_path":str(log)})
            pd.DataFrame(records).to_csv(folder/"native_unified_ablation_evaluation_stages.csv",index=False,encoding="utf-8")
            print("UNIFIED_FINAL_STAGE_END",tag,"returncode",result.returncode,"elapsed_seconds",seconds,flush=True)
        raise SystemExit(0 if all(row["returncode"]==0 for row in records) else 1)
    frame=pd.read_parquet(root/"data/sota_dev/recomputed_clean/lemna_task_period_acf_input.parquet")
    data=observations(frame);assert np.array_equal(data["metadata"].index.to_numpy(dtype=str),main["trace_id"])
    period=main["period"];nf,nr=main["calibration_within_scale"].shape
    anchor=raw_no_anchor_density(frame,period,main["calibration_anchor_coefficient"].reshape(nf*nr,-1),period_scale=main["period_scale"])
    raw=softmax(anchor["log_probability"],axis=-1).reshape(nf,nr,len(main["trace_id"]),len(period))
    original_weight=main["reference_calibration_weight"]
    raw_mixture=np.einsum("frn,frnp->fnp",original_weight,raw,optimize=True)
    np.testing.assert_allclose(raw_mixture,main["raw_individual_probability"],rtol=0.,atol=1e-7)
    catalogue=[];models={}
    def save_variant(name,model,probability,component,weight,parameter_policy):
        point=probability@period;mean=point@data["average"].T
        model.update(posterior=probability,source_period=point,estimate=point,group_mean=mean,
            deviation=point-mean[:,data["run_index"]],cohort_probability=np.einsum("gn,fnp->fgp",data["average"],probability,optimize=True),
            reference_calibration_probability=component,reference_calibration_weight=weight,
            reference_calibration_mean=(component@period)@data["average"].T,
            raw_individual_probability=np.einsum("frn,frnp->fnp",weight,raw,optimize=True),calibrated_mean_target=mean,
            control_name=np.asarray("ablation_"+name),ablation_policy=np.asarray(name),
            ablation_field_scope=np.asarray("reference-calibration-mixture"),ablation_parameter_policy=np.asarray(parameter_policy),parameter_policy=np.asarray(parameter_policy))
        active=np.asarray(model["reference_supervision_mask"],dtype=bool).copy()
        if name=="base_only":
            active[:,1:]=False
        if name=="refined_only":
            active[:,:-1]=False
        active=active.any(1)
        units=np.unique(model["source_refs_independent_unit"])
        unit_member=(units[:,None]==model["source_refs_independent_unit"][None]).astype(float)
        model["ablation_active_reference_source_count"]=active.sum(1)
        model["ablation_active_reference_unit_count"]=(active.astype(float)@unit_member.T>0.).sum(1)
        model["ablation_count_semantics"]=np.asarray("evaluation_506_sources_235_LTO_unchanged; active_reference_supervision_recorded_separately")
        path=folder/("native_ablation_"+name+"_model.npz");np.savez_compressed(path,**model);models[name]=model
        catalogue.append({"control_name":"ablation_"+name,"ablation_policy":name,"model_path":str(path),
            "comparison_model_path":Path(path.relative_to(root)).as_posix(),"case_count":nf,"case_index":"0..29",
            "input_records":len(main["trace_id"]),"qualified_reference_sources":int(main["source_refs_qualified"].sum()),
            "evaluation_source_denominator":506,"evaluation_LTO_denominator":235,"full_input_runs":len(main["run_names"]),
            "parameter_policy":parameter_policy,"scope":"reference-calibration-mixture","status":"complete",
            "fit_success_all":bool(np.asarray(model["success"]).all()),
            "active_reference_source_count":json.dumps(model["ablation_active_reference_source_count"].tolist()),
            "active_reference_unit_count":json.dumps(model["ablation_active_reference_unit_count"].tolist()),
            "calibration_reference_source_count":json.dumps(model["calibration_reference_source_count"].tolist()),
            "calibration_reference_unit_count":json.dumps(model["calibration_reference_unit_count"].tolist())})
        pd.DataFrame(catalogue).to_csv(folder/"native_unified_ablation_catalogue.csv",index=False,encoding="utf-8")
        print("UNIFIED_ABLATION_MODEL",name,path,nf,"cases",flush=True)
        return path
    components=main["reference_calibration_probability"]
    for name,index in [("base_only",0),("refined_only",nr-1)]:
        weight=np.zeros_like(original_weight);weight[:,index]=1.
        save_variant(name,dict(main),components[:,index],components,weight,"readout_only_frozen_reference_parameters")
    weight=np.full_like(original_weight,1./nr)
    save_variant("equal_reference_weights",dict(main),np.einsum("frn,frnp->fnp",weight,components,optimize=True),components,weight,"readout_only_frozen_reference_parameters")
    save_variant("no_W2",dict(main),raw_mixture,raw,original_weight.copy(),"raw_individual_mass_without_W2; no_point_anchor_composition")
    transported=[]
    for expert in range(nr):
        value=calibrate_source_density(anchor["log_probability"].reshape(nf,nr,len(main["trace_id"]),len(period))[:,expert],period,data["run_index"],
            main["reference_calibration_mean"][:,expert],calibration_kind="wasserstein_affine",within_scale=np.ones((nf,1)))
        transported.append(value["posterior"])
    component=np.stack(transported,axis=1);variant=dict(main);variant["calibration_within_scale"]=np.ones_like(main["calibration_within_scale"])
    save_variant("within_scale_one",variant,np.einsum("frn,frnp->fnp",original_weight,component,optimize=True),component,original_weight.copy(),"within_scale_fixed_one; frozen_reference_PAV_and_measurements")
    if args.stage=="readout":
        raise SystemExit(0)
    from source_assignment_unified import reference_source_observations,predict_reference_source_w2,predict_reference_calibration_mixture,_fit_period_role
    from source_assignment_core import fit_monotone_mean
    from scipy.special import logsumexp
    import time as clock
    began=clock.perf_counter()
    parameters=["core_coefficient","anchor_coefficient","background_coefficient","role_covariance","role_coefficient",        "calibration_prediction","calibration_mean","calibration_experiments","within_scale","core_coherent_peer","core_partial_peer","mean_identity_tails"]
    assert not np.any(main["reference_supervision_mask"]&main["source_refs_query_mask"][:,None])
    original_options=[json.loads(str(value)) for value in main["calibration_representation_options_json"]]
    def build_reference(item):
        index,options=item
        print("UNIFIED_ABLATION_MEASUREMENT",index,options,flush=True)
        return reference_source_observations(frame,period,options,fundamental_support=bool(main["calibration_fundamental_support"][index]),device=args.device)
    bundles=list(map(build_reference,enumerate(original_options)))
    baseline=predict_reference_calibration_mixture(frame,main,cases=np.arange(nf),device=args.device,include_diagnostics=True,source_bundles=bundles)
    baseline_error=float(np.max(abs(baseline["estimate"]-main["estimate"])))
    print("UNIFIED_ABLATION_DEFAULT_READBACK",baseline_error,flush=True);assert baseline_error<1e-6
    def fresh_readback(name,model,measured,policy):
        value=predict_reference_calibration_mixture(frame,model,cases=np.arange(nf),device=args.device,include_diagnostics=True,source_bundles=measured)
        model.update({key:val for key,val in value.items() if isinstance(val,np.ndarray)})
        return save_variant(name,model,value["posterior"],value["reference_calibration_probability"],value["reference_calibration_weight"],policy)
    # Re-read all five through the public source API; no cached legacy predictions.
    for name in ["base_only","refined_only","equal_reference_weights","no_W2","within_scale_one"]:
        model=models[name]
        for key in ["training_run","reference_supervision_mask","calibration_reference_source_count","calibration_reference_unit_count","reference_source_count","effective_training_units"]:
            model[key]=main[key].copy()
        fresh_readback(name,model,bundles,str(model["ablation_parameter_policy"]))
    # Catalogue rows are unique even after the public readback refresh above.
    catalogue=list({row["control_name"]:row for row in catalogue}.values())
    def recalibrate_training_pav(model,measured,name):
        for key in ["calibration_calibration_prediction","calibration_calibration_mean","calibration_calibration_experiments"]:
            model[key]=model[key].copy()
        pav_allowed=[]
        for expert in range(nr):
            parent={key:model["calibration_"+key][:,expert] for key in parameters}
            if "calibration_source_features" in model:
                parent["source_features"]=np.asarray(model["calibration_source_features"][expert])
            value=predict_reference_source_w2(measured[expert],parent,cases=np.arange(nf),device=args.device)
            mask=np.asarray(main["reference_supervision_mask"][:,expert],dtype=bool)
            assert not np.any(mask&main["source_refs_query_mask"])
            count=mask.astype(float)@data["membership"].T
            target=np.divide(np.where(mask,main["source_refs_reference"][None],0.)@data["membership"].T,count,out=np.zeros_like(count),where=count>0.)
            allowed=count>0.;pav_allowed.append(allowed)
            _,x,y,number=fit_monotone_mean(value["cohort_source_mean"]/float(main["period_scale"]),target/float(main["period_scale"]),allowed)
            model["calibration_calibration_prediction"][:,expert]=x;model["calibration_calibration_mean"][:,expert]=y;model["calibration_calibration_experiments"][:,expert]=number
        model["ablation_PAV_training_run"]=np.stack(pav_allowed,axis=1)
        model["ablation_PAV_policy"]=np.asarray("expert_own_original_qualified_reference_supervision_only_no_query")
        model["calibration_parameter_provenance"]=np.asarray([[str(main["calibration_parameter_provenance"][case,expert])+"; frozen_measurement_PAV_recalibrated_"+name for expert in range(nr)] for case in range(nf)])
        return model
    model=dict(main);model["calibration_role_coefficient"]=np.zeros_like(main["calibration_role_coefficient"])
    model=recalibrate_training_pav(model,bundles,"no_source_role31")
    fresh_readback("no_source_role31",model,bundles,"frozen_core_BG_covariance_anchor_within; role31_zero; own_training_PAV_recalibration")
    representation_cases=[("no_common_baseline",{"common_baseline":False}),
        ("white_residual",{"residual_noise_model":"white"}),
        ("map_two_clock",{"frequency_reduction":"map"})]
    for name,change in representation_cases:
        model=dict(main);options=[];measured=[]
        for expert in range(nr):
            current=dict(original_options[expert])
            applicable= name=="no_common_baseline" and bool(current.get("common_baseline",True)) or name=="white_residual" and (bool(current.get("colored_order",True)) or bool(current.get("frequency_correlated",True)) or bool(main["calibration_fundamental_support"][expert])) or name=="map_two_clock" and bool(current.get("frequency_marginal",True))
            if applicable:
                current.update(change);measured.append(build_reference((expert,current)))
            else:
                measured.append(bundles[expert])
            options.append(json.dumps(current) if applicable else str(main["calibration_representation_options_json"][expert]))
        model["calibration_representation_options_json"]=np.asarray(options)
        model=recalibrate_training_pav(model,measured,name)
        fresh_readback(name,model,measured,"frozen_reference_measurement_parameters; changed_No_representation; expert_training_only_PAV_recalibration; not_matched_refit")
    # Strong same-information comparator: train only the 31 density terms on the
    # current frozen per-reference core/BG fields, never reuse mismatched parents.
    density=dict(main);density["calibration_role_coefficient"]=np.empty_like(main["calibration_role_coefficient"])
    density["calibration_source_features"]=np.full(nr,"density")
    density["calibration_measurement_kind"]=np.full(nr,"density31")
    density_status=[]
    for expert in range(nr):
        bundle=bundles[expert];core=main["calibration_core_coefficient"][:,expert]
        fixed_raw=np.einsum("npd,fd->fnp",bundle["raw_features"],core,optimize=True)
        fixed_coherent=np.einsum("npd,fd->fnp",bundle["core_features"],core,optimize=True)
        fixed=np.where(main["calibration_core_coherent_peer"][:,expert,None,None],fixed_coherent,fixed_raw)
        background=np.einsum("npd,fd->fnp",bundle["raw_features"],main["calibration_background_coefficient"][:,expert],optimize=True)
        fields=np.stack([fixed-logsumexp(fixed,axis=-1,keepdims=True),background-logsumexp(background,axis=-1,keepdims=True)],axis=-1)
        mask=main["reference_supervision_mask"][:,expert];qualified=main["reference_supervision_mask"][0,expert]
        target=np.where(qualified,main["source_refs_reference"]/float(main["period_scale"]),0.)
        count=data["membership"]@qualified.astype(float)
        group=np.divide(data["membership"]@target,count,out=np.zeros_like(count),where=count>0.)
        allowed=(mask.astype(float)@data["membership"].T)>0.
        print("MATCHED_DENSITY31_REFERENCE_START",expert,"cases",nf,"N",main["calibration_reference_source_count"][:,expert].tolist(),flush=True)
        coefficient,status=_fit_period_role(bundle["base_features"],fields,fixed,target,qualified,mask,period/float(main["period_scale"]),900,args.device,
            cohort=(data["membership"],data["average"],group,allowed),reference_count=main["calibration_reference_source_count"][:,expert])
        density["calibration_role_coefficient"][:,expert]=coefficient;density_status.append(status)
    density["density31_optimizer_success"]=np.asarray([status["success"] for status in density_status])
    density["density31_optimizer_iterations"]=np.asarray([status["iterations"] for status in density_status])
    density["density31_optimizer_gradient"]=np.asarray([status["gradient"] for status in density_status])
    density["success"]=main["success"]&all(status["success"] for status in density_status)
    density=recalibrate_training_pav(density,bundles,"matched_density31")
    fresh_readback("matched_density31",density,bundles,"matched_density31_refit_only; identical_frozen_reference_core_BG_No_fields_and_proper_plus_mean_role_objective; own_training_PAV; unchanged_anchor_within_information_W2")
    catalogue=list({row["control_name"]:row for row in catalogue}.values())
    pd.DataFrame(catalogue).to_csv(folder/"native_unified_ablation_catalogue.csv",index=False,encoding="utf-8")
    print("UNIFIED_ABLATION_MODELS_COMPLETE",len(catalogue),"models","baseline_readback",baseline_error,"seconds",clock.perf_counter()-began,flush=True)
    if args.stage=="evaluate":
        command=[sys.executable,"-X","utf8",str(root/"experiments/evaluate_unified_native.py"),"--model",str(main_path),"--device",args.device]
        for row in catalogue:
            command.extend(["--comparison-model",row["model_path"]])
        subprocess.run(command,cwd=root,check=True)

    raise SystemExit(0)

CONFIG["variants"]={"posterior":CONFIG["variants"],"frequency":SELECTED_ABLATIONS}[args.family]
model_stem={"posterior":"lemna_task_period_source_assignment_posterior_clock","frequency":SELECTED_MODEL_STEM}[args.family]

common=model_stem=="lemna_task_period_source_assignment_common_baseline"
list(map(lambda variant:subprocess.run([sys.executable,"-X","utf8","-u",
    str(root/"experiments/evaluate_posterior_clock.py"),"--model-stem",model_stem+"__"+variant]+(
    ["--reference-fit",str(root/"data/sota_dev/recomputed_clean/lemna_task_period_actual_source_assignment_common_baseline_fit.npz"),
     "--balance-cache",str(root/("data/sota_dev/recomputed_clean/amplitude_0_2_4/native_no_common_baseline.npz" if variant=="no_common_baseline" else "data/sota_dev/recomputed_clean/amplitude_0_2_4/native_common.npz")),
     "--controlled-cache",str(root/("data/sota_dev/recomputed_clean/amplitude_0_2_4/controlled_no_common_baseline.npz" if variant=="no_common_baseline" else "data/sota_dev/recomputed_clean/amplitude_0_2_4/controlled_common.npz"))]+(["--refresh-cache"] if variant=="no_common_baseline" else []) if common else []),
    cwd=root,env=os.environ|{"PYTHONPATH":str(root/"software")},check=True),CONFIG["variants"]))
print("POSTERIOR_MATCHED_ABLATIONS_COMPLETE",len(CONFIG["variants"]),flush=True)
