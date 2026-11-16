"""Shared source-role evidence, score features and experiment-mean calibration."""
import numpy as np
import torch


def legacy_source_role_observations(time,mask,source_signal,reference_signal=None,run_index=None,period=None,noise_tau=None,*,prepare_source=False,representation_options=None,fundamental_support=True,reference_peer_original=True):
    """Original source-score observations on actual timestamps and masks.

    prepare_source=True consumes raw No and estimates its representation using
    source_amplitude_representation. Otherwise source_signal/reference_signal
    are the caller's prepared source and original-No peer inputs. noise_tau has
    the same time unit as time and period. Positive regular-grid AR1 support
    is recovered at tau=-dt/log(rho); the common whitening factor cancels.
    Frozen scoring weights and the reference-trained PAV remain unchanged.
    """
    from source_assignment_unified import profile_features,_role_geometry,observed_fundamental_support
    prepared=None
    if prepare_source:
        from source_amplitude_normalization import source_amplitude_representation
        prepared=source_amplitude_representation(time,mask,source_signal,run_index,period,**(representation_options or {}))
        source_signal=prepared["signal"];reference_signal=prepared["reference_signal"] if reference_peer_original else None;noise_tau=prepared["noise_tau"]

    time=np.asarray(time,dtype=float);mask=np.asarray(mask,dtype=bool)
    period=np.asarray(period,dtype=float);run_index=np.asarray(run_index,dtype=int)
    member=(np.arange(int(run_index.max())+1)[:,None]==run_index[None]).astype(float)
    data={"time":time,"mask":mask,"count":mask.sum(1),"run_index":run_index,
        "membership":member,"average":member/member.sum(1)[:,None]}
    scale=float(np.sqrt(period.min()*period.max()))
    base,_=profile_features(source_signal,data,period,scale,peer_reference=reference_signal,normalize_reference_peer=True)
    geometry=_role_geometry(source_signal,data,period,scale,peer_reference=reference_signal,normalize_reference_peer=True)
    support=observed_fundamental_support(geometry["unit"],geometry["peer"],data,period,scale,np.asarray(noise_tau,dtype=float))
    support=np.where(np.broadcast_to(np.asarray(fundamental_support,dtype=bool),(len(mask),))[:,None,None],support,1.)
    base[:,:,1]*=support[:,:,0];base[:,:,5]*=support[:,:,1]
    coherent=generic_coherent_fwl_power(time,mask,geometry["unit"],geometry["peer"],period,half_gram=True)
    core_base=base.copy();core_base[:,:,11]=coherent
    left,right=np.triu_indices(base.shape[-1])
    return {"period":period,"period_scale":np.asarray(scale),"time":time,"mask":mask,"count":data["count"],
        "run_index":run_index,"membership":member,"average":data["average"],
        "base_features":base,"raw_features":np.concatenate([base,base[:,:,left]*base[:,:,right]],axis=-1),
        "core_features":np.concatenate([core_base,core_base[:,:,left]*core_base[:,:,right]],axis=-1),
        "coherent_power":coherent,"fundamental_support":support,"source_unit":geometry["unit"],
        "physical_peer":geometry["peer"],"joint_gram":geometry["joint"],"layout":geometry["layout"],
        "rhs":geometry["rhs"],"noise_tau":np.asarray(noise_tau,dtype=float),"source_representation":prepared}


def legacy_source_role_readout(bundle,model,device="cpu"):
    """Original 104-term/31-role score and PAV, without a new FG prior.

    Model rows are explicit caller-selected frozen fits, independent of
    record length. The returned PAV target mean can calibrate the complete
    source distribution through calibrate_source_density, without an anchor.
    """
    from scipy.special import softmax
    from source_correlated_geometry import correlated_role_assignment_evidence
    raw=bundle["raw_features"];period=bundle["period"]
    background=softmax(np.einsum("npd,fd->fnp",raw,model["background_coefficient"],optimize=True),axis=-1)
    role,_=correlated_role_assignment_evidence(bundle["joint_gram"],bundle["rhs"],bundle["layout"],
        model["role_covariance"],background,len(period)//2,len(period)//2,device=device)
    role_terms=role_score_features(role,bundle["base_features"])
    role_score=np.einsum("fnpd,fd->fnp",role_terms,model["role_coefficient"],optimize=True)
    assert not np.any(model.get("core_contrast_coefficient",0.))
    coherent=np.asarray(model.get("core_coherent_peer",np.ones(len(model["coefficient"]),dtype=bool)))
    partial=np.asarray(model.get("core_partial_peer",np.zeros(len(model["coefficient"]),dtype=bool)))
    assert not np.any(partial&~coherent)
    score_raw=np.einsum("npd,fd->fnp",raw,model["coefficient"],optimize=True)
    score_coherent=np.einsum("npd,fd->fnp",bundle["core_features"],model["coefficient"],optimize=True)
    core_score=np.where(coherent[:,None,None],score_coherent,score_raw)
    posterior=softmax(core_score+role_score,axis=-1)
    source_period=posterior@period;source_mean=source_period@bundle["average"].T
    group_mean=interpolate_mean(source_mean,model["calibration_prediction"],model["calibration_mean"],
        model["calibration_experiments"],model.get("mean_identity_tails",False))
    return {"posterior":posterior,"core_score":core_score,"role_score":role_score,
        "source_period":source_period,"source_mean":source_mean,"group_mean":group_mean}


def measured_peer_partial_power(time, count, mixed, peer, design, layout, coherent=False, half_gram=False):
    """H1 gain after jointly removing the measured peer and its quadrature.

    The harmonic design is residualized against the same nuisance columns as
    the observed No signal, as required by the Frisch-Waugh-Lovell identity.
    Peer is the leave-one-out average of other cells in the current run.
    Coherent evidence uses the cross product of both temporal halves in the
    same physical harmonic coordinates and the same global FWL metric.
    half_gram estimates each half's coefficients in its actual design geometry.
    """
    from scipy.signal import hilbert
    mask = np.arange(len(time))[None] < count[:, None]
    centered = (time[None] - (mask*time[None]).sum(axis=-1)[:, None]/count[:, None])*mask
    trend = np.stack([mask/np.sqrt(count[:, None]),
        centered/np.sqrt((centered**2).sum(axis=-1))[:, None]], axis=-1)
    nuisance = np.stack([peer, hilbert(peer, axis=-1).imag*mask], axis=-1)
    nuisance -= np.einsum("nti,nij->ntj", trend,
        np.einsum("nti,ntj->nij", trend, nuisance, optimize=True), optimize=True)
    inverse = np.linalg.inv(np.einsum("nti,ntj->nij", nuisance, nuisance, optimize=True))
    harmonic = design[layout]
    cross = np.einsum("npti,ntj->npij", harmonic, nuisance, optimize=True)
    gram = np.einsum("npti,nptj->npij", harmonic, harmonic, optimize=True)
    gram -= cross@inverse[:, None]@cross.swapaxes(-1, -2)
    nuisance_rhs = np.einsum("nti,nt->ni", nuisance, mixed, optimize=True)
    rhs = np.einsum("npti,nt->npi", harmonic, mixed, optimize=True)
    rhs -= np.einsum("npij,nj->npi", cross,
        np.einsum("nij,nj->ni", inverse, nuisance_rhs, optimize=True), optimize=True)
    if coherent:
        partial_h = harmonic - np.einsum("nti,npij->nptj", nuisance,
            np.einsum("nij,npkj->npik", inverse, cross, optimize=True), optimize=True)
        partial_y = mixed - np.einsum("nti,ni->nt", nuisance,
            np.einsum("nij,nj->ni", inverse, nuisance_rhs, optimize=True), optimize=True)
        first = (np.arange(len(time))[None] < count[:, None]/2.)*mask
        halves = np.stack([first, mask & ~first])
        half_rhs = np.einsum("hnt,npti,nt->hnpi", halves, partial_h, partial_y, optimize=True)
        partial_gram = np.einsum("npti,nptj->npij", partial_h, partial_h, optimize=True)
        if half_gram:
            half_design_gram = np.einsum("hnt,npti,nptj->hnpij", halves, partial_h, partial_h, optimize=True)
            half_coefficient = np.linalg.solve(half_design_gram, half_rhs[..., None])[..., 0]
            return np.einsum("npi,npij,npj->np", half_coefficient[0], partial_gram,
                half_coefficient[1], optimize=True)/count[:, None]
        return 4.*np.sum(half_rhs[0]*np.linalg.solve(
            partial_gram, half_rhs[1, :, :, :, None])[..., 0], axis=-1)/count[:, None]
    return np.sum(rhs*np.linalg.solve(gram, rhs[..., None])[..., 0], axis=-1)/count[:, None]


def role_geometry(time, count, mixed, peer, period, linear_envelope=False):
    """Construct both source designs on the actual physical sampling times."""
    lengths, layout = np.unique(count, return_inverse=True)
    mask = np.arange(len(time))[None] < lengths[:, None]
    centered = (time[None] - (mask*time[None]).sum(axis=1)[:, None]/lengths[:, None])*mask
    trend = np.stack([mask/np.sqrt(lengths[:, None]),
        centered/np.sqrt((centered**2).sum(axis=1))[:, None]], axis=-1)
    angle = 2*np.pi*time[None]/period[:, None]
    harmonic = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    trend_cross = np.einsum("lti,ptj->lpij", trend, harmonic, optimize=True)
    design = harmonic[None]*mask[:, None, :, None] - np.einsum(
        "lti,lpij->lptj", trend, trend_cross, optimize=True)
    if linear_envelope:
        envelope=centered/np.sqrt(np.sum(centered**2,axis=1)/lengths)[:,None]
        raw=harmonic[None]*mask[:,None,:,None]
        raw=np.concatenate([raw,raw*envelope[:,None,:,None]],axis=-1)
        zcross=np.einsum("lti,lptj->lpij",trend,raw,optimize=True)
        design=raw-np.einsum("lti,lpij->lptj",trend,zcross,optimize=True)
    cross = np.einsum("lpti,lqtj->lpqij", design, design, optimize=True)
    grid = np.arange(len(period))
    gram = cross[:, grid, grid]
    joint = np.concatenate([
        np.concatenate([np.broadcast_to(gram[:, :, None], cross.shape), cross], axis=-1),
        np.concatenate([cross.swapaxes(-1, -2),
            np.broadcast_to(gram[:, None], cross.shape)], axis=-1)], axis=-2)
    rhs = np.stack([np.einsum("pti,nt->npi", harmonic, mixed, optimize=True),
        np.einsum("pti,nt->npi", harmonic, peer, optimize=True)], axis=-1)
    if linear_envelope:
        rhs=np.stack([np.einsum("npti,nt->npi",design[layout],mixed,optimize=True),
            np.einsum("npti,nt->npi",design[layout],peer,optimize=True)],axis=-1)
    return design, layout, joint, rhs


def role_assignment_evidence(joint_gram, rhs, layout, covariance, background, check_p, check_q, reduction="expected_bf", record_chunk=64):
    """Integrate the same role law in record chunks to bound pair-grid memory."""
    from source_correlated_geometry import correlated_role_assignment_evidence
    return correlated_role_assignment_evidence(joint_gram,rhs,layout,covariance,background,
        check_p,check_q,device="cpu",record_chunk=record_chunk,reduction=reduction)


def joint_source_role_posterior(joint_gram,rhs,layout,covariance,background,period,run_index,mean_log_prior,within_sd,quality=None,device="cuda"):
    """Integrate heterogeneous BG clocks and a shared FG population location.

    Each physical No enters once. Its BG frequency is a nuisance variable with
    the measured-peer conditional distribution. No peer likelihood or anchor is repeated
    in the joint likelihood. Returns marginals of the same joint distribution.
    """
    from scipy.special import logsumexp
    dtype=torch.float64
    axis=np.asarray(period,dtype=float)
    run_index=np.asarray(run_index,dtype=int)
    number=int(run_index.max())+1
    cases=len(covariance)
    difference=axis[:,None]-axis[None]
    precision=np.divide(1.,within_sd**2,out=np.zeros_like(within_sd),where=within_sd>0.)
    transition=-.5*difference[None]**2*precision[:,None,None]
    transition-=logsumexp(transition,axis=-1,keepdims=True)
    point=np.full((len(axis),len(axis)),-np.inf);np.fill_diagonal(point,0.)
    transition=np.where((within_sd==0.)[:,None,None],point[None],transition)
    individual=np.empty((cases,len(rhs),len(axis)))
    mean_probability=np.empty((cases,number,len(axis)))
    bg_probability=np.empty_like(mean_probability)
    ttransition=torch.as_tensor(transition,dtype=dtype,device=device)
    tprior=torch.as_tensor(mean_log_prior,dtype=dtype,device=device)
    tprior=tprior-torch.logsumexp(tprior,dim=-1,keepdim=True)

    def cohort(group):
        selected=np.flatnonzero(run_index==group)
        geometry,local=np.unique(layout[selected],return_inverse=True)
        h=np.asarray(covariance)
        if h.ndim==4:
            q=float(quality[selected[0]])
            h=np.exp((1.-q)*np.log(h[:,0])+q*np.log(h[:,1]))
        h=h[:,0]
        th=torch.as_tensor(h,dtype=dtype,device=device)
        gram=torch.as_tensor(joint_gram[geometry],dtype=dtype,device=device)
        r=torch.as_tensor(rhs[selected,:,:,0],dtype=dtype,device=device)
        li=torch.as_tensor(local,dtype=torch.long,device=device)
        amplitude=torch.repeat_interleave(th[:,:2],2,dim=-1)
        ratio=(amplitude/th[:,2,None]).sqrt()
        matrix=torch.eye(4,dtype=dtype,device=device)[None,None,None,None]+gram[None]*ratio[:,None,None,None,:,None]*ratio[:,None,None,None,None,:]
        inverse=torch.linalg.inv(matrix)[:,li]
        determinant=torch.linalg.slogdet(matrix)[1][:,li]
        pp=torch.einsum("npi,fnpqij,npj->fnpq",r,inverse[...,:2,:2],r)
        qq=torch.einsum("nqi,fnpqij,nqj->fnpq",r,inverse[...,2:,2:],r)
        pq=torch.einsum("npi,fnpqij,nqj->fnpq",r,inverse[...,:2,2:],r)
        gain=(pp*th[:,0,None,None,None]+qq*th[:,1,None,None,None]+2.*pq*(th[:,0]*th[:,1]).sqrt()[:,None,None,None])/th[:,2,None,None,None].square()
        pair=.5*(gain-determinant)
        log_bg=np.log(np.maximum(background[:,selected],np.finfo(float).tiny))
        log_bg-=logsumexp(log_bg,axis=-1,keepdims=True)
        tbg=torch.as_tensor(log_bg,dtype=dtype,device=device)
        likelihood=torch.logsumexp(pair+tbg[:,:,None,:],dim=-1)
        def integrate(record):
            return torch.logsumexp(likelihood[:,record,None]+ttransition,dim=-1)
        integrated=torch.stack(list(map(integrate,np.arange(len(selected)))),dim=1)
        joint=tprior+integrated.sum(1)
        joint=joint-torch.logsumexp(joint,dim=-1,keepdim=True)
        mean_probability[:,group]=joint.exp().cpu().numpy()
        def marginal(record):
            cavity=joint-integrated[:,record]
            field=torch.logsumexp(cavity[:,:,None]+ttransition,dim=-2)
            foreground=likelihood[:,record]+field
            bg=tbg[:,record]+torch.logsumexp(pair[:,record]+field[:,:,None],dim=-2)
            return ((foreground-torch.logsumexp(foreground,dim=-1,keepdim=True)).exp(),
                (bg-torch.logsumexp(bg,dim=-1,keepdim=True)).exp())
        posteriors=list(map(marginal,np.arange(len(selected))))
        individual[:,selected]=torch.stack(list(map(lambda value:value[0],posteriors)),dim=1).cpu().numpy()
        bg_probability[:,group]=torch.stack(list(map(lambda value:value[1],posteriors)),dim=1).mean(1).cpu().numpy()
        return None

    list(map(cohort,np.arange(number)))
    point=individual@axis
    membership=(np.arange(number)[:,None]==run_index[None]).astype(float)
    mean=point@((membership/membership.sum(1)[:,None]).T)
    return {"posterior":individual,"mean_probability":mean_probability,"background_probability":bg_probability,
        "estimate":point,"source_period":point,"group_mean":mean,"deviation":point-mean[:,run_index]}


def conditional_role_assignment_evidence(joint_gram, rhs, layout, covariance, q, background, check_p, check_q, reduction="expected_bf"):
    """Interpolate positive role variances in log space at each cohort's q.

    covariance has shape (F, 2 endpoints, 2 views, 3 variances), and q has
    shape (N,) with an identical value for every record in the same cohort.
    The remaining inputs and returned tuple follow role_assignment_evidence.
    Exact q groups reuse one pair-grid covariance calculation at a time.
    """
    covariance=np.asarray(covariance,dtype=float)
    q=np.asarray(q,dtype=float)
    levels,owner=np.unique(q,return_inverse=True)
    log_covariance=np.log(covariance)
    features=np.empty((len(covariance),len(rhs),rhs.shape[1],rhs.shape[-1]),dtype=float)
    diagnostics=np.empty((len(levels),len(covariance),3),dtype=float)

    def cohort(index):
        selected=np.flatnonzero(owner==index)
        needed,local=np.unique(layout[selected],return_inverse=True)
        current=np.exp((1.-levels[index])*log_covariance[:,0]+levels[index]*log_covariance[:,1])
        fitted,diagnostic=role_assignment_evidence(joint_gram[needed],rhs[selected],local,current,
            background[:,selected],check_p,check_q,reduction=reduction)
        features[:,selected]=fitted
        diagnostics[index]=diagnostic

    list(map(cohort,np.arange(len(levels))))
    diagnostic=np.column_stack([diagnostics[owner[0],:,0],diagnostics[:,:,1].min(axis=0),
        diagnostics[:,:,2].max(axis=0)])
    return features,diagnostic


def foreground_presence_evidence(joint_gram,rhs,layout,covariance,background,device="cpu",record_chunk=64):
    """Marginal No, peer and joint evidence for an added foreground clock.

    The two models share noise and background priors. The foreground-present
    model integrates a uniform period law; the null model sets its amplitude
    to zero, without assigning a period label to the null state.
    """
    dtype=torch.float64
    joint=torch.as_tensor(joint_gram,dtype=dtype,device=device)
    trhs=torch.as_tensor(rhs,dtype=dtype,device=device)
    tlayout=torch.as_tensor(layout,dtype=torch.long,device=device)
    tcov=torch.as_tensor(covariance,dtype=dtype,device=device)
    log_bg=torch.as_tensor(np.log(np.maximum(background,np.finfo(float).tiny)),dtype=dtype,device=device)
    log_periods=np.log(rhs.shape[1])

    def single_case(h,log_q):
        ratio=torch.repeat_interleave((h[:,:2]/h[:,2,None]).sqrt(),2,dim=-1)
        precision=torch.eye(4,dtype=dtype,device=device)[None,None,None,None]+(
            joint[None]*ratio[:,None,None,None,:,None]*ratio[:,None,None,None,None,:])
        inverse=torch.linalg.inv(precision)
        determinant=torch.linalg.slogdet(precision)[1]
        bg_precision=torch.eye(2,dtype=dtype,device=device)[None,None,None]+(
            joint[:,0,:,2:,2:][None]*(h[:,1]/h[:,2])[:,None,None,None,None])
        bg_inverse=torch.linalg.inv(bg_precision)
        bg_determinant=torch.linalg.slogdet(bg_precision)[1]

        def single_record(r,current_layout,current_log_q):
            selected=torch.index_select(inverse,1,current_layout.reshape(1))[:,0]
            ld=torch.index_select(determinant,1,current_layout.reshape(1))[:,0]
            pp=torch.einsum("piv,vpqij,pjv->vpq",r,selected[:,:,:,:2,:2],r)
            qq=torch.einsum("qiv,vpqij,qjv->vpq",r,selected[:,:,:,2:,2:],r)
            pq=torch.einsum("piv,vpqij,qjv->vpq",r,selected[:,:,:,:2,2:],r)
            gain=(pp*h[:,0,None,None]+qq*h[:,1,None,None]
                +2*pq*(h[:,0]*h[:,1]).sqrt()[:,None,None])/h[:,2,None,None].square()
            pair=.5*(gain-ld)
            b_inv=torch.index_select(bg_inverse,1,current_layout.reshape(1))[:,0]
            b_ld=torch.index_select(bg_determinant,1,current_layout.reshape(1))[:,0]
            bg_quad=torch.einsum("qiv,vqij,qjv->vq",r,b_inv,r)
            bg_pair=.5*(bg_quad*h[:,1,None]/h[:,2,None].square()-b_ld)
            separate=torch.logsumexp(pair+current_log_q[None,None],dim=(-1,-2))-log_periods
            separate-=torch.logsumexp(bg_pair+current_log_q[None],dim=-1)
            shared=torch.logsumexp(pair.sum(dim=0)+current_log_q[None],dim=(-1,-2))-log_periods
            shared-=torch.logsumexp(bg_pair.sum(dim=0)+current_log_q,dim=-1)
            return torch.cat([separate,shared.reshape(1)])

        return torch.vmap(single_record,in_dims=(0,0,0),chunk_size=record_chunk)(trhs,tlayout,log_q)

    evidence=torch.vmap(single_case,in_dims=(0,0),chunk_size=1)(tcov,log_bg)
    return evidence.cpu().numpy()


def gain_marginal_role_evidence(joint_gram, rhs, layout, covariance, log_period_prior, check_p, check_q):
    """Compare target/background roles under one shared latent gain per No/peer pair.

    Gaussian amplitudes are integrated analytically. A three-node Gaussian rule
    integrates the existing unit-normal log gain; the two features are No-only
    and joint No/peer posterior role odds. Period priors contain allowed labels.
    """
    joint = torch.as_tensor(joint_gram, dtype=torch.float64)
    tensor_rhs = torch.as_tensor(rhs, dtype=torch.float64)
    tensor_layout = torch.as_tensor(layout, dtype=torch.long)
    record = torch.arange(len(layout))
    log_gain = torch.as_tensor(np.sqrt(3.)*np.asarray([-1., 0., 1.]), dtype=torch.float64)
    log_weight = torch.log(torch.as_tensor([1./6., 2./3., 1./6.], dtype=torch.float64))
    gain_factor = torch.stack([(-log_gain).exp(), log_gain.exp()], dim=-1)

    def single_case(h, log_prior):
        source_variance = h[None, :, :2]*gain_factor[:, None]
        source_variance *= h[:, :2].sum(dim=-1)[None, :, None]/source_variance.sum(dim=-1, keepdim=True)
        amplitude = torch.repeat_interleave(source_variance, 2, dim=-1)
        ratio = (amplitude/h[None, :, 2, None]).sqrt()
        precision = torch.eye(4, dtype=torch.float64)[None, None, None, None, None] + (
            joint[None, None]*ratio[:, :, None, None, None, :, None]
            *ratio[:, :, None, None, None, None, :])
        inverse = torch.linalg.inv(precision)
        pp = torch.einsum("npiv,kvlpqij,npjv->kvlnpq", tensor_rhs,
            inverse[:, :, :, :, :, :2, :2], tensor_rhs)
        qq = torch.einsum("nqiv,kvlpqij,nqjv->kvlnpq", tensor_rhs,
            inverse[:, :, :, :, :, 2:, 2:], tensor_rhs)
        pq = torch.einsum("npiv,kvlpqij,nqjv->kvlnpq", tensor_rhs,
            inverse[:, :, :, :, :, :2, 2:], tensor_rhs)
        gain = (pp*source_variance[:, :, 0, None, None, None, None]
            + qq*source_variance[:, :, 1, None, None, None, None]
            + 2*pq*source_variance.prod(dim=-1).sqrt()[:, :, None, None, None, None]
            )/h[None, :, 2, None, None, None, None].square()
        pair = .5*(gain-torch.linalg.slogdet(precision)[1][:, :, :, None])
        pair = pair[:, :, tensor_layout, record]
        no_pair = torch.logsumexp(pair[:, 0]+log_weight[:, None, None, None], dim=0)
        joint_pair = torch.logsumexp(pair.sum(dim=1)+log_weight[:, None, None, None], dim=0)
        integrated = torch.stack([no_pair, joint_pair])
        target = log_prior[0, None, None] + torch.logsumexp(
            integrated+log_prior[1, None, None, None], dim=-1)
        background = log_prior[1, None, None] + torch.logsumexp(
            integrated.transpose(-1, -2)+log_prior[0, None, None, None], dim=-1)
        assigned = joint_pair+log_prior[0, None, :, None]+log_prior[1, None, None, :]
        diagonal = torch.diagonal(assigned-assigned.transpose(-1, -2), dim1=-2, dim2=-1)
        diagnostic = torch.stack([pair[1, 0, 0, check_p, check_q]-pair[1, 0, 0, check_q, check_p],
            diagonal.min(), diagonal.max(), (source_variance[1]-h[:, :2]).abs().max()])
        return (target-background).permute(1, 2, 0).contiguous(), diagnostic

    features, diagnostic = torch.vmap(single_case, in_dims=(0, 0), chunk_size=1)(
        torch.as_tensor(covariance, dtype=torch.float64), torch.as_tensor(log_period_prior, dtype=torch.float64))
    return features.numpy(), diagnostic.numpy()


def joint_observation_role_evidence(joint_gram,rhs,layout,covariance,background,check_p,check_q,device="cpu",source_covariance=None,
        observation_energy=None,observation_dof=None,mixture_weight=None):
    """Use conditional No-given-peer and marginal-peer source-role likelihoods.

    Optional source coefficient moments support amplitude-envelope bases.
    Without them, source covariances are isotropic in each harmonic pair;
    diagonal view covariances recover the independent role features.
    Multiple source states are marginalized as full observation likelihoods,
    including each state\'s noise normalizer, rather than averaging moments.
    """
    dtype=torch.float64
    joint=torch.as_tensor(joint_gram,dtype=dtype,device=device)
    r=torch.as_tensor(rhs,dtype=dtype,device=device)
    tl=torch.as_tensor(layout,dtype=torch.long,device=device)
    record=torch.arange(len(layout),device=device)
    log_bg=torch.as_tensor(np.log(np.maximum(background,np.finfo(float).tiny)),dtype=dtype,device=device)
    omega_tensor=torch.as_tensor(covariance,dtype=dtype,device=device)
    omega_tensor=omega_tensor[:,None] if omega_tensor.ndim==4 else omega_tensor
    c,k=omega_tensor.shape[:2]
    mixture=np.ones((c,k))/k if mixture_weight is None else np.broadcast_to(mixture_weight,(c,k))
    log_mixture=torch.as_tensor(np.log(mixture.copy()),dtype=dtype,device=device)
    assert k==1 or (observation_energy is not None and observation_dof is not None)
    energy=None if observation_energy is None else torch.as_tensor(observation_energy,dtype=dtype,device=device)
    dof=None if observation_dof is None else torch.as_tensor(observation_dof,dtype=dtype,device=device)
    d=rhs.shape[2];block=2*d;total=4*d
    order=torch.as_tensor(np.r_[np.arange(d),np.arange(2*d,3*d),np.arange(d,2*d),np.arange(3*d,4*d)],dtype=torch.long,device=device)
    eye=torch.eye(d,dtype=dtype,device=device)
    peer_eye=torch.eye(block,dtype=dtype,device=device)
    full_eye=torch.eye(total,dtype=dtype,device=device)
    if source_covariance is None:
        sf=torch.einsum("ckvw,ij->ckviwj",omega_tensor[:,:,0],eye).reshape(c,k,block,block)
        sb=torch.einsum("ckvw,ij->ckviwj",omega_tensor[:,:,1],eye).reshape(c,k,block,block)
        source_tensor=torch.stack([sf,sb],dim=2)
    else:
        source_tensor=torch.as_tensor(source_covariance,dtype=dtype,device=device)
        source_tensor=source_tensor[:,None] if source_tensor.ndim==4 else source_tensor

    def one_component(omega,source):
        noise_inverse=torch.linalg.inv(omega[2])
        sf=source[0];sb=source[1];zero=torch.zeros_like(sf)
        prior=torch.cat([torch.cat([sf,zero],dim=-1),torch.cat([zero,sb],dim=-1)],dim=-2)
        lower=torch.linalg.cholesky(prior)
        weighted_gram=torch.einsum("vw,lpqij->lpqviwj",noise_inverse,joint).reshape(*joint.shape[:3],total,total)
        weighted_gram=weighted_gram[:,:,:,order][:,:,:,:,order]
        precision=full_eye+lower.T@weighted_gram@lower
        kernel=lower@torch.linalg.inv(precision)@lower.T
        weighted_rhs=torch.einsum("npiv,vw->npiw",r,noise_inverse).permute(0,1,3,2).reshape(len(layout),r.shape[1],block)
        pp=torch.einsum("npi,lpqij,npj->lnpq",weighted_rhs,kernel[:,:,:,:block,:block],weighted_rhs)
        qq=torch.einsum("nqi,lpqij,nqj->lnpq",weighted_rhs,kernel[:,:,:,block:,block:],weighted_rhs)
        pq=torch.einsum("npi,lpqij,nqj->lnpq",weighted_rhs,kernel[:,:,:,:block,block:],weighted_rhs)
        all_pair=.5*(pp+qq+2*pq-torch.linalg.slogdet(precision)[1][:,None])
        pair=all_pair[tl,record]
        if energy is not None:
            noise_base=-.5*(dof*torch.linalg.slogdet(omega[2])[1]+torch.einsum("nij,ji->n",energy,noise_inverse))
            pair=pair+noise_base[:,None,None]
        peer_sf=sf[d:,d:];peer_sb=sb[d:,d:];peer_zero=torch.zeros_like(peer_sf)
        peer_prior=torch.cat([torch.cat([peer_sf,peer_zero],dim=-1),torch.cat([peer_zero,peer_sb],dim=-1)],dim=-2)
        peer_lower=torch.linalg.cholesky(peer_prior);noise=omega[2,1,1]
        peer_precision=peer_eye+peer_lower.T@joint@peer_lower/noise
        peer_kernel=peer_lower@torch.linalg.inv(peer_precision)@peer_lower.T/noise.square()
        rp=r[:,:,:,1]
        peer_pp=torch.einsum("npi,lpqij,npj->lnpq",rp,peer_kernel[:,:,:,:d,:d],rp)
        peer_qq=torch.einsum("nqi,lpqij,nqj->lnpq",rp,peer_kernel[:,:,:,d:,d:],rp)
        peer_pq=torch.einsum("npi,lpqij,nqj->lnpq",rp,peer_kernel[:,:,:,:d,d:],rp)
        peer_pair=(.5*(peer_pp+peer_qq+2*peer_pq-torch.linalg.slogdet(peer_precision)[1][:,None]))[tl,record]
        if energy is not None:
            peer_base=-.5*(dof*torch.log(noise)+energy[:,1,1]/noise)
            peer_pair=peer_pair+peer_base[:,None,None]
        return pair,peer_pair

    def one_case(omega,log_rho,source,log_weight):
        joint_component,peer_component=torch.vmap(one_component,in_dims=(0,0),chunk_size=1)(omega,source)
        pair=torch.logsumexp(joint_component+log_weight[:,None,None,None],dim=0)
        peer_pair=torch.logsumexp(peer_component+log_weight[:,None,None,None],dim=0)
        joint_bf=pair-pair.transpose(-1,-2)
        peer_bf=peer_pair-peer_pair.transpose(-1,-2)
        conditional_bf=joint_bf-peer_bf
        evidence=torch.stack([torch.logsumexp(conditional_bf+log_rho[:,None],dim=-1),
            torch.logsumexp(peer_bf+log_rho[:,None],dim=-1)],dim=-1)
        diagonal=torch.diagonal(joint_bf,dim1=-2,dim2=-1)
        diagnostic=torch.stack([joint_bf[0,check_p,check_q],diagonal.min(),diagonal.max()])
        return evidence,diagnostic

    value,diagnostic=torch.vmap(one_case,in_dims=(0,0,0,0),chunk_size=1)(omega_tensor,log_bg,source_tensor,log_mixture)
    return value.cpu().numpy(),diagnostic.cpu().numpy()


def reference_template_phase_diffusion(coefficients,phase,half_rhs,half_gram,template_weight,time,mask,reference,device="cuda"):
    """Estimate phase diffusion from permitted FG waveform references only.

    The second circular moment respects the signed-amplitude ambiguity.
    Diffusion is phase variance per cycle, so physical time units cancel.
    """
    dtype=torch.float64
    c=torch.as_tensor(coefficients,dtype=dtype,device=device)
    r=torch.as_tensor(half_rhs,dtype=dtype,device=device)
    g=torch.as_tensor(half_gram,dtype=dtype,device=device)
    amplitude=torch.einsum("fkd,hnd->fhnk",c,r)
    norm=torch.einsum("fki,hnij,fkj->fhnk",c,g,c).clamp_min(0.)
    log=.5*(amplitude.square()/(norm+1.)-torch.log1p(norm))
    probability=torch.softmax(log,dim=-1).cpu().numpy()
    moment=probability@np.exp(2j*np.asarray(phase))
    disagreement=np.maximum(1.-np.real(moment[:,0]*moment[:,1].conj()),0.)/2.
    mask=np.asarray(mask,dtype=bool);time=np.asarray(time,dtype=float)
    first=np.min(np.where(mask,time[None],np.inf),axis=1)
    last=np.max(np.where(mask,time[None],-np.inf),axis=1)
    left=mask&(time[None]<=(first+last)[:,None]/2.);right=mask&~left
    center_left=(left*time[None]).sum(1)/left.sum(1)
    center_right=(right*time[None]).sum(1)/right.sum(1)
    cycles=(center_right-center_left)/np.asarray(reference)
    return np.sum(template_weight*disagreement/cycles[None],axis=1)


def template_source_log_likelihood(coefficients,phase,rhs,gram,energy,degrees,device="cuda",layout=None,record_chunk=32):
    """Phase-integrated template score with an integrated residual noise scale."""
    dtype=torch.float64
    tr=torch.as_tensor(rhs,dtype=dtype,device=device)
    tg=torch.as_tensor(gram,dtype=dtype,device=device)
    te=torch.as_tensor(energy,dtype=dtype,device=device)
    td=torch.as_tensor(degrees,dtype=dtype,device=device)
    score=np.empty((len(coefficients),rhs.shape[-3],rhs.shape[-2]))
    owner=np.arange(rhs.shape[-3]) if layout is None else np.asarray(layout,dtype=int)
    def case(index):
        c=torch.as_tensor(coefficients[index],dtype=dtype,device=device)
        current_rhs=tr[index] if tr.ndim==4 else tr
        current_energy=te[index] if te.ndim==2 else te
        norm_by_layout=torch.einsum("ki,lpij,kj->lpk",c,tg,c).clamp_min(0.)
        def block(start):
            stop=min(start+record_chunk,rhs.shape[-3])
            amplitude=torch.einsum("kd,npd->npk",c,current_rhs[start:stop])
            norm=norm_by_layout[owner[start:stop]]
            d=td[start:stop,None,None]
            rank=norm>1e-10*(d+4.)*c.square().sum(-1)[None,None]
            denominator=norm*current_energy[start:stop,None,None]
            r2=torch.where(rank&(denominator>0.),amplitude.square()/denominator.clamp_min(torch.finfo(dtype).tiny),0.).clamp(0.,1.)
            log=-.5*rank*torch.log1p(d)-.5*d*torch.log1p(-d/(1.+d)*r2)
            score[index,start:stop]=(torch.logsumexp(log,dim=-1)-np.log(len(phase))).cpu().numpy()
            return None
        list(map(block,np.arange(0,rhs.shape[-3],record_chunk)))
        return None
    list(map(case,np.arange(len(coefficients))))
    return score


def dynamic_template_log_likelihood(coefficients,phase,half_rhs,half_gram,phase_diffusion,period,time,mask,device="cuda",record_chunk=16,half_energy=None,half_degrees=None):
    """Integrate phase drift and amplitude in both observed temporal halves."""
    dtype=torch.float64
    mask=np.asarray(mask,dtype=bool);time=np.asarray(time,dtype=float)
    first=np.min(np.where(mask,time[None],np.inf),axis=1)
    last=np.max(np.where(mask,time[None],-np.inf),axis=1)
    left=mask&(time[None]<=(first+last)[:,None]/2.);right=mask&~left
    separation=(right*time[None]).sum(1)/right.sum(1)-(left*time[None]).sum(1)/left.sum(1)
    cosine=np.cos(2.*(np.asarray(phase)[:,None]-np.asarray(phase)[None]))
    tcos=torch.as_tensor(cosine,dtype=dtype,device=device)
    tr=torch.as_tensor(half_rhs,dtype=dtype,device=device)
    tg=torch.as_tensor(half_gram,dtype=dtype,device=device)
    score=np.empty((len(coefficients),len(mask),len(period)))
    def case(index):
        c=torch.as_tensor(coefficients[index],dtype=dtype,device=device)
        outer=c[:,:,None]*c[:,None,:]
        current_rhs=tr[index] if tr.ndim==5 else tr
        current_energy=half_energy[index] if half_energy is not None and np.ndim(half_energy)==3 else half_energy
        def block(start):
            stop=min(start+record_chunk,len(mask))
            amplitude=torch.einsum("kd,hnpd->hnpk",c,current_rhs[:,start:stop])
            norm=torch.einsum("kij,hnpij->hnpk",outer,tg[:,start:stop]).clamp_min(0.)
            if half_energy is None:
                log=.5*(amplitude.square()/(norm+1.)-torch.log1p(norm))
            else:
                energy=torch.as_tensor(current_energy[:,start:stop],dtype=dtype,device=device)
                degree=torch.as_tensor(half_degrees[:,start:stop],dtype=dtype,device=device)
                rank=norm>1e-10*(degree+4.)[:,:,None,None]*c.square().sum(-1)[None,None,None]
                denominator=norm*energy[:,:,None,None]
                r2=torch.where(rank&(denominator>0.),amplitude.square()/denominator.clamp_min(torch.finfo(dtype).tiny),0.).clamp(0.,1.)
                log=-.5*rank*torch.log1p(degree)[:,:,None,None]-.5*degree[:,:,None,None]*torch.log1p(-degree[:,:,None,None]/(1.+degree[:,:,None,None])*r2)
            variance=phase_diffusion[index]*separation[start:stop,None]/np.asarray(period)[None]
            if phase_diffusion[index]==0.:
                kernel=np.where(np.isclose(cosine,1.,rtol=0.,atol=1e-12),0.,-np.inf)
                log_kernel=torch.as_tensor(kernel,dtype=dtype,device=device)[None,None]
            else:
                concentration=torch.as_tensor(1./(4.*variance),dtype=dtype,device=device)
                log_kernel=concentration[:,:,None,None]*tcos[None,None]
            log_kernel=log_kernel-torch.logsumexp(log_kernel,dim=-1,keepdim=True)
            joint=log[0,:,:, :,None]+log[1,:,:,None,:]+log_kernel
            value=torch.logsumexp(joint,dim=(-1,-2))-np.log(len(phase))
            score[index,start:stop]=value.cpu().numpy()
            return None
        list(map(block,np.arange(0,len(mask),record_chunk)))
        return None
    list(map(case,np.arange(len(coefficients))))
    return score


def role_score_features(source_features, base_features):
    """Return the original 31 role terms in their fitted coefficient order."""
    base = np.broadcast_to(base_features, (*source_features.shape[:-1], base_features.shape[-1]))
    left, right = np.triu_indices(source_features.shape[-1])
    return np.concatenate([source_features,
        (source_features[..., None]*base[..., None, :]).reshape(*source_features.shape[:-1], -1),
        source_features[..., left]*source_features[..., right]], axis=-1)


def interpolate_mean(query, calibration_x, calibration_y, count, identity_tails=False):
    """Interpolate fitted means, optionally extending the endpoint residuals."""
    valid = np.arange(calibration_x.shape[-1]) < count[..., None]
    lower = np.sum(query[..., :, None] >= np.where(
        valid, calibration_x, np.inf)[..., None, :], axis=-1) - 1
    lower = np.maximum(np.minimum(lower, count[..., None]-1), 0)
    upper = np.minimum(lower+1, count[..., None]-1)
    x_lower = np.take_along_axis(calibration_x, lower, axis=-1)
    x_upper = np.take_along_axis(calibration_x, upper, axis=-1)
    y_lower = np.take_along_axis(calibration_y, lower, axis=-1)
    y_upper = np.take_along_axis(calibration_y, upper, axis=-1)
    last = np.broadcast_to(count-1, calibration_x.shape[:-1])[..., None]
    clipped = np.clip(query, calibration_x[..., 0, None],
        np.take_along_axis(calibration_x, last, axis=-1))
    fraction = np.divide(clipped-x_lower, x_upper-x_lower,
        out=np.zeros_like(clipped), where=x_upper > x_lower)
    return y_lower + fraction*(y_upper-y_lower) + np.asarray(identity_tails)[..., None]*(query-clipped)


def fit_monotone_mean(source_mean, target, training_run):
    """Fit the original equally weighted monotone means on allowed experiments."""
    target = np.broadcast_to(target, source_mean.shape)
    count = training_run.sum(axis=-1)
    position = np.arange(source_mean.shape[-1])
    order = np.argsort(np.where(training_run, source_mean, np.inf), axis=-1)
    sorted_x = np.take_along_axis(source_mean, order, axis=-1)
    eligible = position < count[..., None]
    sorted_y = np.where(eligible, np.take_along_axis(target, order, axis=-1), 0.)
    prefix = np.pad(np.cumsum(sorted_y, axis=-1), ((0, 0),)*(source_mean.ndim-1)+((1, 0),))
    interval_sum = prefix[..., None, 1:] - prefix[..., :-1, None]
    interval_n = position[None, :] - position[:, None] + 1
    valid = (interval_n > 0) & (position < count[..., None, None])
    interval_mean = np.divide(interval_sum, interval_n,
        out=np.full_like(interval_sum, np.inf), where=valid)
    suffix_minimum = np.minimum.accumulate(interval_mean[..., ::-1], axis=-1)[..., ::-1]
    fitted_y = np.max(np.where(position[:, None] <= position[None, :],
        suffix_minimum, -np.inf), axis=-2)
    fitted_y = np.where(eligible, fitted_y, 0.)
    return interpolate_mean(source_mean, sorted_x, fitted_y, count), sorted_x, fitted_y, count


def fit_group_label_shift(posterior, period, training, target, membership, solver, prior_kind="label"):
    """Fit one unlabeled exponential tilt of the training period law per batch.

    The likelihood is the Saerens prior-shift likelihood, restricted to a
    one-parameter exponential family on the unchanged period grid. The
    standardized tilt has a unit Gaussian prior. Its likelihood uses every
    input record; only the original training labels define the source law.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp, softmax
    weight = training/training.sum(axis=-1)[:, None]
    center = weight@target
    scale = np.sqrt(weight@(target**2)-center**2)
    bandwidth = (4./(3*training.sum(axis=-1)))**.2*scale
    density = softmax(-.5*((period[None, None]-target[None, :, None])
        /bandwidth[:, None, None])**2, axis=-1)
    label_prior = np.einsum("gn,gnp->gp", weight, density, optimize=True)
    model_prior = np.einsum("gn,gnp->gp", weight, posterior, optimize=True)
    prior = {"label": label_prior, "posterior": model_prior}[prior_kind]
    z = (period[None]-center[:, None])/scale[:, None]
    log_prior = np.log(np.maximum(prior, np.finfo(float).tiny))
    log_posterior = np.log(np.maximum(posterior, np.finfo(float).tiny))
    run = membership.argmax(axis=0)
    count = membership.sum(axis=1)
    shape = (len(training), len(membership))

    def objective(parameter):
        tilt = parameter.reshape(shape)
        query_score = log_posterior + tilt[:, run, None]*z[:, None]
        prior_score = log_prior[:, None] + tilt[:, :, None]*z[:, None]
        query_log_partition = logsumexp(query_score, axis=-1)
        prior_log_partition = logsumexp(prior_score, axis=-1)
        query_moment = np.sum(softmax(query_score, axis=-1)*z[:, None], axis=-1)
        prior_moment = np.sum(softmax(prior_score, axis=-1)*z[:, None], axis=-1)
        loss = -query_log_partition@membership.T + count[None]*prior_log_partition + .5*tilt**2
        gradient = -query_moment@membership.T + count[None]*prior_moment + tilt
        return loss.sum()/len(training), gradient.ravel()/len(training)

    initial = np.zeros(shape).ravel()
    value, gradient = objective(initial)
    probe = np.zeros_like(initial)
    probe[0] = 1e-4
    gradient_check = abs((objective(initial+probe)[0]-objective(initial-probe)[0])/(2e-4)-gradient[0])
    assert gradient_check < 1e-7
    fit = minimize(objective, initial, jac=True, method="L-BFGS-B", options=solver)
    tilt = fit.x.reshape(shape)
    adjusted = softmax(log_posterior + tilt[:, run, None]*z[:, None], axis=-1)
    mean = (adjusted@period)@ (membership/count[:, None]).T
    return mean, tilt, fit, gradient_check


def generic_coherent_fwl_power(time,mask,mixed,peer,period,half_gram=False,
                               period_chunk=16,quadrature_modes=64,rcond=1e-10,harmonics=1,return_statistics=False,reference_signal=None):
    """Cross-half harmonic energy in the same observed-time FWL metric.

    Harmonic/trend geometry is shared by acquisition mask. Record-specific
    peer removal uses exact low-rank Gram updates, retaining every observed
    coordinate and the signed cross-half estimate.
    """
    from scipy.signal import hilbert
    time=np.asarray(time,dtype=float);mask=np.asarray(mask,dtype=bool)
    mixed=np.asarray(mixed,dtype=float);peer=np.asarray(peer,dtype=float)
    period=np.asarray(period,dtype=float)
    if return_statistics=="source":
        order=np.argsort(time)
        patterns,owner=np.unique(mask[:,order],axis=0,return_inverse=True)
        dimension=2*harmonics
        statistics={"ordinary_rhs":np.zeros((len(mask),len(period),dimension)),
            "nuisance_rhs":np.zeros((len(mask),len(period),dimension,2)),
            "ordinary_half_rhs":np.zeros((2,len(mask),len(period),dimension)),
            "nuisance_half_rhs":np.zeros((2,len(mask),len(period),dimension,2)),
            "ordinary_gram":np.zeros((len(patterns),len(period),dimension,dimension)),
            "ordinary_half_gram":np.zeros((2,len(patterns),len(period),dimension,dimension)),
            "ordinary_layout":owner,"raw_covariance":np.zeros((len(mask),3,3)),
            "half_raw_covariance":np.zeros((2,len(mask),3,3)),"nuisance_coefficient":np.zeros((len(mask),2)),
            "energy":np.zeros(len(mask)),"half_energy":np.zeros((2,len(mask))),
            "count":mask.sum(1),"half_count":np.zeros((2,len(mask)),dtype=int),
            "nuisance_rank":np.zeros(len(mask),dtype=int)}
        if reference_signal is not None:
            reference_signal=np.asarray(reference_signal,dtype=float)
            statistics.update({"reference_cross":np.zeros((len(mask),3)),"reference_energy":np.zeros(len(mask))})
        def source_acquisition(layout):
            records=np.flatnonzero(owner==layout)
            observed=order[patterns[layout]]
            count=len(observed)
            if count<3:
                return None
            actual=time[observed];duration=actual[-1]-actual[0]
            if duration<=0.:
                return None
            t=(actual-actual[0])/duration
            centered=t-t.mean()
            trend=np.column_stack([np.ones(count)/np.sqrt(count),centered/np.linalg.norm(centered)])
            left=t<=.5;right=~left
            y=mixed[records[:,None],observed[None]].copy()
            c=peer[records[:,None],observed[None]].copy()
            y-=(y@trend)@trend.T
            spacing=np.diff(t)
            regular=np.allclose(spacing,spacing[0],rtol=1e-10,atol=abs(spacing[0])*1e-12)
            if regular:
                quadrature=hilbert(c,axis=-1).imag
            else:
                window=1.+np.median(spacing[spacing>0.])
                modes=min(quadrature_modes,(count-1)//2,max(1,int(np.ceil(2.*window*duration/period.min()))))
                angle=2.*np.pi*t[:,None]*np.arange(1,modes+1)/window
                cosine,sine=np.cos(angle),np.sin(angle)
                fourier=np.column_stack([cosine,sine]);fourier-=fourier.mean(axis=0)
                fourier_inverse=np.linalg.inv(fourier.T@fourier+rcond*count*np.eye(2*modes))
                coefficient=(c@fourier)@fourier_inverse.T
                quadrature=coefficient[:,:modes]@sine.T-coefficient[:,modes:]@cosine.T
            nuisance=np.stack([c,quadrature],axis=-1)
            nuisance-=np.einsum("ti,bij->btj",trend,np.einsum("ti,btj->bij",trend,nuisance,optimize=True),optimize=True)
            raw=np.concatenate([y[:,:,None],nuisance],axis=-1)
            raw0=raw[:,left];raw1=raw[:,right]
            covariance=np.einsum("bti,btj->bij",raw,raw,optimize=True)
            covariance0=np.einsum("bti,btj->bij",raw0,raw0,optimize=True)
            covariance1=np.einsum("bti,btj->bij",raw1,raw1,optimize=True)
            value,vector=np.linalg.eigh((covariance[:,1:,1:]+covariance[:,1:,1:].swapaxes(-1,-2))/2.)
            weight=np.divide(1.,value,out=np.zeros_like(value),where=value>rcond*count)
            inverse=(vector*weight[:,None,:])@vector.swapaxes(-1,-2)
            ols=np.einsum("bij,bj->bi",inverse,covariance[:,1:,0],optimize=True)
            residual=y-np.einsum("btj,bj->bt",nuisance,ols,optimize=True)
            statistics["raw_covariance"][records]=covariance
            statistics["half_raw_covariance"][:,records]=np.stack([covariance0,covariance1])
            statistics["nuisance_coefficient"][records]=ols
            statistics["energy"][records]=np.sum(residual**2,axis=1)
            statistics["half_energy"][:,records]=np.stack([np.sum(residual[:,left]**2,axis=1),np.sum(residual[:,right]**2,axis=1)])
            statistics["half_count"][:,records]=np.asarray([left.sum(),right.sum()])[:,None]
            statistics["nuisance_rank"][records]=np.sum(value>rcond*count,axis=1)
            if reference_signal is not None:
                teacher=reference_signal[records[:,None],observed[None]].copy()
                teacher-=(teacher@trend)@trend.T
                statistics["reference_cross"][records]=np.einsum("bti,bt->bi",raw,teacher,optimize=True)
                statistics["reference_energy"][records]=np.sum(teacher**2,axis=1)
            def source_geometry(start):
                stop=min(start+period_chunk,len(period))
                angle=2.*np.pi*t[None,:,None]/(period[start:stop]/duration)[:,None,None]*np.arange(1,harmonics+1)[None,None]
                h=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(stop-start,count,dimension)
                h-=np.einsum("ti,pij->ptj",trend,np.einsum("ti,ptj->pij",trend,h,optimize=True),optimize=True)
                rhs0=np.einsum("pti,btj->bpij",h[:,left],raw0,optimize=True)
                rhs1=np.einsum("pti,btj->bpij",h[:,right],raw1,optimize=True)
                statistics["ordinary_rhs"][records,start:stop]=rhs0[:,:,:,0]+rhs1[:,:,:,0]
                statistics["nuisance_rhs"][records,start:stop]=rhs0[:,:,:,1:]+rhs1[:,:,:,1:]
                statistics["ordinary_half_rhs"][:,records,start:stop]=np.stack([rhs0[:,:,:,0],rhs1[:,:,:,0]])
                statistics["nuisance_half_rhs"][:,records,start:stop]=np.stack([rhs0[:,:,:,1:],rhs1[:,:,:,1:]])
                statistics["ordinary_gram"][layout,start:stop]=np.einsum("pti,ptj->pij",h,h,optimize=True)
                statistics["ordinary_half_gram"][:,layout,start:stop]=np.stack([
                    np.einsum("pti,ptj->pij",h[:,left],h[:,left],optimize=True),
                    np.einsum("pti,ptj->pij",h[:,right],h[:,right],optimize=True)])
                return None
            list(map(source_geometry,np.arange(0,len(period),period_chunk)))
            return None
        list(map(source_acquisition,np.arange(len(patterns))))
        return statistics
    result=np.zeros((len(mask),len(period)),dtype=float)
    statistics=None
    if return_statistics:
        reference_signal=None if reference_signal is None else np.asarray(reference_signal,dtype=float)
        statistics={"rhs":np.zeros((len(mask),len(period),2*harmonics)),
            "gram":np.zeros((len(mask),len(period),2*harmonics,2*harmonics)),
            "half_rhs":np.zeros((2,len(mask),len(period),2*harmonics)),
            "half_gram":np.zeros((2,len(mask),len(period),2*harmonics,2*harmonics)),
            "energy":np.zeros(len(mask)),"half_energy":np.zeros((2,len(mask))),
            "count":mask.sum(1),"half_count":np.zeros((2,len(mask)),dtype=int),
            "nuisance_rank":np.zeros(len(mask),dtype=int),
            "raw_covariance":np.zeros((len(mask),3,3)),
            "half_raw_covariance":np.zeros((2,len(mask),3,3)),
            "nuisance_coefficient":np.zeros((len(mask),2)),
            "ordinary_rhs":np.zeros((len(mask),len(period),2*harmonics)),
            "nuisance_rhs":np.zeros((len(mask),len(period),2*harmonics,2)),
            "ordinary_half_rhs":np.zeros((2,len(mask),len(period),2*harmonics)),
            "nuisance_half_rhs":np.zeros((2,len(mask),len(period),2*harmonics,2))}
        if reference_signal is not None:
            statistics.update({"reference_cross":np.zeros((len(mask),3)),"reference_energy":np.zeros(len(mask))})
    order=np.argsort(time)
    patterns,owner=np.unique(mask[:,order],axis=0,return_inverse=True)
    if return_statistics:
        statistics.update({"ordinary_layout":owner,
            "ordinary_gram":np.zeros((len(patterns),len(period),2*harmonics,2*harmonics)),
            "ordinary_half_gram":np.zeros((2,len(patterns),len(period),2*harmonics,2*harmonics))})
    def inverse_metric(gram,count):
        value,vector=np.linalg.eigh((gram+gram.swapaxes(-1,-2))/2.)
        weight=np.divide(1.,value,out=np.zeros_like(value),where=value>rcond*count)
        return (vector*weight[...,None,:])@vector.swapaxes(-1,-2)
    def acquisition(layout):
        records=np.flatnonzero(owner==layout);observed=order[patterns[layout]]
        count=len(observed)
        if count<3:
            return None
        actual=time[observed];duration=actual[-1]-actual[0]
        if duration<=0.:
            return None
        t=(actual-actual[0])/duration
        centered=t-t.mean()
        trend=np.column_stack([np.ones(count)/np.sqrt(count),centered/np.linalg.norm(centered)])
        left=t<=.5;right=~left
        spacing=np.diff(t)
        regular=np.allclose(spacing,spacing[0],rtol=1e-10,atol=abs(spacing[0])*1e-12)
        def geometry(start):
            stop=min(start+period_chunk,len(period))
            angle=2.*np.pi*t[None,:,None]/(period[start:stop]/duration)[:,None,None]*np.arange(1,harmonics+1)[None,None]
            h=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(stop-start,count,2*harmonics)
            h-=np.einsum("ti,pij->ptj",trend,np.einsum("ti,ptj->pij",trend,h,optimize=True),optimize=True)
            gram=np.einsum("pti,ptj->pij",h,h,optimize=True)
            g0=np.einsum("pti,ptj->pij",h[:,left],h[:,left],optimize=True)
            g1=np.einsum("pti,ptj->pij",h[:,right],h[:,right],optimize=True)
            if return_statistics:
                statistics["ordinary_gram"][layout,start:stop]=gram
                statistics["ordinary_half_gram"][:,layout,start:stop]=np.stack([g0,g1])
            return start,stop,h,gram,g0,g1
        cache=list(map(geometry,np.arange(0,len(period),period_chunk)))
        if not regular:
            window=1.+np.median(spacing[spacing>0.])
            modes=min(quadrature_modes,(count-1)//2,max(1,int(np.ceil(2.*window*duration/period.min()))))
            angle=2.*np.pi*t[:,None]*np.arange(1,modes+1)/window
            cosine,sine=np.cos(angle),np.sin(angle)
            fourier=np.column_stack([cosine,sine]);fourier-=fourier.mean(axis=0)
            fourier_inverse=np.linalg.inv(fourier.T@fourier+rcond*count*np.eye(2*modes))
        def record(index):
            y=mixed[index,observed].copy();c=peer[index,observed].copy()
            y-=trend@(trend.T@y)
            if regular:
                quadrature=hilbert(c).imag
            else:
                coefficient=fourier_inverse@(fourier.T@c)
                quadrature=sine@coefficient[:modes]-cosine@coefficient[modes:]
            nuisance=np.column_stack([c,quadrature]);nuisance-=trend@(trend.T@nuisance)
            inverse=inverse_metric(nuisance.T@nuisance,count)
            if return_statistics:
                raw=np.column_stack([y,nuisance])
                nuisance_coefficient=inverse@(nuisance.T@y)
                statistics["raw_covariance"][index]=raw.T@raw
                statistics["half_raw_covariance"][:,index]=np.stack([raw[left].T@raw[left],raw[right].T@raw[right]])
                statistics["nuisance_coefficient"][index]=nuisance_coefficient
                if reference_signal is not None:
                    teacher=reference_signal[index,observed].copy()
                    teacher-=trend@(trend.T@teacher)
                    statistics["reference_cross"][index]=raw.T@teacher
                    statistics["reference_energy"][index]=teacher@teacher
            y-=nuisance@(inverse@(nuisance.T@y))
            if return_statistics:
                statistics["energy"][index]=np.sum(y**2)
                statistics["half_energy"][:,index]=np.asarray([np.sum(y[left]**2),np.sum(y[right]**2)])
                statistics["half_count"][:,index]=np.asarray([left.sum(),right.sum()])
                statistics["nuisance_rank"][index]=np.sum(np.linalg.eigvalsh(nuisance.T@nuisance)>rcond*count)
            u0=nuisance[left];u1=nuisance[right]
            ug0=u0.T@u0;ug1=u1.T@u1
            uy0=u0.T@y[left];uy1=u1.T@y[right]
            def evaluate(piece):
                start,stop,h,ordinary,g0,g1=piece
                cross=np.einsum("pti,tj->pij",h,nuisance,optimize=True)
                cross0=np.einsum("pti,tj->pij",h[:,left],u0,optimize=True)
                cross1=np.einsum("pti,tj->pij",h[:,right],u1,optimize=True)
                projection=cross@inverse
                transpose=projection.swapaxes(-1,-2)
                gram=ordinary-projection@cross.swapaxes(-1,-2)
                rhs0=np.einsum("pti,t->pi",h[:,left],y[left],optimize=True)-np.einsum("pij,j->pi",projection,uy0,optimize=True)
                rhs1=np.einsum("pti,t->pi",h[:,right],y[right],optimize=True)-np.einsum("pij,j->pi",projection,uy1,optimize=True)
                if half_gram or return_statistics:
                    gram0=g0-cross0@transpose-projection@cross0.swapaxes(-1,-2)+projection@ug0@transpose
                    gram1=g1-cross1@transpose-projection@cross1.swapaxes(-1,-2)+projection@ug1@transpose
                if return_statistics:
                    statistics["rhs"][index,start:stop]=rhs0+rhs1
                    statistics["gram"][index,start:stop]=gram
                    statistics["half_rhs"][:,index,start:stop]=np.stack([rhs0,rhs1])
                    statistics["half_gram"][:,index,start:stop]=np.stack([gram0,gram1])
                    statistics["ordinary_rhs"][index,start:stop]=rhs0+rhs1+np.einsum("pij,j->pi",cross,nuisance_coefficient,optimize=True)
                    statistics["nuisance_rhs"][index,start:stop]=cross
                    ordinary0=rhs0+np.einsum("pij,j->pi",projection,uy0,optimize=True)+np.einsum("pij,j->pi",cross0,nuisance_coefficient,optimize=True)
                    ordinary1=rhs1+np.einsum("pij,j->pi",projection,uy1,optimize=True)+np.einsum("pij,j->pi",cross1,nuisance_coefficient,optimize=True)
                    statistics["ordinary_half_rhs"][:,index,start:stop]=np.stack([ordinary0,ordinary1])
                    statistics["nuisance_half_rhs"][:,index,start:stop]=np.stack([cross0,cross1])
                elif half_gram:
                    beta0=(inverse_metric(gram0,count)@rhs0[:,:,None])[:,:,0]
                    beta1=(inverse_metric(gram1,count)@rhs1[:,:,None])[:,:,0]
                    result[index,start:stop]=np.einsum("pi,pij,pj->p",beta0,gram,beta1,optimize=True)/count
                else:
                    result[index,start:stop]=4.*np.einsum("pi,pij,pj->p",rhs0,inverse_metric(gram,count),rhs1,optimize=True)/count
                return None
            list(map(evaluate,cache))
            return None
        list(map(record,records))
        return None
    list(map(acquisition,np.arange(len(patterns))))
    return statistics if return_statistics else result


def _dual_stationary_ou_whiten(values,time,tau):
    """Stationary OU innovations on observed coordinates, time axis second-last."""
    actual=torch.as_tensor(np.diff(np.asarray(time,dtype=float)),dtype=values.dtype,device=values.device)
    scale=torch.as_tensor(np.asarray(tau,dtype=float),dtype=values.dtype,device=values.device)
    ratio=actual[None]/scale[:,None]
    shape=(len(scale),)+(1,)*(values.ndim-3)+(len(actual),1)
    correlation=torch.exp(-ratio).reshape(shape)
    innovation=torch.sqrt(-torch.expm1(-2.*ratio)).reshape(shape)
    tail=(values[...,1:,:]-correlation*values[...,:-1,:])/innovation
    return torch.cat([values[...,:1,:],tail],dim=-2)

def dual_source_template_role_evidence(time, mask, signals, period, covariance, background,
        template_coefficients, background_period=None, background_harmonics=1,
        device="cpu", period_chunk=8, record_chunk=4, return_pair=False, gram_method="direct", foreground_prior="gaussian", bg_harmonic_power=None, noise_tau=None):
    """Joint raw-observation FG-template/BG-harmonic evidence with both role orders.

    signals is (record,time,view); covariance is (view,3) or (record,view,3),
    containing the existing FG/BG/noise variances. Each view retains its own
    phase integral, as in the original independent-view role law. BG harmonic
    coefficients and two analytic FG-template coefficients integrate exactly.
    Only one foreground physical phase remains to quadrature. An H1 template
    gives exactly the original isotropic H1 Gaussian role law. Templates have
    unit coefficient norm; BG power is split across its retained harmonics.
    Use the same resolved period grid for both roles in production inference.
    gram_method changes computation only: analytic streams temporal designs
    and obtains exact self/cross Grams from harmonic_cross_gram. The optional
    moment prior is radial in the two FG coefficients only; halving its
    Gaussian base variance preserves the original total foreground power.
    noise_tau is a physical-time OU constant per view. Observations, harmonic
    designs and trends share its stationary innovation metric; physical source
    variances are unchanged and whitened observations are never RMS-scaled.
    """
    time=np.asarray(time,dtype=float);mask=np.asarray(mask,dtype=bool)
    signals=np.asarray(signals,dtype=float);period=np.asarray(period,dtype=float)
    other=period if background_period is None else np.asarray(background_period,dtype=float)
    coefficient=np.asarray(template_coefficients,dtype=float)
    harmonics=coefficient.shape[-1]//2
    power=np.full(background_harmonics,1./background_harmonics) if bg_harmonic_power is None else np.asarray(bg_harmonic_power,dtype=float)
    assert power.shape==(background_harmonics,) and np.all(power>=0.) and np.isclose(power.sum(),1.)
    coefficient=coefficient/np.linalg.norm(coefficient,axis=-1,keepdims=True)
    quadrature=coefficient.reshape(len(coefficient),harmonics,2)[:,:,[1,0]].copy()
    quadrature[:,:,0]*=-1.
    transform=np.stack([coefficient,quadrature.reshape(coefficient.shape)],axis=-1)
    observations=len(mask);views=signals.shape[-1]
    tau=np.zeros(views) if noise_tau is None else np.broadcast_to(np.asarray(noise_tau,dtype=float),(views,)).copy()
    assert np.all(np.isfinite(tau)) and np.all(tau>=0.)
    colored=bool(np.any(tau>0.))
    assert not colored or gram_method=="direct"
    law=np.broadcast_to(np.asarray(covariance,dtype=float),(observations,views,3)).copy()
    log_background=np.log(np.asarray(background,dtype=float))
    assert signals.shape[:2]==mask.shape and len(time)==mask.shape[1]
    assert log_background.shape==(observations,len(other)) and np.all(law>0.)
    assert coefficient.shape[1]==2*harmonics and background_harmonics>=1
    result=np.empty((observations,len(period),views))
    likelihood=np.empty((observations,len(period)))
    pairs=np.empty((observations,len(period),len(other))) if return_pair else None
    patterns,owner=np.unique(mask,axis=0,return_inverse=True)
    same_axis=np.array_equal(period,other)
    assert gram_method in ("direct","analytic")
    assert foreground_prior in ("gaussian","moment")
    dtype=torch.float64
    tensor=lambda value:torch.as_tensor(np.ascontiguousarray(value),dtype=dtype,device=device)
    shape_transform=tensor(transform)
    bg_eye=torch.eye(2*background_harmonics,dtype=dtype,device=device)
    background_scale=tensor(np.repeat(np.sqrt(power),2))

    def acquisition(index):
        observed=np.flatnonzero(patterns[index]);records=np.flatnonzero(owner==index)
        observed=observed[np.argsort(time[observed],kind="stable")]
        actual=time[observed]-time[observed[0]]
        centered=actual-actual.mean()
        trend=np.column_stack([np.ones(len(actual))/np.sqrt(len(actual)),centered/np.linalg.norm(centered)])
        trend_tensor=tensor(trend)
        if colored:
            white_trend=_dual_stationary_ou_whiten(trend_tensor[None].expand(views,-1,-1),actual,tau)
            trend_tensor=torch.linalg.qr(white_trend,mode="reduced")[0]
        def design(axis,order):
            angle=2.*np.pi*actual[None,:,None]/axis[:,None,None]*np.arange(1,order+1)[None,None]
            raw=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(len(axis),len(actual),2*order)
            value=tensor(raw)
            if colored:
                value=_dual_stationary_ou_whiten(value[None].expand(views,-1,-1,-1),actual,tau)
                return value-torch.einsum("vti,vpij->vptj",trend_tensor,torch.einsum("vti,vptj->vpij",trend_tensor,value))
            return value-torch.einsum("ti,pij->ptj",trend_tensor,torch.einsum("ti,ptj->pij",trend_tensor,value))
        observed_signals=signals[records[:,None],observed[None],:].copy()
        if colored:
            data=_dual_stationary_ou_whiten(tensor(observed_signals).permute(2,0,1)[:,:,:,None],actual,tau)[:,:,:,0].permute(1,2,0)
            data-=torch.einsum("vti,niv->ntv",trend_tensor,torch.einsum("vti,ntv->niv",trend_tensor,data))
        else:
            observed_signals-=np.einsum("ti,niv->ntv",trend,np.einsum("ti,ntv->niv",trend,observed_signals,optimize=True),optimize=True)
            data=tensor(observed_signals)
        rhs=lambda value:torch.einsum("vpti,ntv->npiv",value,data) if colored else torch.einsum("pti,ntv->npiv",value,data)
        if gram_method=="direct":
            fg_p=design(period,harmonics)
            fg_q=fg_p if same_axis else design(other,harmonics)
            bg_p=fg_p if background_harmonics==harmonics else design(period,background_harmonics)
            bg_q=bg_p if same_axis else (fg_q if background_harmonics==harmonics else design(other,background_harmonics))
            gram=lambda value:torch.einsum("vpti,vptj->vpij",value,value) if colored else torch.einsum("pti,ptj->pij",value,value)
            gf_p=gram(fg_p);gf_q=gf_p if same_axis else gram(fg_q)
            gb_p=gf_p if background_harmonics==harmonics else gram(bg_p)
            gb_q=gb_p if same_axis else (gf_q if background_harmonics==harmonics else gram(bg_q))
            rf_p=rhs(fg_p);rf_q=rf_p if same_axis else rhs(fg_q)
            rb_p=rf_p if background_harmonics==harmonics else rhs(bg_p)
            rb_q=rb_p if same_axis else (rf_q if background_harmonics==harmonics else rhs(bg_q))
        else:
            def axis_statistics(axis):
                gf=torch.empty((len(axis),2*harmonics,2*harmonics),dtype=dtype,device=device)
                gb=gf if background_harmonics==harmonics else torch.empty((len(axis),2*background_harmonics,2*background_harmonics),dtype=dtype,device=device)
                rf=torch.empty((len(records),len(axis),2*harmonics,views),dtype=dtype,device=device)
                rb=rf if background_harmonics==harmonics else torch.empty((len(records),len(axis),2*background_harmonics,views),dtype=dtype,device=device)
                def statistics_block(first):
                    last=min(first+period_chunk,len(axis));current=axis[first:last]
                    geometry=harmonic_cross_gram(actual,None,current,current,harmonics,background_harmonics)
                    gf[first:last]=tensor(geometry["fg_gram"])
                    rf[:,first:last]=rhs(design(current,harmonics))
                    if background_harmonics!=harmonics:
                        gb[first:last]=tensor(geometry["bg_gram"])
                        rb[:,first:last]=rhs(design(current,background_harmonics))
                    return None
                list(map(statistics_block,np.arange(0,len(axis),period_chunk)))
                return gf,gb,rf,rb
            gf_p,gb_p,rf_p,rb_p=axis_statistics(period)
            gf_q,gb_q,rf_q,rb_q=(gf_p,gb_p,rf_p,rb_p) if same_axis else axis_statistics(other)
        unique_law,law_owner=np.unique(law[records].reshape(len(records),-1),axis=0,return_inverse=True)

        def law_group(law_index):
            local=np.flatnonzero(law_owner==law_index);selected=records[local]
            h=tensor(unique_law[law_index].reshape(views,3))
            prior=tensor(log_background[selected])
            foreground_p=rf_p[local];foreground_q=rf_q[local]
            background_p=rb_p[local];background_q=rb_q[local]
            def background_cache(geometry,observed_rhs,view):
                variance=h[view]
                ratio=variance[1]/variance[2]
                white_geometry=geometry*background_scale[None,:,None]*background_scale[None,None,:]
                precision=bg_eye+ratio*white_geometry
                inverse=torch.linalg.inv(precision)
                metric=ratio*inverse*background_scale[None,:,None]*background_scale[None,None,:]
                weighted_rhs=torch.einsum("qij,nqj->nqi",metric,observed_rhs)
                gain=(observed_rhs*weighted_rhs).sum(-1)/variance[2]
                score=.5*(gain-torch.linalg.slogdet(precision)[1][None])
                return {"metric":metric,"rhs":weighted_rhs,"score":score}
            def one_background(view):
                p_cache=background_cache(gb_p[view] if colored else gb_p,background_p[:,:,:,view],view)
                q_cache=p_cache if same_axis else background_cache(gb_q[view] if colored else gb_q,background_q[:,:,:,view],view)
                return p_cache,q_cache
            caches=list(map(one_background,np.arange(views)))

            def period_block(first):
                last=min(first+period_chunk,len(period))
                if gram_method=="direct":
                    if colored:
                        cross_forward=torch.einsum("vfti,vqtj->vfqij",fg_p[:,first:last],bg_q)
                        cross_reverse=cross_forward.transpose(1,2).transpose(-1,-2) if same_axis and harmonics==background_harmonics else torch.einsum("vfti,vqtj->vfqij",fg_q,bg_p[:,first:last])
                    else:
                        cross_forward=torch.einsum("fti,qtj->fqij",fg_p[first:last],bg_q)
                        cross_reverse=cross_forward.transpose(0,1).transpose(-1,-2) if same_axis and harmonics==background_harmonics else torch.einsum("fti,qtj->fqij",fg_q,bg_p[first:last])
                else:
                    geometry=harmonic_cross_gram(actual,None,period[first:last],other,harmonics,background_harmonics)
                    cross_forward=tensor(geometry["cross"])
                    if same_axis and harmonics==background_harmonics:
                        cross_reverse=cross_forward.transpose(0,1).transpose(-1,-2)
                    else:
                        reverse_geometry=harmonic_cross_gram(actual,None,other,period[first:last],harmonics,background_harmonics)
                        cross_reverse=tensor(reverse_geometry["cross"])
                def orientation(foreground_gram,foreground_rhs,cross,bg_metric,bg_rhs,bg_score,view):
                    variance=h[view]
                    conditional_gram=foreground_gram[:,None]-torch.einsum("fqia,qab,fqjb->fqij",cross,bg_metric,cross)
                    template_gram=torch.einsum("kia,fqij,kjb->fqkab",shape_transform,conditional_gram,shape_transform)
                    foreground_variance=variance[0]/(2. if foreground_prior=="moment" else 1.)
                    fg_ratio=foreground_variance/variance[2]
                    a=1.+fg_ratio*template_gram[:,:,:,0,0]
                    b=fg_ratio*template_gram[:,:,:,0,1]
                    c=fg_ratio*template_gram[:,:,:,1,0]
                    d=1.+fg_ratio*template_gram[:,:,:,1,1]
                    determinant=a*d-b*c
                    multiplier=foreground_variance/variance[2].square()/determinant
                    quadratic_00=d*multiplier;quadratic_11=a*multiplier
                    quadratic_01=-(b+c)*multiplier
                    log_determinant=torch.log(determinant)
                    if foreground_prior=="moment":
                        radial_covariance=.5*(a+d)/determinant
                        radial_multiplier=foreground_variance/(2.*variance[2].square()*determinant.square())
                    def records_block(start):
                        stop=min(start+record_chunk,len(local))
                        conditional_rhs=foreground_rhs[start:stop,:,None]-torch.einsum("fqia,nqa->nfqi",cross,bg_rhs[start:stop])
                        amplitude=torch.einsum("kia,nfqi->nfqka",shape_transform,conditional_rhs)
                        left=amplitude[:,:,:,:,0];right=amplitude[:,:,:,:,1]
                        gain=left.square()*quadratic_00[None]+right.square()*quadratic_11[None]+left*right*quadratic_01[None]
                        score=.5*(gain-log_determinant[None])
                        if foreground_prior=="moment":
                            radial_mean=(d[None]*left-b[None]*right).square()+(a[None]*right-c[None]*left).square()
                            radial_factor=radial_covariance[None]+radial_multiplier[None]*radial_mean
                            score+=torch.log(radial_factor)
                        return torch.logsumexp(score,dim=-1)-np.log(len(coefficient))+bg_score[start:stop,None]
                    return torch.cat(list(map(records_block,np.arange(0,len(local),record_chunk))),dim=0)
                def one_view(view):
                    p_cache,q_cache=caches[view]
                    forward=orientation(gf_p[view,first:last] if colored else gf_p[first:last],foreground_p[:,first:last,:,view],cross_forward[view] if colored else cross_forward,q_cache["metric"],q_cache["rhs"],q_cache["score"],view)
                    reverse=orientation(gf_q[view] if colored else gf_q,foreground_q[:,:,:,view],cross_reverse[view] if colored else cross_reverse,p_cache["metric"][first:last],p_cache["rhs"][:,first:last],p_cache["score"][:,first:last],view).transpose(1,2)
                    result[selected,first:last,view]=torch.logsumexp(forward-reverse+prior[:,None],dim=-1).cpu().numpy()
                    return forward
                joint=torch.stack(list(map(one_view,np.arange(views))),dim=-1).sum(-1)
                likelihood[selected,first:last]=torch.logsumexp(joint+prior[:,None],dim=-1).cpu().numpy()
                if return_pair:
                    pairs[selected,first:last]=joint.cpu().numpy()
                return None
            list(map(period_block,np.arange(0,len(period),period_chunk)))
            return None
        list(map(law_group,np.arange(len(unique_law))))
        return None
    with torch.no_grad():
        list(map(acquisition,np.arange(len(patterns))))
    output={"role_evidence":result,"foreground_log_likelihood":likelihood}
    if return_pair:
        output["pair_log_likelihood"]=pairs
    return output



def harmonic_cross_gram(time,mask,fg_period,bg_period=None,fg_harmonics=1,bg_harmonics=None,rcond=1e-10,period_chunk=16):
    """Exact harmonic self/cross Gram geometry on one observed-time mask.

    Columns alternate cosine/sine. Time starts at the first actual observation;
    intercept and linear trend use the same observed samples. A caller may pass
    a foreground-period chunk while keeping the full background-period axis.
    """
    time=np.asarray(time,dtype=float)
    actual=np.sort(time if mask is None else time[np.asarray(mask,dtype=bool)])
    actual=actual-actual[0]
    count=len(actual)
    foreground=np.atleast_1d(np.asarray(fg_period,dtype=float))
    background=foreground if bg_period is None else np.atleast_1d(np.asarray(bg_period,dtype=float))
    hf=int(fg_harmonics)
    hb=hf if bg_harmonics is None else int(bg_harmonics)
    wf=(2*np.pi/foreground[:,None])*np.arange(1,hf+1)[None]
    wb=(2*np.pi/background[:,None])*np.arange(1,hb+1)[None]
    centered=actual-actual.mean()
    linear_energy=centered@centered
    step=actual[-1]/(count-1)
    grid=step*np.arange(count)
    regular=np.max(abs(actual-grid))<=64*np.finfo(float).eps*max(abs(actual[-1]),abs(step))
    if not regular:
        trend=np.column_stack([np.ones(count)/np.sqrt(count),centered/np.sqrt(linear_energy)])
        trend_inverse=np.linalg.pinv(trend,rcond=rcond)
        angle=actual[None,:,None]*wf[:,None]
        fg=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(len(foreground),count,2*hf)
        fg-=np.einsum("ti,pij->ptj",trend,np.einsum("it,ptj->pij",trend_inverse,fg,optimize=True),optimize=True)
        bg_gram=np.empty((len(background),2*hb,2*hb))
        cross=np.empty((len(foreground),len(background),2*hf,2*hb))
        def background_chunk(start):
            stop=min(start+period_chunk,len(background))
            angle=actual[None,:,None]*wb[start:stop,None]
            bg=np.stack([np.cos(angle),np.sin(angle)],axis=-1).reshape(stop-start,count,2*hb)
            bg-=np.einsum("ti,qij->qtj",trend,np.einsum("it,qtj->qij",trend_inverse,bg,optimize=True),optimize=True)
            bg_gram[start:stop]=np.einsum("qti,qtj->qij",bg,bg,optimize=True)
            cross[:,start:stop]=np.einsum("pti,qtj->pqij",fg,bg,optimize=True)
            return None
        list(map(background_chunk,np.arange(0,len(background),period_chunk)))
        return {"fg_gram":np.einsum("pti,ptj->pij",fg,fg,optimize=True),"bg_gram":bg_gram,"cross":cross}
    def fourier_sum(frequency):
        x=frequency*step
        a=x/2.
        return np.exp(.5j*(count-1)*x)*(count*np.sinc(count*a/np.pi)/np.sinc(a/np.pi))
    def sinc_derivative(z):
        square=z*z
        series=z*(-1/3.+square*(1/30.+square*(-1/840.+square*(1/45360.+square*(-1/3991680.+square/518918400.)))))
        small=abs(z)<.1
        denominator=np.where(small,1.,square)
        ordinary=(z*np.cos(z)-np.sin(z))/denominator
        return np.where(small,series,ordinary)
    def trend_moments(frequency):
        x=frequency*step
        a=x/2.
        sinc=np.sinc(a/np.pi)
        long_sinc=np.sinc(count*a/np.pi)
        phase=np.exp(.5j*(count-1)*x)
        total=phase*(count*long_sinc/sinc)
        derivative=(count/2.)*(count*sinc_derivative(count*a)*sinc-long_sinc*sinc_derivative(a))/(sinc*sinc)
        first=-1j*step*phase*derivative+(.5*(count-1)*step-actual.mean())*total
        return (np.stack([total.real,total.imag],axis=-1).reshape(len(frequency),-1),
            np.stack([first.real,first.imag],axis=-1).reshape(len(frequency),-1))
    def raw_gram(left,right):
        difference=fourier_sum(left-right)
        summation=fourier_sum(left+right)
        result=np.empty(difference.shape[:-2]+(2*left.shape[-2],2*right.shape[-1]))
        result[...,0::2,0::2]=(difference.real+summation.real)/2.
        result[...,1::2,1::2]=(difference.real-summation.real)/2.
        result[...,0::2,1::2]=(summation.imag-difference.imag)/2.
        result[...,1::2,0::2]=(summation.imag+difference.imag)/2.
        return result
    sf,cf=trend_moments(wf)
    sb,cb=trend_moments(wb)
    fg_gram=raw_gram(wf[:,:,None],wf[:,None,:])-sf[:,:,None]*sf[:,None,:]/count-cf[:,:,None]*cf[:,None,:]/linear_energy
    bg_gram=raw_gram(wb[:,:,None],wb[:,None,:])-sb[:,:,None]*sb[:,None,:]/count-cb[:,:,None]*cb[:,None,:]/linear_energy
    cross=raw_gram(wf[:,None,:,None],wb[None,:,None,:])-sf[:,None,:,None]*sb[None,:,None,:]/count-cf[:,None,:,None]*cb[None,:,None,:]/linear_energy
    return {"fg_gram":fg_gram,"bg_gram":bg_gram,"cross":cross}


def fit_dual_source_variance_from_stats(fg_gram,bg_gram,cross,fg_rhs,bg_rhs,energy,degrees,
        template_coefficients,training_weight,reference_count,initial_covariance,
        bg_harmonic_power=None,foreground_prior="moment",maxiter=900,device="cpu"):
    """Fit FG/BG/noise variances at known continuous source periods.

    Grams and RHS use exactly the observed coordinates and the same detrending
    projection as the prediction kernel. No period grid is evaluated here.
    Each case supplies a phase bank and normalized allowed-reference weights.
    BG coefficient powers sum to one over harmonics, with cosine and sine
    receiving the same power. A shared (HB,) or case-specific (F,HB) power
    array is accepted. FG coefficients have unit template norm.
    The loss is weighted complete negative log likelihood per observed degree
    of freedom plus squared log-variance increments/(2*reference_count).
    """
    from scipy.optimize import minimize
    fg_gram=np.asarray(fg_gram,dtype=float);bg_gram=np.asarray(bg_gram,dtype=float)
    cross=np.asarray(cross,dtype=float);fg_rhs=np.asarray(fg_rhs,dtype=float)
    bg_rhs=np.asarray(bg_rhs,dtype=float);energy=np.asarray(energy,dtype=float)
    degrees=np.asarray(degrees,dtype=float)
    coefficient=np.asarray(template_coefficients,dtype=float)
    weight=np.asarray(training_weight,dtype=float);count=np.asarray(reference_count,dtype=float)
    initial=np.asarray(initial_covariance,dtype=float)
    records,df,views=fg_rhs.shape;db=bg_rhs.shape[1];hb=db//2
    cases,phases,_=coefficient.shape
    power=np.full((cases,hb),1./hb) if bg_harmonic_power is None else np.broadcast_to(np.asarray(bg_harmonic_power,dtype=float),(cases,hb)).copy()
    assert foreground_prior in ("gaussian","moment")
    assert fg_gram.shape==(records,df,df) and bg_gram.shape==(records,db,db)
    assert cross.shape==(records,df,db) and bg_rhs.shape==(records,db,views)
    assert energy.shape==(records,views) and degrees.shape==(records,)
    assert df%2==0 and db%2==0 and coefficient.shape==(cases,phases,df)
    assert weight.shape==(cases,records) and count.shape==(cases,) and initial.shape==(cases,views,3)
    assert np.all(weight>=0.) and np.allclose(weight.sum(1),1.) and np.all(count>0.) and np.all(initial>0.)
    assert power.shape==(cases,hb) and np.all(power>=0.) and np.allclose(power.sum(1),1.)
    coefficient=coefficient/np.linalg.norm(coefficient,axis=-1,keepdims=True)
    quadrature=coefficient.reshape(cases,phases,df//2,2)[:,:,:,[1,0]].copy()
    quadrature[:,:,:,0]*=-1.
    transform=np.stack([coefficient,quadrature.reshape(coefficient.shape)],axis=-1)
    dtype=torch.float64
    tensor=lambda value:torch.as_tensor(np.ascontiguousarray(value),dtype=dtype,device=device)
    eye=torch.eye(db,dtype=dtype,device=device)

    def case(index):
        allowed=np.flatnonzero(weight[index]>0.)
        scale=tensor(np.repeat(np.sqrt(power[index]),2))
        g=tensor(fg_gram[allowed])
        gb=tensor(bg_gram[allowed])*scale[None,:,None]*scale[None,None,:]
        c=tensor(cross[allowed])*scale[None,None,:]
        rf=tensor(fg_rhs[allowed]).permute(2,0,1)
        rb=(tensor(bg_rhs[allowed])*scale[None,:,None]).permute(2,0,1)
        e=tensor(energy[allowed]).T;dof=tensor(degrees[allowed]);w=tensor(weight[index,allowed])
        shape=tensor(transform[index]);start=tensor(initial[index])

        def objective(parameter,readout=False):
            increment=tensor(parameter).reshape(views,3).requires_grad_(not readout)
            variance=start*increment.exp()
            noise=variance[:,2]
            ratio=variance[:,1]/noise
            bg_precision=eye+ratio[:,None,None,None]*gb[None]
            bg_inverse=torch.linalg.inv(bg_precision)
            bg_metric=ratio[:,None,None,None]*bg_inverse
            bg_weighted_rhs=torch.einsum("vnij,vnj->vni",bg_metric,rb)
            conditional_gram=g[None]-torch.einsum("nia,vnab,njb->vnij",c,bg_metric,c)
            conditional_rhs=rf-torch.einsum("nia,vna->vni",c,bg_weighted_rhs)
            template_gram=torch.einsum("kia,vnij,kjb->vnkab",shape,conditional_gram,shape)
            amplitude=torch.einsum("kia,vni->vnka",shape,conditional_rhs)
            base=variance[:,0]/(2. if foreground_prior=="moment" else 1.)
            fg_ratio=base/noise
            a=1.+fg_ratio[:,None,None]*template_gram[:,:,:,0,0]
            b=fg_ratio[:,None,None]*template_gram[:,:,:,0,1]
            c10=fg_ratio[:,None,None]*template_gram[:,:,:,1,0]
            d=1.+fg_ratio[:,None,None]*template_gram[:,:,:,1,1]
            determinant=a*d-b*c10
            left=amplitude[:,:,:,0];right=amplitude[:,:,:,1]
            quadratic=(d*left.square()+a*right.square()-(b+c10)*left*right)/determinant
            gain=base[:,None,None]/noise[:,None,None].square()*quadratic
            phase_score=.5*(gain-torch.log(determinant))
            if foreground_prior=="moment":
                radial_covariance=.5*(a+d)/determinant
                radial_mean=(d*left-b*right).square()+(a*right-c10*left).square()
                radial_factor=radial_covariance+base[:,None,None]/(2.*noise[:,None,None].square()*determinant.square())*radial_mean
                phase_score=phase_score+torch.log(radial_factor)
            fg_score=torch.logsumexp(phase_score,dim=-1)-np.log(phases)
            bg_gain=(rb*bg_weighted_rhs).sum(-1)/noise[:,None]
            bg_score=.5*(bg_gain-torch.linalg.slogdet(bg_precision)[1])
            noise_score=-.5*(dof[None]*(np.log(2.*np.pi)+torch.log(noise[:,None]))+e/noise[:,None])
            log_likelihood=noise_score+bg_score+fg_score
            data_loss=-(log_likelihood/dof[None]*w[None]).sum()
            penalty=.5*increment.square().sum()/count[index]
            loss=data_loss+penalty
            if readout:
                return {"covariance":variance.detach().cpu().numpy(),
                    "log_increment":increment.detach().cpu().numpy(),
                    "objective":float(loss.detach().cpu()),
                    "data_loss":float(data_loss.detach().cpu()),
                    "penalty":float(penalty.detach().cpu())}
            gradient=torch.autograd.grad(loss,increment)[0]
            return float(loss.detach().cpu()),gradient.detach().cpu().numpy().ravel()

        initial_value=objective(np.zeros(views*3),True)
        fitted=minimize(objective,np.zeros(views*3),jac=True,method="L-BFGS-B",
            options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-6,"maxcor":25})
        final=objective(fitted.x,True)
        final.update({"initial_objective":initial_value["objective"],
            "initial_data_loss":initial_value["data_loss"],
            "success":bool(fitted.success),"iterations":int(fitted.nit),
            "gradient_max":float(np.max(abs(fitted.jac)))})
        return final

    fitted=list(map(case,np.arange(cases)))
    output={key:np.stack(list(map(lambda value:value[key],fitted))) for key in fitted[0]}
    output.update({"foreground_prior":np.asarray(foreground_prior),
        "bg_harmonic_power":power,"reference_count":count})
    return output


def source_pair_covariance(shared,private,peer_weight_square,observation_scale):
    """Push shared/private coefficient moments through the actual two-view map."""
    shared=np.asarray(shared,dtype=float);private=np.asarray(private,dtype=float)
    weight=np.asarray(peer_weight_square,dtype=float);scale=np.asarray(observation_scale,dtype=float)
    count=len(weight);dimension=shared.shape[-1]
    ss=np.broadcast_to(shared[:,None],(len(shared),count,dimension,dimension))
    pp=np.broadcast_to(private[:,None],ss.shape)
    covariance=np.concatenate([np.concatenate([ss+pp,ss],axis=-1),
        np.concatenate([ss,ss+weight[None,:,None,None]*pp],axis=-1)],axis=-2)
    diagonal=np.repeat(scale,dimension,axis=-1)
    return covariance*diagonal[None,:,:,None]*diagonal[None,:,None,:]


def source_noise_covariance(shared,private,peer_weight_square,observation_scale):
    """Shared noise remains shared; private peer variance uses sum of weights squared."""
    shared=np.asarray(shared,dtype=float);private=np.asarray(private,dtype=float)
    weight=np.asarray(peer_weight_square,dtype=float);scale=np.asarray(observation_scale,dtype=float)
    common=np.broadcast_to(shared[:,None],(len(shared),len(weight)))
    own=common+private[:,None]
    peer=common+private[:,None]*weight[None]
    covariance=np.stack([np.stack([own,common],axis=-1),np.stack([common,peer],axis=-1)],axis=-2)
    return covariance*scale[None,:,:,None]*scale[None,:,None,:]


def fit_shared_private_moments(own,cross,peer,peer_weight_square,weight):
    """Nonnegative moment fit of an exchangeable shared/private population law."""
    own=np.asarray(own,dtype=float);cross=np.asarray(cross,dtype=float)
    peer=np.asarray(peer,dtype=float);u=np.asarray(peer_weight_square,dtype=float)
    weight=np.asarray(weight,dtype=float);weight=weight/weight.sum()
    a=np.asarray([[4.,weight@(1.+u)],[weight@(1.+u),weight@(1.+u*u)]])
    b=np.asarray([weight@(own+2.*cross+peer),weight@(own+u*peer)])
    unconstrained=np.linalg.solve(a,b)
    candidates=np.vstack([unconstrained,[0.,max(b[1]/a[1,1],0.)],[max(b[0]/a[0,0],0.),0.],[0.,0.]])
    value=np.einsum("ki,ij,kj->k",candidates,a,candidates)-2.*(candidates@b)
    value[np.any(candidates<0.,axis=1)]=np.inf
    selected=candidates[np.argmin(value)]
    return {"shared":selected[0],"private":selected[1]}



def source_operator_covariance(shared,private,peer_weight_square,observation_scale,finite_loading=None):
    """Shared full two-view moments plus actual-weight private diagonal blocks."""
    shared=np.asarray(shared,dtype=float);private=np.asarray(private,dtype=float)
    weight=np.asarray(peer_weight_square,dtype=float);scale=np.asarray(observation_scale,dtype=float)
    dimension=shared.shape[-1]//2;states=len(shared);records=len(weight)
    covariance=np.broadcast_to(shared[:,None],(states,records,2*dimension,2*dimension)).copy()
    covariance[:,:,:dimension,:dimension]+=private[:,None,:dimension,:dimension]
    covariance[:,:,dimension:,dimension:]+=private[:,None,dimension:,dimension:]*weight[None,:,None,None]
    if finite_loading is not None:
        response=np.concatenate([np.ones((records,dimension)),-np.repeat(weight[:,None],dimension,axis=1)],axis=-1)
        covariance+=np.asarray(finite_loading)[:,None]*response[None,:,:,None]*response[None,:,None,:]
    diagonal=np.repeat(scale,dimension,axis=-1)
    return covariance*diagonal[None,:,:,None]*diagonal[None,:,None,:]


def source_population_moments(coefficients,weight,phase_isotropic=True,shared_source=False):
    """PSD shared/private reference moments; duplicate copies do not enlarge N."""
    coefficients=np.asarray(coefficients,dtype=float);weight=np.asarray(weight,dtype=float)
    weight=weight/weight.sum();mean=weight@coefficients
    second=np.einsum("n,ni,nj->ij",weight,coefficients,coefficients,optimize=True)
    mean_second=np.outer(mean,mean)
    if phase_isotropic:
        rotation=np.kron(np.eye(coefficients.shape[1]//2),np.asarray([[0.,-1.],[1.,0.]]))
        second=.5*(second+rotation@second@rotation.T)
        mean_second=.5*(mean_second+rotation@mean_second@rotation.T)
    effective=1.-np.sum(weight*weight)
    private=(second-mean_second)/effective
    shared=mean_second-private*np.sum(weight*weight)
    def psd(value):
        eigen,basis=np.linalg.eigh((value+value.T)/2.)
        return (basis*np.maximum(eigen,0.)[None])@basis.T
    if shared_source:
        return {"shared":psd(mean_second),"private":np.zeros_like(private),"finite_loading":psd(second-mean_second)}
    return {"shared":psd(shared),"private":psd(private),"finite_loading":np.zeros_like(private)}



def source_noise_population_moments(values,mask,weight):
    """Time-averaged PSD shared/private moments in the fixed two-view operator."""
    values=np.asarray(values,dtype=float);mask=np.asarray(mask,dtype=bool)
    weight=np.asarray(weight,dtype=float);weight=weight/weight.sum()
    observed=weight[:,None]*mask
    total=observed.sum(0)
    w=np.divide(observed,total[None],out=np.zeros_like(observed),where=total[None]>0.)
    mean=np.einsum("nt,ntv->tv",w,values,optimize=True)
    second=np.einsum("nt,ntv,ntw->tvw",w,values,values,optimize=True)
    mean_second=mean[:,:,None]*mean[:,None,:]
    square=(w*w).sum(0)
    private=np.divide(second-mean_second,1.-square[:,None,None],
        out=np.zeros_like(second),where=(1.-square[:,None,None])>np.finfo(float).eps)
    shared=mean_second-private*square[:,None,None]
    selected=total>0.
    def psd(value):
        eigen,basis=np.linalg.eigh((value+value.T)/2.)
        return (basis*np.maximum(eigen,0.)[None])@basis.T
    return {"shared":psd(shared[selected].mean(0)),"private":psd(private[selected].mean(0))}
