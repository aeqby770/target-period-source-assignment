"""Innovation-weighted peer residual geometry for periodic evidence."""
import numpy as np
from scipy.signal import hilbert


def observed_innovation_whiten(values,time,mask,tau):
    """Use the unified OU operator for scalar and harmonic-design arrays."""
    from source_assignment_unified import ou_whiten
    values=np.asarray(values,dtype=float)
    if values.ndim<=3:
        return ou_whiten(values,time,mask,tau)
    shape=values.shape
    flat=np.moveaxis(values,-2,1).reshape(shape[0],shape[-2],-1)
    white=ou_whiten(flat,time,mask,tau).reshape((shape[0],shape[-2])+shape[1:-2]+(shape[-1],))
    return np.moveaxis(white,1,-2)


def _whiten(values,rho,mask):
    shape=(len(rho),)+(1,)*(values.ndim-1)
    coefficient=rho.reshape(shape)
    first=values[...,:1,:]*np.sqrt(1.-coefficient**2)
    following=values[...,1:,:]-coefficient*values[...,:-1,:]
    result=np.concatenate([first,following],axis=-2)
    mask_shape=(len(rho),)+(1,)*(values.ndim-3)+(mask.shape[-1],1)
    return result*mask.reshape(mask_shape)


def correlated_peer_partial_power(time,count,mixed,peer,design,layout,rho):
    """Estimate cross-half physical harmonic energy using AR1 precision.

    Response, harmonic design and measured-peer nuisances use the same precision.
    Physical energy is evaluated after fitting, preserving the original units.
    """
    mask=np.arange(len(time))[None]<count[:,None]
    centered=(time[None]-(mask*time[None]).sum(axis=-1)[:,None]/count[:,None])*mask
    trend=np.stack([mask/np.sqrt(count[:,None]),
        centered/np.sqrt((centered**2).sum(axis=-1))[:,None]],axis=-1)
    weighted_trend=_whiten(trend,rho,mask)
    trend_inverse=np.linalg.inv(np.einsum("nti,ntj->nij",weighted_trend,weighted_trend,optimize=True))

    def remove_trend(values):
        weighted=_whiten(values,rho,mask)
        cross=np.einsum("nti,n...tj->n...ij",weighted_trend,weighted,optimize=True)
        coefficient=np.einsum("nij,n...jk->n...ik",trend_inverse,cross,optimize=True)
        physical=values-np.einsum("nti,n...ij->n...tj",trend,coefficient,optimize=True)
        return physical,_whiten(physical,rho,mask)

    physical_h,weighted_h=remove_trend(design[layout])
    nuisance=np.stack([peer,hilbert(peer,axis=-1).imag*mask],axis=-1)
    physical_n,weighted_n=remove_trend(nuisance)
    physical_y,weighted_y=remove_trend(mixed[:,:,None])
    inverse=np.linalg.inv(np.einsum("nti,ntj->nij",weighted_n,weighted_n,optimize=True))
    cross=np.einsum("npti,ntj->npij",weighted_h,weighted_n,optimize=True)
    coefficient=np.einsum("nij,npkj->npik",inverse,cross,optimize=True)
    partial_h=physical_h-np.einsum("nti,npij->nptj",physical_n,coefficient,optimize=True)
    rhs=np.einsum("nti,ntj->nij",weighted_n,weighted_y,optimize=True)
    nuisance_coefficient=np.einsum("nij,njk->nik",inverse,rhs,optimize=True)
    partial_y=physical_y-np.einsum("nti,nij->ntj",physical_n,nuisance_coefficient,optimize=True)
    white_h=_whiten(partial_h,rho,mask)
    white_y=_whiten(partial_y,rho,mask)[:,:,0]
    first=(np.arange(len(time))[None]<count[:,None]/2.)*mask
    halves=np.stack([first,mask & ~first])
    half_gram=np.einsum("hnt,npti,nptj->hnpij",halves,white_h,white_h,optimize=True)
    half_rhs=np.einsum("hnt,npti,nt->hnpi",halves,white_h,white_y,optimize=True)
    half_coefficient=np.linalg.solve(half_gram,half_rhs[...,None])[...,0]
    physical_gram=np.einsum("npti,nptj->npij",partial_h,partial_h,optimize=True)
    return np.einsum("npi,npij,npj->np",half_coefficient[0],physical_gram,
        half_coefficient[1],optimize=True)/count[:,None]

def harmonic_fundamental_support(time,count,mixed,peer,harmonic,rho):
    """Probability that a candidate fundamental is needed beyond its second harmonic.

    Both nested models share measured AR1 precision and the intercept/trend.
    The conditional two-coefficient block uses a unit-information g prior and
    integrates the innovation variance. Equal prior model odds are retained.
    """
    from scipy.special import expit
    acquisition,layout=np.unique(np.column_stack([count,rho]),axis=0,return_inverse=True)
    lengths=acquisition[:,0];layout_rho=acquisition[:,1]
    mask=np.arange(len(time))[None]<lengths[:,None]
    centered=(time[None]-(mask*time[None]).sum(axis=-1)[:,None]/lengths[:,None])*mask
    trend=np.stack([mask/np.sqrt(lengths[:,None]),
        centered/np.sqrt((centered**2).sum(axis=-1))[:,None]],axis=-1)
    white_z=_whiten(trend,layout_rho,mask)
    inverse=np.linalg.inv(np.einsum("lti,ltj->lij",white_z,white_z,optimize=True))
    white_h=_whiten(np.broadcast_to(harmonic,(len(lengths),*harmonic.shape)),layout_rho,mask)
    cross=np.einsum("lti,lptj->lpij",white_z,white_h,optimize=True)
    white_h-=np.einsum("lti,lpij->lptj",white_z,
        np.einsum("lij,lpjk->lpik",inverse,cross,optimize=True),optimize=True)
    own_mask=np.arange(len(time))[None]<count[:,None]
    white_y=_whiten(np.stack([mixed,peer],axis=-1),rho,own_mask)
    zy=np.einsum("nti,ntv->niv",white_z[layout],white_y,optimize=True)
    white_y-=np.einsum("nti,niv->ntv",white_z[layout],
        np.einsum("nij,njv->niv",inverse[layout],zy,optimize=True),optimize=True)
    gram=np.einsum("lpti,lptj->lpij",white_h,white_h,optimize=True)
    rhs=np.einsum("npti,ntv->npiv",white_h[layout],white_y,optimize=True)
    full=np.sum(rhs*np.linalg.solve(gram[layout],rhs),axis=2)
    half=np.sum(rhs[:,:,2:]*np.linalg.solve(gram[layout,:,2:,2:],rhs[:,:,2:]),axis=2)
    energy=np.sum(white_y**2,axis=1)
    partial=np.clip((full-half)/(energy[:,None]-half),0.,1.)
    degree=count-4.
    log_bf=-np.log1p(degree)[:,None,None]-.5*degree[:,None,None]*np.log1p(
        -degree[:,None,None]/(1.+degree[:,None,None])*partial)
    return expit(log_bf),log_bf


def harmonic_alias_log_weight(log_bf,period):
    """Compare conditional fundamental evidence with the half-period rival.

    Only rivals inside the fitted period grid enter this diagnostic.
    This relative weight tests harmonic assignment; it is not a calibrated
    probability that an otherwise unresolved biological clock is rhythmic.
    """
    half=period/2.
    position=(half-period[0])/(period[1]-period[0])
    low=np.clip(np.floor(position).astype(int),0,len(period)-1)
    high=np.minimum(low+1,len(period)-1)
    fraction=np.clip(position-low,0.,1.)
    rival=log_bf[:,low]*(1.-fraction)[None,:,None]+log_bf[:,high]*fraction[None,:,None]
    return np.where((half>=period[0])[None,:,None],-np.logaddexp(0.,rival-log_bf),0.)


def correlated_role_geometry(time,count,mixed,peer,period,rho):
    """H1 source geometry under stationary AR1 correlation and flat trends."""
    acquisition,layout=np.unique(np.column_stack([count,rho]),axis=0,return_inverse=True)
    lengths=acquisition[:,0];layout_rho=acquisition[:,1]
    mask=np.arange(len(time))[None]<lengths[:,None]
    centered=(time[None]-(mask*time[None]).sum(axis=-1)[:,None]/lengths[:,None])*mask
    trend=np.stack([mask/np.sqrt(lengths[:,None]),
        centered/np.sqrt((centered**2).sum(axis=-1))[:,None]],axis=-1)
    scale=np.sqrt(1.-layout_rho**2)
    white_z=_whiten(trend,layout_rho,mask)/scale[:,None,None]
    inverse=np.linalg.inv(np.einsum("lti,ltj->lij",white_z,white_z,optimize=True))
    angle=2*np.pi*time[None]/period[:,None]
    harmonic=np.stack([np.cos(angle),np.sin(angle)],axis=-1)
    white_h=_whiten(np.broadcast_to(harmonic,(len(lengths),*harmonic.shape)),
        layout_rho,mask)/scale[:,None,None,None]
    cross=np.einsum("lti,lptj->lpij",white_z,white_h,optimize=True)
    white_h-=np.einsum("lti,lpij->lptj",white_z,
        np.einsum("lij,lpjk->lpik",inverse,cross,optimize=True),optimize=True)
    own_mask=np.arange(len(time))[None]<count[:,None]
    white_y=_whiten(np.stack([mixed,peer],axis=-1),rho,own_mask)/np.sqrt(1.-rho**2)[:,None,None]
    zy=np.einsum("nti,ntv->niv",white_z[layout],white_y,optimize=True)
    white_y-=np.einsum("nti,niv->ntv",white_z[layout],
        np.einsum("nij,njv->niv",inverse[layout],zy,optimize=True),optimize=True)
    cross=np.einsum("lpti,lqtj->lpqij",white_h,white_h,optimize=True)
    grid=np.arange(len(period));gram=cross[:,grid,grid]
    joint=np.concatenate([np.concatenate([np.broadcast_to(gram[:,:,None],cross.shape),cross],axis=-1),
        np.concatenate([cross.swapaxes(-1,-2),np.broadcast_to(gram[:,None],cross.shape)],axis=-1)],axis=-2)
    rhs=np.einsum("npti,ntv->npiv",white_h[layout],white_y,optimize=True)
    return layout,joint,rhs


def conditional_foreground_presence_evidence(time,count,mixed,peer,period,rho,background,device="cpu",record_chunk=64):
    """Evidence for H1 foreground beyond an H2 background, with unknown noise scale.

    The intercept, trend and background coefficients are common nuisance
    parameters. A unit-information g prior integrates the additional
    foreground coefficients and the residual variance. Coincident source
    frequencies have zero added rank and contribute a Bayes factor of one.
    """
    import torch
    acquisition,layout=np.unique(np.column_stack([count,rho]),axis=0,return_inverse=True)
    lengths=acquisition[:,0];layout_rho=acquisition[:,1]
    mask=np.arange(len(time))[None]<lengths[:,None]
    center=(time[None]-(mask*time[None]).sum(-1)[:,None]/lengths[:,None])*mask
    trend=np.stack([mask/np.sqrt(lengths[:,None]),center/np.sqrt((center*center).sum(-1))[:,None]],axis=-1)
    scale=np.sqrt(1.-layout_rho**2)
    wz=_whiten(trend,layout_rho,mask)/scale[:,None,None]
    iz=np.linalg.inv(np.einsum("lti,ltj->lij",wz,wz))
    phase=2*np.pi*time[None]/period[:,None]
    harmonic=np.stack([np.cos(phase),np.sin(phase),np.cos(2*phase),np.sin(2*phase)],axis=-1)
    wh=_whiten(np.broadcast_to(harmonic,(len(lengths),*harmonic.shape)),layout_rho,mask)/scale[:,None,None,None]
    z_cross=np.einsum("lti,lptj->lpij",wz,wh,optimize=True)
    wh-=np.einsum("lti,lpij->lptj",wz,iz[:,None]@z_cross,optimize=True)
    own_mask=np.arange(len(time))[None]<count[:,None]
    wy=_whiten(np.stack([mixed,peer],axis=-1),rho,own_mask)/np.sqrt(1.-rho**2)[:,None,None]
    zy=np.einsum("nti,ntv->niv",wz[layout],wy,optimize=True)
    wy-=np.einsum("nti,niv->ntv",wz[layout],np.einsum("nij,njv->niv",iz[layout],zy,optimize=True),optimize=True)
    gram=np.einsum("lpti,lptj->lpij",wh,wh,optimize=True)
    bg_inverse=np.linalg.inv(gram)
    cross=np.einsum("lpti,lqtj->lpqij",wh[:,:,:,:2],wh,optimize=True)
    projection=np.einsum("lpqik,lqkj->lpqij",cross,bg_inverse,optimize=True)
    partial=gram[:,:,:2,:2][:,:,None]-projection@cross.swapaxes(-1,-2)
    partial=(partial+partial.swapaxes(-1,-2))/2.
    eigen,basis=np.linalg.eigh(partial)
    reference=np.linalg.eigvalsh(gram[:,:,:2,:2]).max(-1)
    tolerance=np.finfo(float).eps*len(time)*reference[:,:,None,None]
    keep=eigen>tolerance
    inverse_eigen=np.divide(1.,eigen,out=np.zeros_like(eigen),where=keep)
    partial_inverse=np.einsum("lpqik,lpqk,lpqjk->lpqij",basis,inverse_eigen,basis,optimize=True)
    rank=keep.sum(-1)
    rhs=np.einsum("npti,ntv->npiv",wh[layout],wy,optimize=True)
    energy=(wy*wy).sum(axis=1)
    dtype=torch.float64
    tensor=lambda a:torch.as_tensor(a,dtype=dtype,device=device)
    tp=tensor(projection);ti=tensor(partial_inverse);tr=tensor(rank);tb=tensor(bg_inverse)
    trhs=tensor(rhs);te=tensor(energy);tn=tensor(count-6.)
    tl=torch.as_tensor(layout,dtype=torch.long,device=device)
    single=lambda r,e,n,l:(
        p:=torch.index_select(tp,0,l.reshape(1))[0],inv:=torch.index_select(ti,0,l.reshape(1))[0],
        current_rank:=torch.index_select(tr,0,l.reshape(1))[0],bg_inv:=torch.index_select(tb,0,l.reshape(1))[0],
        rss:=torch.clamp(e[None]-torch.einsum("qiv,qij,qjv->qv",r,bg_inv,r),min=torch.finfo(dtype).tiny),
        partial_rhs:=r[:,:2,None,:].permute(0,2,1,3)-torch.einsum("pqij,qjv->pqiv",p,r),
        gain:=torch.einsum("pqiv,pqij,pqjv->pqv",partial_rhs,inv,partial_rhs),
        partial_r2:=torch.clamp(gain/rss[None],min=0.,max=1.),
        -.5*current_rank[:,:,None]*torch.log1p(n)-.5*n*torch.log1p(-n/(1.+n)*partial_r2))[-1]
    pair=torch.vmap(single,in_dims=(0,0,0,0),chunk_size=record_chunk)(trhs,te,tn,tl)
    log_bg=tensor(np.log(np.maximum(background,np.finfo(float).tiny)))
    reduce=lambda q:(
        separate:=torch.logsumexp(pair+q[:,None,:,None],dim=(1,2))-np.log(len(period)),
        shared:=torch.logsumexp(pair.sum(dim=-1)+q[:,None],dim=(1,2))-np.log(len(period)),
        torch.cat([separate,shared[:,None]],dim=-1))[-1]
    evidence=torch.vmap(reduce,in_dims=0,chunk_size=1)(log_bg)
    return evidence.cpu().numpy(),pair[0].cpu().numpy(),rank[layout[0]]


def correlated_role_assignment_evidence(joint_gram,rhs,layout,covariance,background,check_p,check_q,device="cpu",record_chunk=64,moment_prior=False,reduction="expected_bf"):
    """Integrate the unchanged opposite-role law using each record's noise layout.

    Record chunks limit memory only; no data are excluded or approximated.
    """
    import torch
    joint=torch.as_tensor(joint_gram,dtype=torch.float64,device=device)
    tensor_rhs=torch.as_tensor(rhs,dtype=torch.float64,device=device)
    tensor_layout=torch.as_tensor(layout,dtype=torch.long,device=device)
    log_background=torch.as_tensor(np.log(np.maximum(background,np.finfo(float).tiny)),
        dtype=torch.float64,device=device)

    reduction_kind=np.broadcast_to(np.asarray(reduction),(len(covariance),))
    use_presence=reduction_kind=="foreground_presence"
    use_marginal=reduction_kind=="marginal_likelihood"

    def single_case(h,log_rho,presence_mode,marginal_mode):
        if moment_prior:
            # Product-moment priors double each H1 coefficient variance.
            # Halving the Gaussian base keeps the original source power.
            h=torch.cat([h[:,:2]/2.,h[:,2:]],dim=-1)
        amplitude=torch.repeat_interleave(h[:,:2],2,dim=-1)
        ratio=(amplitude/h[:,2,None]).sqrt()
        precision=torch.eye(4,dtype=torch.float64,device=device)[None,None,None,None]+(
            joint[None]*ratio[:,None,None,None,:,None]*ratio[:,None,None,None,None,:])
        inverse=torch.linalg.inv(precision)
        determinant=torch.linalg.slogdet(precision)[1]
        if np.any(use_presence):
            bg_precision=torch.eye(2,dtype=torch.float64,device=device)[None,None,None]+(
                joint[:,0,:,2:,2:][None]*(h[:,1]/h[:,2])[:,None,None,None,None])
            bg_inverse=torch.linalg.inv(bg_precision)
            bg_determinant=torch.linalg.slogdet(bg_precision)[1]

        def single_record(r,current_layout,current_log_rho):
            selected=torch.index_select(inverse,1,current_layout.reshape(1))[:,0]
            logdet=torch.index_select(determinant,1,current_layout.reshape(1))[:,0]
            pp=torch.einsum("piv,vpqij,pjv->vpq",r,selected[:,:,:,:2,:2],r)
            qq=torch.einsum("qiv,vpqij,qjv->vpq",r,selected[:,:,:,2:,2:],r)
            pq=torch.einsum("piv,vpqij,qjv->vpq",r,selected[:,:,:,:2,2:],r)
            gain=(pp*h[:,0,None,None]+qq*h[:,1,None,None]
                +2*pq*(h[:,0]*h[:,1]).sqrt()[:,None,None])/h[:,2,None,None].square()
            pair=.5*(gain-logdet)
            if moment_prior:
                sd=amplitude.sqrt()
                post=selected*sd[:,None,None,:,None]*sd[:,None,None,None,:]
                rp=r.permute(2,0,1)
                mf=(torch.einsum("vpqij,vpj->vpqi",post[:,:,:,:2,:2],rp)
                    +torch.einsum("vpqij,vqj->vpqi",post[:,:,:,:2,2:],rp))/h[:,2,None,None,None]
                mb=(torch.einsum("vpqij,vpj->vpqi",post[:,:,:,2:,:2],rp)
                    +torch.einsum("vpqij,vqj->vpqi",post[:,:,:,2:,2:],rp))/h[:,2,None,None,None]
                cross=post[:,:,:,:2,2:]
                ef=mf.square().sum(-1)+torch.diagonal(post[:,:,:,:2,:2],dim1=-2,dim2=-1).sum(-1)
                eb=mb.square().sum(-1)+torch.diagonal(post[:,:,:,2:,2:],dim1=-2,dim2=-1).sum(-1)
                moment=ef*eb+2*cross.square().sum((-1,-2))
                moment+=4*torch.einsum("vpqi,vpqij,vpqj->vpq",mf,cross,mb)
                pair+=torch.log(torch.clamp(moment,min=torch.finfo(torch.float64).tiny)/(4*h[:,0]*h[:,1])[:,None,None])
            log_bf=pair-pair.transpose(-1,-2)
            marginal=torch.logsumexp(log_bf+current_log_rho[None,None],dim=-1)
            if np.any(use_marginal):
                likelihood_reduction=(torch.logsumexp(pair+current_log_rho[None,None],dim=-1)
                    -torch.logsumexp(pair.transpose(-1,-2)+current_log_rho[None,None],dim=-1))
                marginal=torch.where(marginal_mode,likelihood_reduction,marginal)
            if np.any(use_presence):
                selected_bg=torch.index_select(bg_inverse,1,current_layout.reshape(1))[:,0]
                selected_bg_det=torch.index_select(bg_determinant,1,current_layout.reshape(1))[:,0]
                bg_quadratic=torch.einsum("qiv,vqij,qjv->vq",r,selected_bg,r)
                bg_gain=bg_quadratic*h[:,1,None]/h[:,2,None].square()
                bg_pair=.5*(bg_gain-selected_bg_det)
                presence_reduction=(torch.logsumexp(pair+current_log_rho[None,None],dim=-1)
                    -torch.logsumexp(bg_pair+current_log_rho[None],dim=-1)[:,None])
                marginal=torch.where(presence_mode,presence_reduction,marginal)
            diagonal=torch.diagonal(log_bf,dim1=-2,dim2=-1)
            diagnostic=torch.stack([log_bf[0,check_p,check_q],diagonal.min(),diagonal.max()])
            return marginal.T.contiguous(),diagnostic

        features,diagnostic=torch.vmap(single_record,in_dims=(0,0,0),chunk_size=record_chunk)(
            tensor_rhs,tensor_layout,log_rho)
        return features,torch.stack([diagnostic[0,0],diagnostic[:,1].min(),diagnostic[:,2].max()])

    features,diagnostic=torch.vmap(single_case,in_dims=(0,0,0,0),chunk_size=1)(
        torch.as_tensor(covariance,dtype=torch.float64,device=device),log_background,
        torch.as_tensor(use_presence,device=device),torch.as_tensor(use_marginal,device=device))
    return features.cpu().numpy(),diagnostic.cpu().numpy()


def correlated_clock_order(amplitude,membership,run_index,count,selected,time=None,mask=None):
    """Evaluate one/two-clock residuals with the shared measured AR1 precision."""
    from scipy.special import expit
    residual_one=amplitude["one_clock_residual"];residual_two=amplitude["two_clock_residual"]
    if time is not None:
        from scipy.optimize import minimize_scalar
        from source_assignment_unified import _observed_steps
        time=np.asarray(time,dtype=float)
        mask=np.arange(len(time))[None]<count[:,None] if mask is None else np.asarray(mask,dtype=bool)
        previous,paired,gap=_observed_steps(time,mask)
        old=np.take_along_axis(residual_two,np.maximum(previous,0),axis=1)
        tau_run=np.empty(len(membership));rho_run=np.empty(len(membership));step_run=np.empty(len(membership))
        signed_run=np.empty(len(membership))
        def cohort(group):
            take=np.flatnonzero(run_index==group);keep=paired[take]
            x=old[take][keep];y=residual_two[take][keep];dt=gap[take][keep]
            step=float(np.median(dt));signed=float(np.sum(x*y)/np.sum(x*x))
            edge=1.-np.sqrt(np.finfo(float).eps)
            if np.allclose(dt,step,rtol=1e-10,atol=step*1e-10):
                rho=float(np.clip(signed,0.,edge))
            else:
                ratio=dt/step
                objective=lambda value:float(np.sum((y-np.power(value,ratio)*x)**2))
                fit=minimize_scalar(objective,bounds=(0.,edge),method="bounded",options={"xatol":1e-12})
                points=np.asarray([0.,fit.x,edge]);cost=np.asarray([objective(0.),fit.fun,objective(edge)])
                rho=float(points[cost.argmin()])
            rho_run[group]=rho;step_run[group]=step;signed_run[group]=signed
            tau_run[group]=0. if rho==0. else -step/np.log(rho)
            return None
        list(map(cohort,np.arange(len(membership))))
        rho=rho_run[run_index];tau=tau_run[run_index]
        first=observed_innovation_whiten(residual_one,time,mask,tau)
        second=observed_innovation_whiten(residual_two,time,mask,tau)
        rss_one=np.sum(first**2,axis=1)*(1.-rho*rho)
        rss_two=np.sum(second**2,axis=1)*(1.-rho*rho)
        effective=np.asarray(count)-2.
        bic_one=effective*np.log(rss_one/effective)+5.*np.log(effective)
        bic_two=effective*np.log(rss_two/effective)+10.*np.log(effective)
        probability=expit(.5*(bic_one-bic_two))
        corrected=amplitude|{"two_clock_probability":np.where(selected,probability,amplitude["two_clock_probability"]),
            "frequency_order_probability":np.where(selected,probability,amplitude.get("frequency_order_probability",amplitude["two_clock_probability"])),
            "bic_one":np.where(selected,bic_one,amplitude["bic_one"]),"bic_two":np.where(selected,bic_two,amplitude["bic_two"])}
        return corrected,{"rho":rho,"tau":tau,"reference_step":step_run[run_index],
            "signed_lag_moment":signed_run[run_index],"negative_lag_moment":signed_run[run_index]<0.,
            "colored_probability":probability}

    valid_pair=np.arange(residual_two.shape[-1]-1)[None]<count[:,None]-1
    numerator=(residual_two[:,1:]*residual_two[:,:-1]*valid_pair).sum(axis=-1)
    denominator=(residual_two[:,:-1]**2*valid_pair).sum(axis=-1)
    rho_run=(membership@numerator)/(membership@denominator)
    rho=np.clip(rho_run[run_index],-1.+np.sqrt(np.finfo(float).eps),1.-np.sqrt(np.finfo(float).eps))
    rss_one=(1.-rho**2)*residual_one[:,0]**2+(((residual_one[:,1:]-rho[:,None]*residual_one[:,:-1])**2)*valid_pair).sum(axis=-1)
    rss_two=(1.-rho**2)*residual_two[:,0]**2+(((residual_two[:,1:]-rho[:,None]*residual_two[:,:-1])**2)*valid_pair).sum(axis=-1)
    bic_one=(count-2.)*np.log(rss_one/(count-2.))+5.*np.log(count-2.)
    bic_two=(count-2.)*np.log(rss_two/(count-2.))+10.*np.log(count-2.)
    probability=expit(.5*(bic_one-bic_two))
    if "clock_family_probability" in amplitude:
        probability=np.where(selected,amplitude["clock_family_probability"],probability)
        bic_one=np.where(selected,-2.*amplitude["clock_family_evidence_one"],bic_one)
        bic_two=np.where(selected,-2.*amplitude["clock_family_evidence_two"],bic_two)
    elif "drift_order_probability" in amplitude:
        probability=np.where(selected,amplitude["drift_order_probability"],probability)
        bic_one=np.where(selected,amplitude["drift_bic_one"],bic_one)
        bic_two=np.where(selected,amplitude["drift_bic_two"],bic_two)
    corrected=amplitude|{"two_clock_probability":np.where(selected,probability,amplitude["two_clock_probability"]),
        "frequency_order_probability":np.where(selected,probability,amplitude.get("frequency_order_probability",amplitude["two_clock_probability"])),
        "bic_one":np.where(selected,bic_one,amplitude["bic_one"]),
        "bic_two":np.where(selected,bic_two,amplitude["bic_two"])}
    return corrected,{"rho":rho,"colored_probability":probability}


def calibrated_joint_source_evidence(joint_gram,rhs,layout,foreground_covariance,background_covariance,
        noise_covariance,observation_energy,observation_degrees,background,state_weight,
        device="cpu",record_chunk=64,period_chunk=8,return_pair=False,shared_clock_probability=.5,
        state_chunk=8,covariance_group=None):
    """Complete state/BG-period marginal density with shared-clock point mass.

    Covariance congruences may first be removed from Y/RHS/energy. Within each
    exact covariance/layout group, the small Gaussian factors are calculated
    once and reused for every record. Chunk sizes change memory only.
    """
    from scipy.special import logsumexp
    from functools import reduce
    import torch
    assert 0.<=shared_clock_probability<=1.
    cf=np.asarray(foreground_covariance,dtype=float);cb=np.asarray(background_covariance,dtype=float)
    ce=np.asarray(noise_covariance,dtype=float)
    cf=cf[:,None] if cf.ndim==3 else cf;cb=cb[:,None] if cb.ndim==3 else cb;ce=ce[:,None] if ce.ndim==3 else ce
    raw_geometry=np.asarray(joint_gram,dtype=float)
    exact_geometry,geometry_owner=np.unique(raw_geometry.reshape(len(raw_geometry),-1),axis=0,return_inverse=True)
    joint_gram=exact_geometry.reshape((-1,)+raw_geometry.shape[1:])
    layout=geometry_owner[np.asarray(layout,dtype=int)]
    records,periods=rhs.shape[:2];states=len(cf)
    cf=np.broadcast_to(cf,(states,records,4,4));cb=np.broadcast_to(cb,(states,records,4,4))
    ce=np.broadcast_to(ce,(states,records,2,2))
    state_weight=np.asarray(state_weight,dtype=float)
    assert state_weight.shape==(states,) and np.all(state_weight>0.) and np.isclose(state_weight.sum(),1.)
    if covariance_group is None:
        signature=np.column_stack([layout,cf.transpose(1,0,2,3).reshape(records,-1),
            cb.transpose(1,0,2,3).reshape(records,-1),ce.transpose(1,0,2,3).reshape(records,-1)])
    else:
        signature=np.column_stack([layout,np.asarray(covariance_group)])
    _,group=np.unique(signature,axis=0,return_inverse=True)
    log_bg=np.log(np.maximum(background,np.finfo(float).tiny));log_weight=np.log(state_weight)
    dtype=torch.float64
    tensor=lambda value:torch.as_tensor(np.array(value,dtype=float,order="C",copy=True),dtype=dtype,device=device)
    tr=tensor(rhs);energy=tensor(observation_energy);degree=tensor(observation_degrees);gram=tensor(joint_gram)
    eye8=torch.eye(8,dtype=dtype,device=device);eye4=torch.eye(4,dtype=dtype,device=device)
    order=torch.as_tensor([0,1,4,5,2,3,6,7],dtype=torch.long,device=device)
    output_joint=np.empty((records,periods));output_peer=np.empty_like(output_joint)
    output_independent=np.empty_like(output_joint);output_shared=np.empty_like(output_joint)
    output_pair=np.empty((records,periods,periods)) if return_pair else None

    def root(covariance):
        value,vector=torch.linalg.eigh((covariance+covariance.transpose(-1,-2))/2.)
        return (vector*value.clamp_min(0.).sqrt()[...,None,:])@vector.transpose(-1,-2)

    def geometry_group(number):
        selected=np.flatnonzero(group==number);first=int(selected[0]);current_layout=int(layout[first])
        source_f=tensor(cf[:,first]);source_b=tensor(cb[:,first]);noise=tensor(ce[:,first])
        inverse_noise=torch.linalg.inv(noise);noise_peer=noise[:,1,1]
        zero=torch.zeros_like(source_f)
        prior=torch.cat([torch.cat([source_f,zero],dim=-1),torch.cat([zero,source_b],dim=-1)],dim=-2)
        square=root(prior)
        peer_f=source_f[:,2:,2:];peer_b=source_b[:,2:,2:];pz=torch.zeros_like(peer_f)
        peer_prior=torch.cat([torch.cat([peer_f,pz],dim=-1),torch.cat([pz,peer_b],dim=-1)],dim=-2)
        peer_square=root(peer_prior)
        logdet_noise=torch.linalg.slogdet(noise)[1]
        current_rhs=tr[selected];current_energy=energy[selected];current_degree=degree[selected]
        def period_block(start):
            stop=min(start+period_chunk,periods);g=gram[current_layout,start:stop]
            shape=(len(selected),stop-start,periods)
            initial=(torch.full(shape,-torch.inf,dtype=dtype,device=device),
                torch.full(shape,-torch.inf,dtype=dtype,device=device))
            def state_block(merged,begin):
                end=min(begin+state_chunk,states);ei=inverse_noise[begin:end];sq=square[begin:end]
                weighted=torch.einsum("kvw,pqij->kpqviwj",ei,g).reshape(end-begin,stop-start,periods,8,8)
                weighted=weighted[:,:,:,order][:,:,:,:,order]
                precision=eye8+sq[:,None,None].transpose(-1,-2)@weighted@sq[:,None,None]
                kernel=sq[:,None,None]@torch.linalg.solve(precision,sq[:,None,None].transpose(-1,-2))
                logdet=torch.linalg.slogdet(precision)[1]
                ps=peer_square[begin:end];nv=noise_peer[begin:end]
                peer_precision=eye4+ps[:,None,None].transpose(-1,-2)@g[None]@ps[:,None,None]/nv[:,None,None,None,None]
                peer_kernel=ps[:,None,None]@torch.linalg.solve(peer_precision,ps[:,None,None].transpose(-1,-2))/nv[:,None,None,None,None].square()
                peer_logdet=torch.linalg.slogdet(peer_precision)[1]
                def record_block(first_record):
                    last=min(first_record+record_chunk,len(selected))
                    r=current_rhs[first_record:last];e=current_energy[first_record:last];d=current_degree[first_record:last]
                    wr=torch.einsum("npiv,kvw->knpiw",r,ei).permute(0,1,2,4,3).reshape(end-begin,last-first_record,periods,4)
                    rp=wr[:,:,start:stop]
                    gain=torch.einsum("knpi,kpqij,knpj->knpq",rp,kernel[:,:,:,:4,:4],rp)
                    gain+=torch.einsum("knqi,kpqij,knqj->knpq",wr,kernel[:,:,:,4:,4:],wr)
                    gain+=2.*torch.einsum("knpi,kpqij,knqj->knpq",rp,kernel[:,:,:,:4,4:],wr)
                    normal=-.5*(d[None]*(2.*np.log(2.*np.pi)+logdet_noise[begin:end,None])+
                        torch.einsum("nij,kji->kn",e,ei))
                    joint=.5*(gain-logdet[:,None])+normal[:,:,None,None]
                    rpeer=r[:,:,:,1];ppeer=rpeer[:,start:stop]
                    peer_gain=torch.einsum("npi,kpqij,npj->knpq",ppeer,peer_kernel[:,:,:,:2,:2],ppeer)
                    peer_gain+=torch.einsum("nqi,kpqij,nqj->knpq",rpeer,peer_kernel[:,:,:,2:,2:],rpeer)
                    peer_gain+=2.*torch.einsum("npi,kpqij,nqj->knpq",ppeer,peer_kernel[:,:,:,:2,2:],rpeer)
                    peer_normal=-.5*(d[None]*(np.log(2.*np.pi)+torch.log(nv)[:,None])+e[None,:,1,1]/nv[:,None])
                    peer=.5*(peer_gain-peer_logdet[:,None])+peer_normal[:,:,None,None]
                    weight=tensor(log_weight[begin:end])[:,None,None,None]
                    return torch.logsumexp(joint+weight,dim=0),torch.logsumexp(peer+weight,dim=0)
                chunks=list(map(record_block,np.arange(0,len(selected),record_chunk)))
                joint=torch.cat(list(map(lambda value:value[0],chunks)),dim=0)
                peer=torch.cat(list(map(lambda value:value[1],chunks)),dim=0)
                return torch.logaddexp(merged[0],joint),torch.logaddexp(merged[1],peer)
            merged_joint,merged_peer=reduce(state_block,np.arange(0,states,state_chunk),initial)
            bg=tensor(log_bg[selected])[:,None]
            independent_joint=torch.logsumexp(merged_joint+bg,dim=-1)
            independent_peer=torch.logsumexp(merged_peer+bg,dim=-1)
            local=torch.arange(stop-start,device=device);same=torch.arange(start,stop,device=device)
            shared_joint=merged_joint[:,local,same];shared_peer=merged_peer[:,local,same]
            if shared_clock_probability==0.:
                joint_marginal=independent_joint;peer_marginal=independent_peer
            elif shared_clock_probability==1.:
                joint_marginal=shared_joint;peer_marginal=shared_peer
            else:
                joint_marginal=torch.logaddexp(np.log(shared_clock_probability)+shared_joint,np.log1p(-shared_clock_probability)+independent_joint)
                peer_marginal=torch.logaddexp(np.log(shared_clock_probability)+shared_peer,np.log1p(-shared_clock_probability)+independent_peer)
            output_joint[selected,start:stop]=joint_marginal.cpu().numpy()
            output_peer[selected,start:stop]=peer_marginal.cpu().numpy()
            output_independent[selected,start:stop]=independent_joint.cpu().numpy()
            output_shared[selected,start:stop]=shared_joint.cpu().numpy()
            if return_pair:
                output_pair[selected,start:stop]=merged_joint.cpu().numpy()
            return None
        list(map(period_block,np.arange(0,periods,period_chunk)))
        return None
    with torch.no_grad():
        list(map(geometry_group,np.unique(group)))
    features=np.stack([output_joint-output_peer,output_peer],axis=-1)
    features-=logsumexp(features,axis=1,keepdims=True)
    result={"features":features,"joint_log_likelihood":output_joint,"peer_log_likelihood":output_peer,
        "independent_clock_log_likelihood":output_independent,"shared_clock_log_likelihood":output_shared}
    if return_pair:
        result["pair_log_likelihood"]=output_pair
    return result
