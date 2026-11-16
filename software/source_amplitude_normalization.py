"""Unlabelled two-clock decomposition and equal-energy representation of No."""
import numpy as np
from scipy.optimize import elementwise,least_squares
from scipy.special import expit, logsumexp

TWO_CLOCK_MAX_NFEV = 2000
AMPLITUDE_SOLVER = f"independent_record_trf_exact_nfev{TWO_CLOCK_MAX_NFEV}"


def frequency_average_representation(design, joint, gram, rhs, layout, count, mixed,
        noise_variance, periodic_variance, period):
    """Integrate Gaussian source coefficients over the fixed frequency-pair grid."""
    lengths = np.unique(count)
    row = np.arange(len(count))
    energy = (mixed**2).sum(axis=-1)
    # Full discrete-period Gaussian evidence; same variance budget as solve_wave.
    # Motivation: Badiu et al., https://arxiv.org/abs/1604.03744 (frequency uncertainty).
    marginal_p, marginal_q = np.triu_indices(len(period), 1)
    marginal_joint = joint[:, marginal_p, marginal_q]
    marginal_eigen, marginal_basis = np.linalg.eigh(marginal_joint)
    marginal_eigen = np.maximum(marginal_eigen, 0.)
    marginal_ridge = 4.*noise_variance/periodic_variance
    marginal_rhs = np.concatenate([rhs[:, marginal_p], rhs[:, marginal_q]], axis=-1)
    marginal_rotated = np.einsum("nbi,lbij->lnbj", marginal_rhs, marginal_basis, optimize=True)[layout, row]
    marginal_denominator = marginal_eigen[layout] + marginal_ridge[:, None, None]
    marginal_inverse = 1./marginal_denominator
    marginal_beta = np.einsum("lbij,nbj->lnbi", marginal_basis,
        marginal_rotated*marginal_inverse, optimize=True)[layout, row]
    marginal_loglike = .5*((marginal_rotated**2*marginal_inverse).sum(axis=-1)/noise_variance[:, None]
        - np.log1p(marginal_eigen[layout]/marginal_ridge[:, None, None]).sum(axis=-1))
    marginal_logweight = marginal_loglike-logsumexp(marginal_loglike, axis=-1)[:, None]
    marginal_weight = np.exp(marginal_logweight)
    marginal_mode = marginal_loglike.argmax(axis=-1)
    check_pair = marginal_joint[layout[0], marginal_mode[0]]
    check_rhs = marginal_rhs[0, marginal_mode[0]]
    check_inverse = np.linalg.inv(check_pair+marginal_ridge[0]*np.eye(check_pair.shape[-1]))
    check_evidence = .5*(check_rhs@check_inverse@check_rhs/noise_variance[0]
        - np.linalg.slogdet(np.eye(check_pair.shape[-1])+check_pair/marginal_ridge[0])[1])
    assert np.isclose(check_evidence,marginal_loglike[0,marginal_mode[0]],rtol=1e-9,atol=1e-8)
    marginal_gram = np.stack([gram[:, marginal_p], gram[:, marginal_q]],axis=2)
    marginal_vectors = marginal_basis.reshape(len(design),len(marginal_p),2,4,8)
    marginal_spectral_energy = np.einsum("lbsik,lbsij,lbsjk->lbsk",
        marginal_vectors,marginal_gram,marginal_vectors,optimize=True)
    marginal_trace = np.einsum("lbsk,nbk->lnbs",marginal_spectral_energy,
        marginal_inverse,optimize=True)[layout,row]
    marginal_beta_source = marginal_beta.reshape(len(count),len(marginal_p),2,4)
    marginal_point_energy = np.einsum("nbsi,lbsij,nbsj->lnbs",marginal_beta_source,
        marginal_gram,marginal_beta_source,optimize=True)[layout,row]
    # PSD contractions can have round-off-sized negative values.
    marginal_point_rms2 = np.maximum(marginal_point_energy/count[:,None,None],0.)
    marginal_posterior_rms = np.sqrt(marginal_point_rms2+
        np.maximum(noise_variance[:,None,None]*marginal_trace/count[:,None,None],0.))
    marginal_common = np.sqrt(marginal_point_rms2.mean(axis=-1))
    marginal_delta_source = (marginal_common[:,:,None,None]/marginal_posterior_rms[:,:,:,None]-1.)*marginal_beta_source
    marginal_delta = marginal_delta_source.reshape(marginal_beta.shape)
    marginal_delta_energy = np.einsum("nbi,lbij,nbj->lnb",marginal_delta,marginal_joint,
        marginal_delta,optimize=True)[layout,row]
    marginal_cross_energy = np.einsum("nbi,nbi->nb",marginal_delta,marginal_rhs,optimize=True)
    marginal_rms = np.sqrt((energy[:,None]+2.*marginal_cross_energy+marginal_delta_energy)/count[:,None])
    assert np.isfinite(marginal_rms).all() and (marginal_rms>0.).all()
    marginal_scaled_weight = marginal_weight/marginal_rms
    marginal_coefficient = np.zeros((len(count),len(period),4))
    np.add.at(marginal_coefficient,(row[:,None],marginal_p[None]),
        marginal_scaled_weight[:,:,None]*marginal_delta_source[:,:,0])
    np.add.at(marginal_coefficient,(row[:,None],marginal_q[None]),
        marginal_scaled_weight[:,:,None]*marginal_delta_source[:,:,1])
    marginal_average_signal = marginal_scaled_weight.sum(axis=-1)[:,None]*mixed + np.einsum(
        "lpti,npi->lnt",design,marginal_coefficient,optimize=True)[layout,row]
    marginal_mode_design = np.concatenate([design[layout,marginal_p[marginal_mode]],
        design[layout,marginal_q[marginal_mode]]],axis=-1)
    marginal_mode_signal = (mixed+np.einsum("nti,ni->nt",marginal_mode_design,
        marginal_delta[row,marginal_mode],optimize=True))/marginal_rms[row,marginal_mode,None]
    assert np.max(abs((marginal_mode_signal**2).sum(axis=-1)/count-1.))<1e-8
    marginal_one_eigen, marginal_one_basis = np.linalg.eigh(gram)
    marginal_one_eigen = np.maximum(marginal_one_eigen,0.)
    marginal_one_rotated = np.einsum("npi,lpij->lnpj",rhs,marginal_one_basis,optimize=True)[layout,row]
    marginal_one_ridge = 2.*noise_variance/periodic_variance
    marginal_one_denominator = marginal_one_eigen[layout]+marginal_one_ridge[:,None,None]
    marginal_one_loglike = .5*((marginal_one_rotated**2/marginal_one_denominator).sum(axis=-1)/noise_variance[:,None]
        -np.log1p(marginal_one_eigen[layout]/marginal_one_ridge[:,None,None]).sum(axis=-1))
    marginal_log_odds = logsumexp(marginal_loglike,axis=-1)-np.log(len(marginal_p))
    marginal_log_odds -= logsumexp(marginal_one_loglike,axis=-1)-np.log(len(period))
    marginal_order_probability = expit(marginal_log_odds)
    marginal_effective_pairs = np.exp(-(marginal_weight*marginal_logweight).sum(axis=-1))
    marginal_period_mass = np.zeros((len(count),len(period)))
    np.add.at(marginal_period_mass,(row[:,None],marginal_p[None]),.5*marginal_weight)
    np.add.at(marginal_period_mass,(row[:,None],marginal_q[None]),.5*marginal_weight)
    assert np.max(abs(marginal_period_mass.sum(axis=-1)-1.))<1e-10

    return {"signal": marginal_average_signal, "order_probability": marginal_order_probability,
        "effective_pairs": marginal_effective_pairs, "period_mass": marginal_period_mass,
        "map_signal": marginal_mode_signal}


def correlated_frequency_average_representation(time,count,mixed,period,rho,noise_variance,periodic_variance,device="cpu",record_chunk=32,power_quadrature=1,harmonic_count=2,mask=None,noise_tau=None):
    """Average frequency pairs using stationary OU/AR1 precision and H1 or H2 designs.

    Coefficients and frequency evidence use correlated noise. Source energy
    and the final signal use the original detrended observation coordinates,
    so a change in precision does not change the RMS definition. The default
    single quadrature node retains equal source power. Additional nodes
    integrate a uniform power fraction with the same total variance budget.
    """
    import torch
    from source_correlated_geometry import _whiten,observed_innovation_whiten
    actual_mode=mask is not None or noise_tau is not None
    count=np.asarray(count,dtype=int)
    own_mask=np.arange(len(time))[None]<count[:,None] if mask is None else np.asarray(mask,dtype=bool)
    count=own_mask.sum(axis=1)
    physical_mask,physical_layout=np.unique(own_mask,axis=0,return_inverse=True)
    lengths=physical_mask.sum(axis=1)
    if actual_mode:
        sampling_scale=float(np.median(np.diff(time)))
        if noise_tau is None:
            rho=np.asarray(rho,dtype=float)
            assert np.all((rho>=0.)&(rho<1.)), "OU correlation at the reference step must lie in [0,1)."
            noise_tau=np.divide(-sampling_scale,np.log(np.maximum(rho,np.finfo(float).tiny)))
            noise_tau=np.where(rho==0.,0.,noise_tau)
        time=(np.asarray(time,dtype=float)-time[0])/sampling_scale
        period=np.asarray(period,dtype=float)/sampling_scale
        noise_tau=np.asarray(noise_tau,dtype=float)/sampling_scale
    physical_center=(time[None]-(physical_mask*time[None]).sum(-1)[:,None]/lengths[:,None])*physical_mask
    physical_z=np.stack([physical_mask/np.sqrt(lengths[:,None]),
        physical_center/np.sqrt((physical_center**2).sum(-1))[:,None]],axis=-1)
    component_dimension=2*harmonic_count
    pair_dimension=2*component_dimension
    phase=2*np.pi*time[None,:,None]*np.arange(1,harmonic_count+1)[None,None]/period[:,None,None]
    harmonic=np.stack([np.cos(phase),np.sin(phase)],axis=-1).reshape(len(period),len(time),component_dimension)
    physical_design=harmonic[None]*physical_mask[:,None,:,None]-np.einsum("lti,lpij->lptj",physical_z,
        np.einsum("lti,ptj->lpij",physical_z,harmonic,optimize=True),optimize=True)
    physical_cross=np.einsum("lpti,lqtj->lpqij",physical_design,physical_design,optimize=True)
    grid=np.arange(len(period));physical_gram=physical_cross[:,grid,grid]
    pair_p,pair_q=np.triu_indices(len(period),1)
    nodes,quadrature_weight=np.polynomial.legendre.leggauss(power_quadrature)
    power_fraction=np.repeat((nodes+1.)/2.,len(pair_p))
    prior_probability=np.repeat(quadrature_weight/2.,len(pair_p))
    p=np.tile(pair_p,power_quadrature);q=np.tile(pair_q,power_quadrature)
    coefficient_scale=np.repeat(np.stack([np.sqrt(2.*power_fraction),np.sqrt(2.*(1.-power_fraction))],axis=-1),component_dimension,axis=-1)
    physical_joint=np.concatenate([np.concatenate([physical_gram[:,p],physical_cross[:,p,q]],axis=-1),
        np.concatenate([physical_cross[:,q,p],physical_gram[:,q]],axis=-1)],axis=-2)
    physical_components=np.stack([physical_gram[:,p],physical_gram[:,q]],axis=2)
    physical_rhs=np.einsum("pti,nt->npi",harmonic,mixed,optimize=True)
    noise_coordinate=noise_tau if actual_mode else np.asarray(rho,dtype=float)
    acquisition,layout=np.unique(np.column_stack([physical_layout,noise_coordinate]),axis=0,return_inverse=True)
    acquisition_physical=acquisition[:,0].astype(int)
    n=lengths[acquisition_physical];r=acquisition[:,1]
    mask=physical_mask[acquisition_physical]
    center=(time[None]-(mask*time[None]).sum(-1)[:,None]/n[:,None])*mask
    z=np.stack([mask/np.sqrt(n[:,None]),center/np.sqrt((center*center).sum(-1))[:,None]],axis=-1)
    if actual_mode:
        wz=observed_innovation_whiten(z,time,mask,r)
        wh=observed_innovation_whiten(np.broadcast_to(harmonic,(len(n),*harmonic.shape)),time,mask,r)
        wy=observed_innovation_whiten(mixed,time,own_mask,noise_tau)
    else:
        innovation=np.sqrt(1-r*r)
        wz=_whiten(z,r,mask)/innovation[:,None,None]
        wh=_whiten(np.broadcast_to(harmonic,(len(n),*harmonic.shape)),r,mask)/innovation[:,None,None,None]
        wy=(_whiten(mixed[:,:,None],rho,own_mask)/np.sqrt(1.-rho*rho)[:,None,None])[:,:,0]
    iz=np.linalg.inv(np.einsum("lti,ltj->lij",wz,wz,optimize=True))
    zcross=np.einsum("lti,lptj->lpij",wz,wh,optimize=True)
    wh-=np.einsum("lti,lpij->lptj",wz,iz[:,None]@zcross,optimize=True)
    cross=np.einsum("lpti,lqtj->lpqij",wh,wh,optimize=True)
    gram=cross[:,grid,grid]
    joint=np.concatenate([np.concatenate([gram[:,p],cross[:,p,q]],axis=-1),
        np.concatenate([cross[:,q,p],gram[:,q]],axis=-1)],axis=-2)
    joint*=coefficient_scale[None,:,:,None]*coefficient_scale[None,:,None,:]
    eigen,basis=np.linalg.eigh(joint);eigen=np.maximum(eigen,0.)
    basis*=coefficient_scale[None,:,:,None]
    one_eigen,one_basis=np.linalg.eigh(gram);one_eigen=np.maximum(one_eigen,0.)
    source_basis=basis.reshape(len(n),len(p),2,component_dimension,pair_dimension)
    spectral_energy=np.einsum("lbsik,lbsij,lbsjk->lbsk",source_basis,
        physical_components[acquisition_physical],source_basis,optimize=True)
    dtype=torch.float64
    tensor=lambda a:torch.as_tensor(a,dtype=dtype,device=device)
    twh=tensor(wh);teigen=tensor(eigen);tbasis=tensor(basis)
    tone_eigen=tensor(one_eigen);tone_basis=tensor(one_basis);tenergy=tensor(spectral_energy)
    tphysical=tensor(physical_design);tjoint=tensor(physical_joint);tcomponents=tensor(physical_components)
    tp=torch.as_tensor(p,dtype=torch.long,device=device);tq=torch.as_tensor(q,dtype=torch.long,device=device)
    tlog_prior=tensor(np.log(prior_probability));t_power_fraction=tensor(power_fraction)

    def single_record(y,yw,physical_r,n,sigma2,periodic2,current_layout,current_physical):
        h=torch.index_select(twh,0,current_layout.reshape(1))[0]
        ev=torch.index_select(teigen,0,current_layout.reshape(1))[0]
        vec=torch.index_select(tbasis,0,current_layout.reshape(1))[0]
        source_energy=torch.index_select(tenergy,0,current_layout.reshape(1))[0]
        pg=torch.index_select(tcomponents,0,current_physical.reshape(1))[0]
        pj=torch.index_select(tjoint,0,current_physical.reshape(1))[0]
        pd=torch.index_select(tphysical,0,current_physical.reshape(1))[0]
        rhs=torch.einsum("pti,t->pi",h,yw)
        pair_rhs=torch.cat([rhs[tp],rhs[tq]],dim=-1)
        rotated=torch.einsum("bi,bij->bj",pair_rhs,vec)
        ridge=component_dimension*sigma2/periodic2
        denominator=ev+ridge
        beta=torch.einsum("bij,bj->bi",vec,rotated/denominator)
        loglike=.5*((rotated.square()/denominator).sum(-1)/sigma2-torch.log1p(ev/ridge).sum(-1))
        weighted_loglike=loglike+tlog_prior
        log_weight=weighted_loglike-torch.logsumexp(weighted_loglike,dim=-1);weight=log_weight.exp()
        source_beta=beta.reshape(len(p),2,component_dimension)
        point_energy=torch.einsum("bsi,bsij,bsj->bs",source_beta,pg,source_beta).clamp_min(0.)
        covariance_energy=sigma2*torch.einsum("bsk,bk->bs",source_energy,1./denominator).clamp_min(0.)
        posterior_rms=((point_energy+covariance_energy)/n).sqrt()
        common=(point_energy.mean(-1)/n).sqrt()
        delta=((common[:,None]/posterior_rms-1.)[:,:,None]*source_beta).reshape(len(p),pair_dimension)
        physical_pair_rhs=torch.cat([physical_r[tp],physical_r[tq]],dim=-1)
        adjusted_energy=y.square().sum()+2.*(delta*physical_pair_rhs).sum(-1)+torch.einsum("bi,bij,bj->b",delta,pj,delta)
        rms=(adjusted_energy/n).clamp_min(torch.finfo(dtype).tiny).sqrt()
        scaled_weight=weight/rms
        delta_source=delta.reshape(len(p),2,component_dimension)
        coefficient=torch.zeros((len(period),component_dimension),dtype=dtype,device=device)
        coefficient=coefficient.index_add(0,tp,scaled_weight[:,None]*delta_source[:,0])
        coefficient=coefficient.index_add(0,tq,scaled_weight[:,None]*delta_source[:,1])
        signal=scaled_weight.sum()*y+torch.einsum("pti,pi->t",pd,coefficient)
        mode=loglike.argmax()
        mode_p=torch.index_select(tp,0,mode.reshape(1))[0]
        mode_q=torch.index_select(tq,0,mode.reshape(1))[0]
        mode_delta=torch.index_select(delta_source,0,mode.reshape(1))[0]
        mode_design=torch.stack([torch.index_select(pd,0,mode_p.reshape(1))[0],
            torch.index_select(pd,0,mode_q.reshape(1))[0]],dim=0)
        map_signal=(y+torch.einsum("sti,si->t",mode_design,mode_delta))/torch.index_select(rms,0,mode.reshape(1))[0]
        ev1=torch.index_select(tone_eigen,0,current_layout.reshape(1))[0]
        vec1=torch.index_select(tone_basis,0,current_layout.reshape(1))[0]
        rot1=torch.einsum("pi,pij->pj",rhs,vec1);ridge1=.5*component_dimension*sigma2/periodic2
        one_loglike=.5*((rot1.square()/(ev1+ridge1)).sum(-1)/sigma2-torch.log1p(ev1/ridge1).sum(-1))
        evidence_two=torch.logsumexp(weighted_loglike,dim=-1)-np.log(len(pair_p))
        evidence_one=torch.logsumexp(one_loglike,dim=-1)-np.log(len(period))
        odds=evidence_two-evidence_one
        mass=torch.zeros(len(period),dtype=dtype,device=device).index_add(0,tp,.5*weight).index_add(0,tq,.5*weight)
        frequency_weight=weight.reshape(power_quadrature,len(pair_p)).sum(dim=0)
        effective=(-(frequency_weight*torch.log(frequency_weight.clamp_min(torch.finfo(dtype).tiny))).sum()).exp()
        fraction_mean=(weight*t_power_fraction).sum()
        return signal,torch.sigmoid(odds),effective,mass,map_signal,fraction_mean,evidence_one,evidence_two

    result=torch.vmap(single_record,in_dims=(0,0,0,0,0,0,0,0),chunk_size=record_chunk)(
        tensor(mixed),tensor(wy),tensor(physical_rhs),tensor(count),tensor(noise_variance),tensor(periodic_variance),
        torch.as_tensor(layout,dtype=torch.long,device=device),torch.as_tensor(physical_layout,dtype=torch.long,device=device))
    signal,probability,effective,mass,map_signal,fraction_mean,evidence_one,evidence_two=result
    return {"signal":signal.cpu().numpy(),"order_probability":probability.cpu().numpy(),
        "effective_pairs":effective.cpu().numpy(),"period_mass":mass.cpu().numpy(),"map_signal":map_signal.cpu().numpy(),
        "power_fraction_mean":fraction_mean.cpu().numpy(),"log_evidence_one":evidence_one.cpu().numpy(),"log_evidence_two":evidence_two.cpu().numpy()}


def balance_source_amplitudes(time, count, mixed, period, frequency_marginal=None, return_residuals=False, mask=None, frequency_reduction="mean"):
    """Average continuous one/two-clock H2 representations using their BIC weights.

    An explicit mask uses observed timestamp geometry. Optimization coordinates
    are measured in median sampling intervals, making all stopping tolerances
    dimensionless while preserving the original unit-spaced calculation.
    """
    actual_mask=mask is not None
    record_mask=np.arange(len(time))[None]<np.asarray(count)[:,None] if mask is None else np.asarray(mask,dtype=bool)
    count=record_mask.sum(axis=1)
    patterns,owner=np.unique(record_mask,axis=0,return_inverse=True)
    if actual_mask and len(patterns)>1:
        # Fits are independent; grouping exact masks avoids an L x N grid tensor.
        def acquisition(layout):
            take=np.flatnonzero(owner==layout)
            frequency=None if frequency_marginal is None else np.asarray(frequency_marginal)[take]
            fitted=balance_source_amplitudes(time,count[take],np.asarray(mixed)[take],period,frequency,return_residuals,record_mask[take],frequency_reduction=frequency_reduction)
            return take,fitted
        grouped=list(map(acquisition,np.arange(len(patterns))))
        order=np.concatenate(list(map(lambda item:item[0],grouped))).argsort()
        first=grouped[0][1]
        def combine(key):
            parts=list(map(lambda item:item[1][key],grouped))
            if key=="optimizer_iterations":
                return np.max(np.stack(parts),axis=0)
            if key=="jacobian_check":
                return max(parts)
            if np.ndim(first[key])==0:
                return first[key]
            return np.concatenate(parts,axis=0)[order]
        return dict(zip(first,map(combine,first)))
    time=np.asarray(time,dtype=float);period=np.asarray(period,dtype=float)
    sampling_scale=float(np.median(np.diff(time))) if actual_mask else 1.
    time=(time-time[0])/sampling_scale if actual_mask else time
    period=period/sampling_scale
    mixed=np.where(record_mask,np.asarray(mixed,dtype=float),0.)
    mask,layout=np.unique(record_mask,axis=0,return_inverse=True)
    lengths=mask.sum(axis=1)
    centered = (time[None] - (mask*time[None]).sum(axis=-1)[:, None]/lengths[:, None])*mask
    trend = np.stack([mask/np.sqrt(lengths[:, None]),
        centered/np.sqrt((centered**2).sum(axis=-1))[:, None]], axis=-1)
    angle = 2*np.pi*time[None, :, None]*np.arange(1, 3)[None, None]/period[:, None, None]
    harmonic = np.stack([np.cos(angle), np.sin(angle)], axis=-1).reshape(len(period), len(time), 4)
    design = harmonic[None]*mask[:, None, :, None] - np.einsum("lti,lpij->lptj", trend,
        np.einsum("lti,ptj->lpij", trend, harmonic, optimize=True), optimize=True)
    cross = np.einsum("lpti,lqtj->lpqij", design, design, optimize=True)
    grid = np.arange(len(period))
    gram = cross[:, grid, grid]
    joint = np.concatenate([
        np.concatenate([np.broadcast_to(gram[:, :, None], cross.shape), cross], axis=-1),
        np.concatenate([cross.swapaxes(-1, -2), np.broadcast_to(gram[:, None], cross.shape)], axis=-1)], axis=-2)
    inverse = np.linalg.pinv(joint, hermitian=True)
    rank = np.linalg.matrix_rank(joint, hermitian=True)
    rhs = np.einsum("pti,nt->npi", harmonic, mixed, optimize=True)
    gain = np.einsum("npi,lpqij,npj->lnpq", rhs, inverse[:, :, :, :4, :4], rhs, optimize=False)
    gain += np.einsum("nqi,lpqij,nqj->lnpq", rhs, inverse[:, :, :, 4:, 4:], rhs, optimize=False)
    gain += 2.*np.einsum("npi,lpqij,nqj->lnpq", rhs, inverse[:, :, :, :4, 4:], rhs, optimize=False)
    row = np.arange(len(count))
    gain = gain[layout, row]
    energy = (mixed**2).sum(axis=-1)
    single_gain = gain[:, grid, grid]
    peak = (single_gain >= np.pad(single_gain, ((0, 0), (1, 1)), constant_values=-np.inf)[:, :-2])
    peak &= single_gain >= np.pad(single_gain, ((0, 0), (1, 1)), constant_values=-np.inf)[:, 2:]
    candidate_row, candidate_grid = np.nonzero(peak)
    two_grid = np.argmax(np.where(np.triu(np.ones((len(period), len(period)), dtype=bool), 1),
        gain, -np.inf).reshape(len(count), -1), axis=-1)
    p, q = np.unravel_index(two_grid, (len(period), len(period)))
    step = float(period[1]-period[0])
    coarse_rss = np.maximum(energy-gain[row, p, q], np.finfo(float).eps*energy)
    noise_variance = coarse_rss/(count-2.)
    periodic_variance = (energy-coarse_rss)/(count-2.)

    def solve_wave(periods, records):
        n_source = periods.shape[-1]
        angle = 2*np.pi*time[None, :, None, None]*np.arange(1, 3)[None, None, None]/periods[:, None, :, None]
        h = np.stack([np.cos(angle), np.sin(angle)], axis=-1).reshape(len(records), len(time), 4*n_source)
        h *= record_mask[records,:,None]
        t = trend[layout[records]]
        h -= np.einsum("nti,nij->ntj", t, np.einsum("nti,ntj->nij", t, h, optimize=True), optimize=True)
        g = np.einsum("nti,ntj->nij", h, h, optimize=True)
        r = np.einsum("nti,nt->ni", h, mixed[records], optimize=True)
        # Equal variance budgets for one/two clocks; the Gaussian coefficient prior
        # makes the profile smooth at coincident and harmonically overlapping periods.
        ridge = noise_variance[records]*(2.*n_source)/periodic_variance[records]
        g_inverse = np.linalg.inv(g+ridge[:, None, None]*np.eye(4*n_source))
        beta = np.einsum("nij,nj->ni", g_inverse, r, optimize=True)
        augmented = np.concatenate([h, np.sqrt(ridge)[:, None, None]*np.eye(4*n_source)], axis=1)
        residual = np.pad(mixed[records], ((0, 0), (0, 4*n_source))) - np.einsum("nti,ni->nt", augmented, beta, optimize=True)
        derivative = np.stack([np.sin(angle), -np.cos(angle)], axis=-1)
        derivative *= (2*np.pi*time[None, :, None, None]*np.arange(1, 3)[None, None, None]/periods[:, None, :, None]**2)[..., None]
        derivative = derivative.reshape(h.shape)*record_mask[records,:,None]
        derivative -= np.einsum("nti,nij->ntj", t, np.einsum("nti,ntj->nij", t, derivative, optimize=True), optimize=True)
        derivative = np.pad(derivative.reshape(len(records), len(time), n_source, 4),
            ((0, 0), (0, 4*n_source), (0, 0), (0, 0)))
        return augmented, beta, residual, derivative, g_inverse

    def single_loss(x, records):
        return (solve_wave(x[:, None], records)[2]**2).sum(axis=-1)

    lower = np.maximum(period.min(), period[candidate_grid]-step)
    upper = np.minimum(period.max(), period[candidate_grid]+step)
    bracket = elementwise.bracket_minimum(single_loss, (lower+upper)/2., xl0=lower, xr0=upper,
        xmin=lower, xmax=upper, args=(candidate_row,), maxiter=100)
    assert np.all(np.isin(bracket.status, [0, -1])), np.unique(bracket.status, return_counts=True)
    bracket_x = np.stack(bracket.bracket)
    bracket_f = np.stack(bracket.f_bracket)
    one_x = np.take_along_axis(bracket_x, bracket_f.argmin(axis=0)[None], axis=0)[0]
    active = bracket.success
    one = elementwise.find_minimum(single_loss, tuple(bracket_x[:, active]), args=(candidate_row[active],),
        tolerances={"xatol":1e-8, "xrtol":0., "fatol":0., "frtol":0.}, maxiter=200)
    assert np.all(one.success), np.unique(one.status, return_counts=True)
    one_x[active] = one.x
    # Refine the same regularized profile at its analytic stationary root.
    # Objective-value minimization alone stops above the BIC precision floor.
    def single_gradient(x, records):
        _, beta, residual, derivative, _ = solve_wave(x[:, None], records)
        direct=np.einsum("ntsi,nsi->nts",derivative,beta.reshape(len(records),1,4),optimize=True)[:,:,0]
        return -2.*np.sum(residual*direct,axis=1)
    slope_left=single_gradient(bracket_x[0],candidate_row)
    slope_right=single_gradient(bracket_x[2],candidate_row)
    stationary=active&(slope_left<0.)&(slope_right>0.)
    def refine_single(index):
        from scipy.optimize import brentq
        records=candidate_row[index:index+1]
        derivative=lambda x:float(single_gradient(np.asarray([x]),records)[0])
        return brentq(derivative,bracket_x[0,index],bracket_x[2,index],
            xtol=4.*np.finfo(float).eps,rtol=4.*np.finfo(float).eps)
    one_x[stationary]=np.asarray(list(map(refine_single,np.flatnonzero(stationary))))
    h_one, beta_one, residual_one, _, _ = solve_wave(one_x[:, None], candidate_row)
    cost_one = (residual_one**2).sum(axis=-1)
    rss_one = (residual_one[:, :len(time)]**2).sum(axis=-1)
    best_cost = np.full(len(count), np.inf)
    np.minimum.at(best_cost, candidate_row, cost_one)
    best_index = np.full(len(count), len(candidate_row), dtype=int)
    np.minimum.at(best_index, candidate_row, np.where(cost_one == best_cost[candidate_row], np.arange(len(candidate_row)), len(candidate_row)))
    one_period = one_x[best_index]
    best_rss = rss_one[best_index]

    def two_residual(flat, records):
        return solve_wave(flat.reshape(len(records), 2), records)[2].ravel()

    def two_jacobian(flat, records):
        h, beta, residual, derivative, g_inverse = solve_wave(flat.reshape(len(records), 2), records)
        direct = np.einsum("ntsi,nsi->nts", derivative, beta.reshape(len(records), 2, 4), optimize=True)
        cross_rhs = np.einsum("ntsi,nt->nsi", derivative, residual, optimize=True)
        cross_rhs = (cross_rhs[:, :, :, None]*np.eye(2)[None, :, None, :]).reshape(len(records), 8, 2)
        cross_rhs -= np.einsum("nti,nts->nis", h, direct, optimize=True)
        d_beta = g_inverse@cross_rhs
        jacobian = -direct-np.einsum("nti,nis->nts", h, d_beta, optimize=True)
        return jacobian

    initial_two = np.stack([period[p], period[q]], axis=-1)
    lower_two = np.maximum(period.min(), initial_two-step)
    upper_two = np.minimum(period.max(), initial_two+step)
    midpoint = initial_two.mean(axis=-1)
    upper_two[:, 0] = np.minimum(upper_two[:, 0], midpoint)
    lower_two[:, 1] = np.maximum(lower_two[:, 1], midpoint)
    check_x = initial_two[0].copy()
    probe = np.asarray([1e-5, 0.])
    numerical = (two_residual(check_x+probe, np.asarray([0]))-two_residual(check_x-probe, np.asarray([0])))/(2e-5)
    jacobian_check = np.max(abs(numerical-two_jacobian(check_x, np.asarray([0]))[0, :, 0]))
    assert jacobian_check < 1e-6, jacobian_check
    # Each recording has its own two-period objective and stopping decision.
    # A dense two-column TRF solve keeps unrelated recordings out of its trust region.
    def fit_two_record(record):
        records = np.asarray([record])
        result = least_squares(two_residual, initial_two[record],
            jac=lambda x, r: two_jacobian(x, r)[0], args=(records,),
            bounds=(lower_two[record], upper_two[record]), x_scale="jac", tr_solver="exact",
            ftol=1e-10, xtol=1e-10, gtol=1e-8, max_nfev=TWO_CLOCK_MAX_NFEV)
        assert result.success, (record, result.status, result.message, result.nfev, result.optimality)
        return result

    two = tuple(map(fit_two_record, row))
    two_period = np.stack(tuple(map(lambda result: result.x, two)))
    two_nfev = np.fromiter(map(lambda result: result.nfev, two), dtype=int)
    selected_design, beta, residual_two, _, posterior_inverse = solve_wave(two_period, row)
    selected_design = selected_design[:, :len(time)]
    rss_two = (residual_two[:, :len(time)]**2).sum(axis=-1)
    # Four amplitudes and one period per source; common trend/noise parameters cancel.
    bic_one = (count-2.)*np.log(best_rss/(count-2.)) + 5.*np.log(count-2.)
    bic_two = (count-2.)*np.log(rss_two/(count-2.)) + 10.*np.log(count-2.)
    use_two = bic_two < bic_one
    two_probability = expit(.5*(bic_one-bic_two))
    component = np.einsum("nti,ni->nt", selected_design[:, :, :4], beta[:, :4], optimize=True)
    other = np.einsum("nti,ni->nt", selected_design[:, :, 4:], beta[:, 4:], optimize=True)
    rms = np.sqrt(np.stack([(component**2).sum(axis=-1), (other**2).sum(axis=-1)], axis=-1)/count[:, None])
    common = np.sqrt((rms**2).mean(axis=-1))
    first = np.divide(component, rms[:, 0, None], out=np.zeros_like(component), where=rms[:, 0, None] > 0.)
    second = np.divide(other, rms[:, 1, None], out=np.zeros_like(other), where=rms[:, 1, None] > 0.)
    # Average models on the predictor's unit-RMS scale before constructing the peer.
    two_signal = mixed+common[:, None]*(first+second)-component-other
    two_signal /= np.sqrt((two_signal**2).sum(axis=-1)/count)[:, None]
    balanced = mixed+two_probability[:, None]*(two_signal-mixed)
    source_gram = np.stack([
        np.einsum("nti,ntj->nij", selected_design[:, :, :4], selected_design[:, :, :4], optimize=True),
        np.einsum("nti,ntj->nij", selected_design[:, :, 4:], selected_design[:, :, 4:], optimize=True)], axis=1)
    source_covariance = noise_variance[:, None, None, None]*np.stack([
        posterior_inverse[:, :4, :4], posterior_inverse[:, 4:, 4:]], axis=1)
    posterior_rms = np.sqrt(rms**2 + np.einsum("nsij,nsji->ns", source_gram,
        source_covariance, optimize=True)/count[:, None])
    posterior_two_signal = mixed+common[:, None]*(
        component/posterior_rms[:, 0, None]+other/posterior_rms[:, 1, None])-component-other
    posterior_two_signal /= np.sqrt((posterior_two_signal**2).sum(axis=-1)/count)[:, None]
    single_prediction = mixed-residual_one[best_index, :len(time)]
    single_rms = .5*np.sqrt((single_prediction**2).sum(axis=-1)/count)
    rms = np.where(use_two[:, None], rms, single_rms[:, None])
    frequency_result = {}
    if frequency_marginal is not None and np.any(frequency_marginal):
        selected = np.flatnonzero(frequency_marginal)
        active_layout, selected_layout = np.unique(layout[selected], return_inverse=True)
        marginal = frequency_average_representation(design[active_layout], joint[active_layout],
            gram[active_layout], rhs[selected], selected_layout, count[selected], mixed[selected],
            noise_variance[selected], periodic_variance[selected], period)
        frequency_signal = posterior_two_signal.copy()
        frequency_probability = two_probability.copy()
        frequency_signal[selected] = marginal["map_signal" if frequency_reduction=="map" else "signal"]
        frequency_probability[selected] = marginal["order_probability"]
        frequency_result = {"frequency_mean_signal": frequency_signal,
            "frequency_order_probability": frequency_probability}
    residual_result = {}
    if return_residuals:
        residual_result = {"one_clock_residual": residual_one[best_index, :len(time)],
            "two_clock_residual": residual_two[:, :len(time)], "one_clock_period": one_period*sampling_scale,
            "two_clock_periods": two_period*sampling_scale}
    return {**frequency_result, **residual_result, "signal": balanced, "periods": np.where(use_two[:, None], two_period, one_period[:, None])*sampling_scale,
        "component_rms": rms, "model_order": 1+use_two.astype(int), "bic_one": bic_one, "bic_two": bic_two,
        "two_clock_probability": two_probability, "two_clock_signal": two_signal,
        "posterior_energy_signal": posterior_two_signal, "posterior_component_rms": posterior_rms,
        "optimizer_iterations": np.asarray([np.max(one.nit,initial=0), two_nfev.max()]), "two_optimizer_nfev": two_nfev,
        "amplitude_solver": np.asarray(AMPLITUDE_SOLVER), "jacobian_check": jacobian_check,
        "noise_variance": noise_variance, "periodic_variance": periodic_variance}


def source_amplitude_representation(time,mask,signal,run_index,period,*,common_baseline=True,
        common_baseline_strength=.05,common_baseline_period=None,share_order=False,
        posterior_energy=True,frequency_marginal=True,frequency_correlated=True,
        colored_order=True,cohort_center=False,device="cpu",residual_noise_model="OU",frequency_reduction="mean"):
    """Original No source representation on actual timestamps and observations.

    Flags are explicit fitted-method choices, never inferred from record length.
    All time arguments share a physical unit. Baseline period is a physical
    curvature scale; its default is the geometric center of the supplied bounds.
    Source energies remain measured in the original detrended RMS geometry.
    Explicit frozen-parameter controls may use residual_noise_model="white"
    or frequency_reduction="map"; defaults preserve the original OU mean law.
    """
    from source_assignment_unified import shared_background,detrend
    from source_correlated_geometry import correlated_clock_order
    assert residual_noise_model in {"OU","white"} and frequency_reduction in {"mean","map"}
    time=np.asarray(time,dtype=float);mask=np.asarray(mask,dtype=bool)
    signal=np.where(mask,np.asarray(signal,dtype=float),0.)
    run_index=np.asarray(run_index,dtype=int);period=np.asarray(period,dtype=float)
    count=mask.sum(1);groups=int(run_index.max())+1
    member=(np.arange(groups)[:,None]==run_index[None]).astype(float)
    data={"time":time,"signal":signal,"mask":mask,"count":count,"run_index":run_index,
        "membership":member,"average":member/member.sum(1)[:,None]}
    flags={key:np.broadcast_to(np.asarray(value,dtype=bool),(len(signal),)) for key,value in
        {"common_baseline":common_baseline,"share_order":share_order,"posterior_energy":posterior_energy,
        "frequency_marginal":frequency_marginal,"frequency_correlated":frequency_correlated,
        "colored_order":colored_order,"cohort_center":cohort_center}.items()}
    baseline_period=float(np.sqrt(period.min()*period.max())) if common_baseline_period is None else float(common_baseline_period)
    original,adjusted=shared_background(data,baseline_period,strength=float(common_baseline_strength))
    reference=np.where(flags["common_baseline"][:,None],adjusted,original)*mask
    rms=np.sqrt(np.sum(reference**2,axis=1)/count)
    source=reference/rms[:,None]
    available=member@mask.astype(float)
    cohort_mean=(member@reference)/np.maximum(available,1.)
    contrast=detrend((reference-cohort_mean[run_index])*mask,time,mask)
    contrast_rms=np.sqrt(np.sum(contrast**2,axis=1)/count)
    contrast=np.divide(contrast,contrast_rms[:,None],out=np.zeros_like(contrast),where=contrast_rms[:,None]>0.)
    source=np.where(flags["cohort_center"][:,None],contrast,source)
    amplitude=balance_source_amplitudes(time,count,source,period,
        frequency_marginal=flags["frequency_marginal"]&~flags["frequency_correlated"],return_residuals=True,mask=mask,frequency_reduction=frequency_reduction)
    if residual_noise_model=="white":
        amplitude=amplitude|{"frequency_order_probability":amplitude.get("frequency_order_probability",amplitude["two_clock_probability"])}
        noise={"rho":np.zeros(len(signal)),"tau":np.zeros(len(signal)),"noise_model":np.asarray("independent_white"),
            "colored_probability":amplitude["two_clock_probability"]}
    else:
        amplitude,noise=correlated_clock_order(amplitude,member,run_index,count,flags["colored_order"],time=time,mask=mask)
    selected=np.flatnonzero(flags["frequency_correlated"])
    frequency_signal=amplitude.get("frequency_mean_signal",amplitude["posterior_energy_signal"].copy())
    frequency_diagnostic={}
    if len(selected):
        frequency_diagnostic=correlated_frequency_average_representation(time,count[selected],source[selected],period,
            noise["rho"][selected],amplitude["noise_variance"][selected],amplitude["periodic_variance"][selected],
            device=device,mask=mask[selected],noise_tau=noise["tau"][selected])
        frequency_signal[selected]=frequency_diagnostic["map_signal" if frequency_reduction=="map" else "signal"]
    energy_signal=np.where(flags["frequency_marginal"][:,None],frequency_signal,amplitude["posterior_energy_signal"])
    individual=np.where(flags["frequency_marginal"],amplitude["frequency_order_probability"],amplitude["two_clock_probability"])
    shared=expit(.5*(member@(amplitude["bic_one"]-amplitude["bic_two"])))[run_index]
    probability=np.where(flags["share_order"],shared,individual)
    two_signal=np.where(flags["posterior_energy"][:,None],energy_signal,amplitude["two_clock_signal"])
    representation=(source+probability[:,None]*(two_signal-source))*mask
    representation/=np.sqrt(np.sum(representation**2,axis=1)/count)[:,None]
    return {"signal":representation,"reference_signal":reference,"original_signal":original,
        "source_input":source,"noise_tau":noise["tau"],"noise_rho":noise["rho"],
        "source_probability":probability,"amplitude":amplitude,"noise_geometry":noise,
        "frequency_diagnostic":frequency_diagnostic,"common_baseline_period":np.asarray(baseline_period)}
