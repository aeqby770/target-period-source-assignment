"""Euclidean box projection preserving a feasible experiment mean.
For each group, minimize squared change from the original periods subject to
the period-grid bounds and the closest feasible original group mean.
The common shift and clipping preserve the original within-group ordering.
"""
from pathlib import Path
import os
CONFIG={"paths":{},"threads":os.cpu_count(),"plot":{}}
os.environ["OMP_NUM_THREADS"]=str(CONFIG["threads"])
import numpy as np
project_periods_with_mean=lambda values,means,membership,lower,upper:(
    counts:=membership.sum(axis=1),
    width:=int(counts.max()),
    indices:=np.argsort(np.where(membership,np.arange(membership.shape[1])[None],membership.shape[1]),axis=1)[:,:width],
    valid:=np.arange(width)[None]<counts[:,None],
    padded:=values[:,indices],
    breaks:=np.sort(np.concatenate([
        np.where(valid[None],lower-padded,np.inf),
        np.where(valid[None],upper-padded,np.inf)],axis=-1),axis=-1),
    mass:=(np.clip(padded[:,:,None,:]+breaks[:,:,:,None],lower,upper)*valid[None,:,None,:]).sum(axis=-1),
    target:=np.clip(means,lower,upper)*counts[None],
    left:=np.minimum(np.maximum((mass<target[:,:,None]).sum(axis=-1)-1,0),2*counts.astype(int)[None]-2),
    right:=left+1,
    at_left:=np.take_along_axis(mass,left[:,:,None],axis=-1)[:,:,0],
    at_right:=np.take_along_axis(mass,right[:,:,None],axis=-1)[:,:,0],
    shift_left:=np.take_along_axis(breaks,left[:,:,None],axis=-1)[:,:,0],
    shift_right:=np.take_along_axis(breaks,right[:,:,None],axis=-1)[:,:,0],
    fraction:=np.divide(target-at_left,at_right-at_left,out=np.zeros_like(target),where=at_right>at_left),
    shift:=shift_left+fraction*(shift_right-shift_left),
    needs:=(((padded<lower)|(padded>upper))&valid[None]).any(axis=-1),
    shift:=np.where(needs,shift,0.),
    np.clip(values+shift@membership,lower,upper))[-1]
