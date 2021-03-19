import mindspore
import mindspore.ops as P
from mindspore.ops import functional as F
import numpy as np

def global_norm(x):
    sqrt = P.Sqrt()
    reduce_sum = P.ReduceSum()
    l2 = P.L2Normalize()
    x  = sqrt(reduce_sum(P.functional.square(l2(x))))
    return x

def clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
    """
    wrapper around ops.clip_by_global_norm that also does summary ops of norms
    compute norms
    use global_norm with one element to handle IndexedSlices vs dense
    """
    norms = [global_norm(t) for t in t_list]

    #summary ops before clipping
    summary_ops = []
    scalar_summary = P.ScalarSummary()
    for ns, v in zip(norms, variables):
        name = 'norm_pre_clip/' + v.name.replace(":", "_")
        #print(ns.shape)
        summary_ops.append(scalar_summary('name', ns))

    #clip
    clipped_t_list, clip_norm = P.clip_by_global_norm(t_list, clip_norm)

    # summary ops after clipping
    norms_post = [global_norm(t) for t in clipped_t_list]
    for ns, v in zip(norms_post,variables):
        name = 'norm_post_clip/' + v.name.replace(":", "_")
        summary_ops.append(scalar_summary('name', ns))
    summary_ops.append(scalar_summary(norm_name, clip_norm))
       
    return clipped_t_list, clip_norm, summary_ops

def clip_grads(grads, all_clip_norm_val=1.0, do_summaries=False):
    # grads = [(grad1, var1), (grad2, var2), ...
    # name = 'norm_grad'
    # grad_tensors =[g for g, v in grads]
    # vv = [v for g, v in grads]
    # scaled_val = all_clip_norm_val
    # if do_summaries:
    #     clipped_tensors, g_norm, summary_ops = clip_by_global_norm_summary(
    #         grad_tensors, scaled_val, name, vv)
    # else:
    #     summary_ops = []
    #     clipped_tensors, g_norm = P.clip_by_global_norm(grad_tensors, scaled_val)
    # ret = []
    # for t, (g, v) in zip(clipped_tensors, grads):
    #     ret.append((t, v))
    
    # assert len(ret) == len(grads)

    # return ret, summary_ops
    grad_tensors = grads
    scaled_val = all_clip_norm_val
    clipped_tensors = P.clip_by_global_norm(grad_tensors, scaled_val)
    return clipped_tensors

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        g0, v0 = grad_and_vars[0]
        if g0 is None:
            average_grads.append((g0, v0))
            continue
        # the gradient is type IndexedSlices
        # to do 
        # a normal tensor can just do a simple  average
        grads = []
        for g, v in grad_and_vars:
            expand_g = P.ExpandDims()(g, 0)
            grads.append(expand_g)
        
        # Average over the 'tower' dimension
        grad = P.Concat(0)(grads)
        grad = P.ReduceMean(grad, 0)
        
        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)
    assert len(average_grads) == len(list(zip(*tower_grads)))
    return average_grads