import os

import numpy as np
import torch
import torch.distributed as dist


MAX_GROUP_SIZE = 32
current_process_group = None


def init_distributed_training(local_rank, port_idx=0):
    ports = ['12950', '12951', '12952', '12953']
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = ports[port_idx]
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                         rank=local_rank)


def get_world_size():
    return dist.get_world_size()


def get_local_rank():
    return dist.get_rank()


def barrier():
    dist.barrier()
    return


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def reduce_scalar(scalar):
    if hasattr(scalar, 'device'):
        scalar = scalar.item()
    reduced_scalar = reduce_tensor(torch.tensor(scalar).cuda()).item()
    return reduced_scalar


def all_gather_no_grad(tensor):
    tensor = tensor.contiguous()
    out = [torch.empty_like(tensor) for i in range(get_world_size())]
    dist.all_gather(out, tensor)
    out = torch.cat(out)
    return out


class AllGatherWithGrads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, *output_list):
        dist.all_gather(list(output_list), tensor)
        return tuple(output_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # collect some info about processes
        local_rank = get_local_rank()
        world_size = get_world_size()
        # use an all-reduce to provide each process with the full grads on
        # all of the all-gathered tensors
        stacked_proc_grads = torch.stack([t for t in grad_outputs], dim=0)
        dist.all_reduce(stacked_proc_grads, op=dist.ReduceOp.SUM)
        # chunk the all-reduced grad tensor into tensors of grads
        # w.r.t. the tensors gathered from each individual process
        per_proc_grads = torch.chunk(stacked_proc_grads, world_size, dim=0)
        # get the grads w.r.t. the original tensor from the current process
        local_grad = per_proc_grads[local_rank]
        return (local_grad,) + grad_outputs


def all_gather_local_single(tensor):
    all_gather_fn = AllGatherWithGrads.apply
    out = [torch.empty_like(tensor) for i in range(get_world_size())]
    out = all_gather_fn(tensor, *out)
    out = torch.cat(out)
    return out


def all_gather_local_multiple(*tensors):
    t_out = []
    for t in tensors:
        t_out.append(all_gather_local_single(t))
    return t_out