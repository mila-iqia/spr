import math

import numpy as np
import numpy.random as npr
import torch
import torch.nn.functional as F

from torch import nn

from src import dist_comms


def random_locs_1d(x, k_hot=1):
    '''
    Sample a k-hot mask over spatial locations for each set of conv features
    in x, where x.shape is like (n_batch, n_feat, n_x).
    '''
    # assume x is (n_batch, n_feat, n_x)
    n_batch = x.size(0)
    n_locs = x.size(2)
    idx_topk = torch.topk(torch.rand((n_batch, n_locs)), k=k_hot, dim=1)[1]
    khot_mask = torch.zeros((n_batch, n_locs)).scatter_(1, idx_topk, 1.)
    rand_locs = khot_mask.reshape((n_batch, 1, n_locs))
    rand_locs = rand_locs.type_as(x).to(x.device)
    return rand_locs


class BufferedNCE(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, feature_dim, queue_size, temperature=0.07,
                 buffer_names=('r1', 'r5')):
        super(BufferedNCE, self).__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.index = 0

        # initialize empty buffers and insertion pointers
        stdv = 1. / math.sqrt(feature_dim / 3)
        self.buffers, self.indexes = {}, {}
        for bname in buffer_names:
            empty_buf = torch.rand(self.queue_size, feature_dim,
                                   requires_grad=False).mul_(2 * stdv).add_(-stdv)
            self.buffers[bname] = empty_buf.cuda()
            self.indexes[bname] = 0
    
    def update_buffer(self, r_buf, buffer='r1'):
        # update memory
        # -- write new values starting from index self.indexes[buffer], and
        #    wrap around back to index 0 when past the end of buffer
        with torch.no_grad():
            new_size = r_buf.shape[0]
            out_ids = torch.arange(new_size, dtype=torch.long).cuda()
            out_ids = torch.fmod(out_ids + self.indexes[buffer], self.queue_size)
            # write new values to buffer
            self.buffers[buffer].index_copy_(0, out_ids, r_buf)
            # update insertion pointer for this buffer
            self.indexes[buffer] = \
                (self.indexes[buffer] + new_size) % self.queue_size
        return

    def compute_loss(self, r_src, r_trg, buffer='r1'):
        '''
        NCE for source features r_src with positive targets r_trg.
        '''
        # compute scores with positive examples
        pos_scores = (r_src * r_trg.detach()).sum(dim=1, keepdim=True)
        # compute scores with negative examples
        r_neg = self.buffers[buffer].clone().detach()
        neg_scores = torch.mm(r_src, r_neg.t())
        raw_scores = torch.cat((pos_scores, neg_scores), dim=1)
        raw_scores = torch.div(raw_scores, self.temperature).contiguous()
        accuracy = (raw_scores.argmax(1) == 0).float()
        # compute simple NCE loss
        log_scores = F.log_softmax(raw_scores, dim=1)
        loss_nce = -log_scores[:, 0].mean()
        return loss_nce, log_scores, accuracy


class LossSource2Target(nn.Module):
    '''
    Input is fixed as r_src_1, r_trg_1, r_src_2, r_trg_2.
    '''

    def __init__(self, temperature=0.1):
        super(LossSource2Target, self).__init__()
        self.temperature = temperature
        return

    def _sample_ftr(self, r_src):
        '''
        Sample a single feature vector from r_src=(n_batch, n_rkhs, n_locs)
        '''
        mask = random_locs_1d(r_src, k_hot=1).type_as(r_src)
        r_ftr = (mask * r_src).sum(dim=2)
        return r_ftr

    def _subscores(self, raw_scores, mask_pos, mask_neg, n_batch_gpu):
        '''
        Compute nce cost stuff given the fake rkhs dot products.

        Input:
          raw_scores  : (n_batch_gpu, n_batch, n_locs)
          mask_pos    : (n_batch_gpu, n_batch, n_locs)
          mask_neg    : (n_batch_gpu, n_batch, n_locs)
          n_batch_gpu : integer
        Output:
          nce_scores  : (n_batch_gpu, n_locs)
          pos_scores  : (n_batch_gpu, n_locs)
        '''
        # pos_scores includes scores for all the positive samples
        # neg_scores includes scores for all the negative samples, with
        # scores for positive samples set to the min score (-10 here)
        pos_scores = (mask_pos * raw_scores).sum(dim=1)                 # (n_batch_gpu, n_locs)
        neg_scores = (mask_neg * raw_scores) - (20. * mask_pos)         # (n_batch_gpu, n_batch, n_locs)
        neg_scores = neg_scores.reshape(n_batch_gpu, -1)                # (n_batch_gpu, n_batch * n_locs)
        mask_neg = mask_neg.reshape(n_batch_gpu, -1)                    # (n_batch_gpu, n_batch * n_locs)
        # for each set of positive examples P_i, compute the max over scores
        # for the set of negative samples N_i that are shared across P_i
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]       # (n_batch_gpu, 1)
        # compute a "partial, safe sum exp" over each negative sample set N_i,
        # to broadcast across the positive samples in P_i which share N_i
        # -- size will be (n_batch_gpu, 1)
        neg_sumexp = (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)
        # use broadcasting of neg_sumexp across the scores in P_i, to compute
        # the log-sum-exps for the denominators in the NCE log-softmaxes
        # -- size will be (n_batch_gpu, n_locs)
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes
        # compute the final log-softmax scores for NCE...
        nce_scores = pos_shiftexp - all_logsumexp
        return nce_scores, pos_scores

    def _all_scores(self, r_src, r_trg, mask_mat, which_cost):
        '''
        Compute the scores required for NCE stuff...

        Input:
          r_src      : (n_batch_gpu, n_rkhs)
          r_trg      : (n_rkhs, n_batch * n_locs)
          mask_mat   : (n_batch_gpu, n_batch)
          which_cost : which cost we're computing (1t7, 1t1, etc)
        Output:
          raw_scores : (n_batch_gpu, n_locs)
          nce_scores : (n_batch_gpu, n_locs)
          lgt_reg    : scalar
        '''
        n_batch_gpu = mask_mat.size(0)
        n_batch = mask_mat.size(1)
        n_locs = r_trg.size(1) // n_batch
        n_rkhs = r_src.size(1)
        # reshape mask_mat for ease-of-use
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, n_locs).float()
        mask_neg = 1. - mask_pos
        # compute glb->lcl matching scores for batch on this gpu
        if which_cost == 'dot':
            raw_scores = torch.mm(r_src, r_trg)
            raw_scores = raw_scores.reshape(n_batch_gpu, n_batch, n_locs)
            raw_scores = raw_scores / n_rkhs**0.5
            raw_scores = raw_scores.float()
            lgt_reg = 1e-2 * (raw_scores**2.).mean()
            raw_scores = 20. * torch.tanh(raw_scores / 20.)
        else:
            # normalize feature vectors so we can compute cosine similarities
            r_src = F.normalize(r_src.float(), p=2., dim=1, eps=1e-6)
            r_trg = F.normalize(r_trg.float(), p=2., dim=0, eps=1e-6)
            raw_scores = torch.mm(r_src, r_trg) / self.temperature
            raw_scores = raw_scores.reshape(n_batch_gpu, n_batch, n_locs)
            raw_scores = raw_scores.float()
            lgt_reg = ((1e-2 * raw_scores)**2.).mean().detach()
        # ...
        nce_scores, pos_scores = \
            self._subscores(raw_scores, mask_pos, mask_neg, n_batch_gpu)
        return nce_scores, pos_scores, lgt_reg

    def _loss_g2l(self, r_src, r_trg, mask_mat, which_cost):
        # compute the nce scores for these features
        nce_scores, raw_scores, lgt_reg = \
            self._all_scores(r_src, r_trg, mask_mat, which_cost)
        # take average over batch, and add regularization cost
        loss_g2l = -nce_scores.mean()
        return loss_g2l, lgt_reg

    def _nce_cost(self, r_src_1, r_trg_1, r_src_2, r_trg_2,
                  mask_mat, which_cost):
        # sample single feature vectors from some tensors
        r_src_1 = self._sample_ftr(r_src_1)
        r_src_2 = self._sample_ftr(r_src_2)
        # compute costs
        loss_nce_1, lgt_reg_1 = \
            self._loss_g2l(r_src_1, r_trg_2, mask_mat, which_cost)
        loss_nce_2, lgt_reg_2 = \
            self._loss_g2l(r_src_2, r_trg_1, mask_mat, which_cost)
        # combine costs
        loss_nce = 0.5 * (loss_nce_1 + loss_nce_2)
        lgt_reg = 0.5 * (lgt_reg_1 + lgt_reg_2)
        return loss_nce, lgt_reg

    def _fix_shape(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(dim=2)
        elif x.dim() == 3:
            x = x
        elif x.dim() == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
        else:
            assert False, 'Bad input!'
        return x.contiguous()

    def forward(self, r_src_1, r_trg_1, r_src_2, r_trg_2, which_cost='dot'):
        '''
        Compute src->trg nce costs for a given set of source/target features.

        Compute costs in both directions, i.e. from/to both images in each pair.

        r_src_N are source features from xN in each image pair.
        r_trg_N are target features from xN in each image pair.
        '''
        # compute feature dimensions
        n_batch = int(r_src_1.size(0))
        n_rkhs = int(r_src_1.size(1))

        # fix input shapes to be like (n_batch, n_rkhs, n_locs)
        r_src_1 = self._fix_shape(r_src_1)
        r_trg_1 = self._fix_shape(r_trg_1)
        r_src_2 = self._fix_shape(r_src_2)
        r_trg_2 = self._fix_shape(r_trg_2)

        # make masking matrix to help compute NCE costs
        idx = dist_comms.get_local_rank()
        n_procs = dist_comms.get_world_size()
        mask_mat = torch.eye(n_batch * n_procs)[idx * n_batch: (idx + 1) * n_batch].cuda()

        # get the features to use as "target" features in NCE
        r_trg_1, r_trg_2 = dist_comms.all_gather_local_multiple(r_trg_1, r_trg_2)
        # reshape for use in NCE
        r_trg_1 = r_trg_1.permute(1, 0, 2).reshape(n_rkhs, -1)
        r_trg_2 = r_trg_2.permute(1, 0, 2).reshape(n_rkhs, -1)
        # compute the various NCE source->target prediction costs
        loss_nce, lgt_reg = self._nce_cost(r_src_1, r_trg_1, r_src_2, r_trg_2,
                                           mask_mat, which_cost)
        return loss_nce, lgt_reg
