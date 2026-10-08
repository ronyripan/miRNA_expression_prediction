"""
Loss functions
"""

import os
import sys
import functools
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import layers

class NegativeBinomialLoss(nn.Module):
    """
    Negative binomial loss. Preds should be a tuple of (mean, dispersion)
    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        eps: float = 1e-10,
        l1_lambda: float = 0.0,
        mean: bool = True,
    ):
        super(NegativeBinomialLoss, self).__init__()
        self.loss = negative_binom_loss(
            scale_factor=scale_factor,
            eps=eps,
            mean=mean,
            debug=True,
        )
        self.l1_lambda = l1_lambda

    def forward(self, preds, target):
        preds, theta = preds[:2]
        l = self.loss(
            preds=preds,
            theta=theta,
            truth=target,
        )
        encoded = preds[:-1]
        l += self.l1_lambda * torch.abs(encoded).sum()
        return l
    
def negative_binom_loss(
    scale_factor: float = 1.0,
    eps: float = 1e-10,
    mean: bool = True,
    debug: bool = False,
    tb: SummaryWriter = None,
) -> Callable:
    """
    Return a function that calculates the binomial loss
    https://github.com/theislab/dca/blob/master/dca/loss.py

    combination of the Poisson distribution and a gamma distribution is a negative binomial distribution
    """

    def loss(preds, theta, truth, tb_step: int = None):
        """Calculates negative binomial loss as defined in the NB class in link above"""
        y_true = truth
        y_pred = preds * scale_factor

        if debug:  # Sanity check before loss calculation
            assert not torch.isnan(y_pred).any(), y_pred
            assert not torch.isinf(y_pred).any(), y_pred
            assert not (y_pred < 0).any()  # should be non-negative
            assert not (theta < 0).any()

        # Clip theta values
        theta = torch.clamp(theta, max=1e6)

        t1 = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + eps)
        )
        t2 = (theta + y_true) * torch.log1p(y_pred / (theta + eps)) + (
            y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
        )
        if debug:  # Sanity check after calculating loss
            assert not torch.isnan(t1).any(), t1
            assert not torch.isinf(t1).any(), (t1, torch.sum(torch.isinf(t1)))
            assert not torch.isnan(t2).any(), t2
            assert not torch.isinf(t2).any(), t2

        retval = t1 + t2
        if debug:
            assert not torch.isnan(retval).any(), retval
            assert not torch.isinf(retval).any(), retval

        if tb is not None and tb_step is not None:
            tb.add_histogram("nb/t1", t1, global_step=tb_step)
            tb.add_histogram("nb/t2", t2, global_step=tb_step)

        return torch.mean(retval) if mean else retval

    return loss

class PairedLoss(nn.Module):
    """
    Paired loss function. Automatically unpacks and encourages the encoded representation to be similar
    using a given distance function. link_strength parameter controls how strongly we encourage this
    loss2_weight controls how strongly we weight the second loss, relative to the first
    A value of 1.0 indicates that they receive equal weight, and a value larger indicates
    that the second loss receives greater weight.

    link_func should be a callable that takes in the two encoded representations and outputs a metric
    where a larger value indicates greater divergence
    """

    def __init__(
        self,
        loss1=NegativeBinomialLoss,
        loss2=NegativeBinomialLoss,
        link_func=lambda x, y: (x - y).abs().mean(),
        link_strength=1e-3,
    ):
        super(PairedLoss, self).__init__()
        self.loss1 = loss1()
        self.loss2 = nn.MSELoss()
        self.link = link_strength
        self.link_f = link_func

        self.warmup = layers.SigmoidWarmup(
            midpoint=1000,
            maximum=link_strength,
        )

    def forward(self, preds, target):
        """Unpack and feed to each loss, averaging at end"""
        preds1, preds2 = preds
        target1, target2 = target

        loss1 = self.loss1(preds1, target1)
        loss2 = self.loss2(preds2, target2)
        retval = loss1 + loss2

        # Align the encoded representation assuming the last output is encoded representation
        encoded1 = preds1[-1]
        encoded2 = preds2[-1]
        if self.link > 0:
            l = next(self.warmup)
            if l > 1e-6:
                d = self.link_f(encoded1, encoded2).mean()
                retval += l * d

        return retval
    

class QuadLoss(PairedLoss):
    """
    Paired loss, but for the spliced autoencoder with 4 outputs
    """

    def __init__(
        self,
        loss1=NegativeBinomialLoss,
        loss2=NegativeBinomialLoss,
        loss2_weight: float = 1.0, #1.33
        cross_weight: float = 1.0,
        cross_warmup_delay: int = 0,
        link_strength: float = 0.0,
        link_func: Callable = lambda x, y: (x - y).abs().mean(),
        link_warmup_delay: int = 0,
        record_history: bool = False,
    ):
        super(QuadLoss, self).__init__()
        self.loss1 = loss1()
        self.loss2 = nn.MSELoss()
        self.loss2_weight = loss2_weight
        self.history = []  # Eventually contains list of tuples per call
        self.record_history = record_history

        if link_warmup_delay:
            self.warmup = layers.SigmoidWarmup(
                midpoint=link_warmup_delay,
                maximum=link_strength,
            )
            # self.warmup = layers.DelayedLinearWarmup(
            #     delay=warmup_delay,
            #     t_max=link_strength,
            #     inc=1e-3,
            # )
        else:
            self.warmup = layers.NullWarmup(t_max=link_strength)
        if cross_warmup_delay:
            self.cross_warmup = layers.SigmoidWarmup(
                midpoint=cross_warmup_delay,
                maximum=cross_weight,
            )
        else:
            self.cross_warmup = layers.NullWarmup(t_max=cross_weight)

        self.link_strength = link_strength
        self.link_func = link_func

    def get_component_losses(self, preds, target):
        """
        Return the four losses that go into the overall loss, without scaling
        """
        preds1, preds2, preds3, preds4 = preds #preds1 is mRNA and preds2 is miRNA
        
        target1 = target[0]
        target2 = target[1] # Both are torch tensors

        loss1 = self.loss1(preds1, target1) #mrna loss
        loss2 = self.loss1(preds2, target1)
        #loss3 = torch.sqrt(self.loss2(preds3[0], target2))  #mirna loss

        return loss1, loss2

    def forward(self, preds, target):
        loss1, loss2 = self.get_component_losses(preds, target)
        #print('mrna loss:  {}; mirna loss: {}'.format(loss1, loss2))

        if self.record_history:
            detensor = lambda x: x.detach().cpu().numpy().item()
            self.history.append([detensor(l) for l in (loss1, loss2)])

        loss = loss1 + loss2

        return loss