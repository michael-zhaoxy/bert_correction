import torch
import torch.nn as nn
import torch.nn.functional as F


class CpoLoss(nn.Module):
    """
    CpoLoss.
    from https://arxiv.org/pdf/2203.00991.pdf
    """

    def __init__(self, k=5):
        super(CpoLoss, self).__init__()
        self.k = k

    def forward(self, logits, target, mask=None):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        batchsize, vocab_size, seqlen  = logits.size()
        logits = logits.contiguous().view(batchsize * seqlen, vocab_size)

        probs = torch.softmax(logits, dim=-1)  # BS*V
        target = target.contiguous().view(-1, 1).long()  # BS*1
        pos_prob = probs.gather(1, target)  # BS*1
        # 正样本概率
        neg_prob, neg_idx = torch.topk(probs, self.k)  # BS * K
        # 负样本概率 BS

        # Contrastive Probability Optimization Objective
        # 正样本概率-负样本概率，求均值
        # 如果正样本在前k个，则负样本是K-1个，反之，负样本为K个

        expand_pos_idx = target.expand(-1, self.k)
        not_equals = expand_pos_idx != neg_idx
        not_equals_num = torch.sum(not_equals, dim=-1)

        expand_pos_pob = pos_prob.expand(-1, self.k)
        minus_porb_sum = torch.sum(expand_pos_pob - neg_prob, dim=-1)
        # 正样本概率-正样本的概率等于0，只是除数不一样
        batch_loss = - minus_porb_sum / not_equals_num
        loss = batch_loss.mean()
        return loss