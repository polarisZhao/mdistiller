import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def single_stage_at_loss(f_s, f_t, p):
    # _at -> 将多个通道压缩到单通道
    def _at(feat, p):
        return F.normalize(feat.pow(p).mean(1).reshape(feat.size(0), -1))
    # 使用 adaptive avg pool 对齐 student 和 teacher 的宽度和高度
    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    return (_at(f_s, p) - _at(f_t, p)).pow(2).mean()


def at_loss(g_s, g_t, p):
    # 对不同的 feature 分别计算 at_loss
    return sum([single_stage_at_loss(f_s, f_t, p) for f_s, f_t in zip(g_s, g_t)])


class AT(Distiller):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
    src code: https://github.com/szagoruyko/attention-transfer
    """

    def __init__(self, student, teacher, cfg):
        super(AT, self).__init__(student, teacher)
        self.p = cfg.AT.P
        self.ce_loss_weight = cfg.AT.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.AT.LOSS.FEAT_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # 对特征层计算 AT Loss
        loss_feat = self.feat_loss_weight * at_loss(
            feature_student["feats"][1:], feature_teacher["feats"][1:], self.p
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict
