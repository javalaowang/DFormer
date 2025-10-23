from .._base_.datasets.Wheatlodgingdata import *

""" Settings for network, this would be different for each kind of model"""
C.backbone = "DFormerv2_L"  # 使用DFormerv2-Large架构
C.pretrained_model = "/root/DFormer/checkpoints/pretrained/DFormerv2_Large_pretrained.pth"  # 使用预训练模型
C.decoder = "ham"
C.decoder_embed_dim = 512  # DFormerv2-Large使用512维
C.optimizer = "AdamW"

"""Train Config - 基于预训练模型的优化配置"""
C.lr = 2e-5  # 预训练模型使用较小的学习率
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01  # 预训练模型使用较小的权重衰减
C.batch_size = 2  # 保持较小的batch size
C.nepochs = 200  # 预训练模型通常需要更少的训练轮数
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8
C.train_scale_array = [0.75, 1, 1.25]  # 适中的数据增强
C.warm_up_epoch = 10  # 预训练模型使用较短的warm up

# 正则化技术 - 预训练模型使用较轻的正则化
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.1  # 预训练模型使用较小的dropout
C.aux_rate = 0.0

# 早停机制
C.early_stopping = True
C.patience = 20  # 预训练模型使用更大的patience
C.min_delta = 0.001

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = True
C.eval_crop_size = [500, 500]

"""Store Config"""
C.checkpoint_start_epoch = 20
C.checkpoint_step = 10

"""Path Config"""
import time
import os
C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone + "_pretrained_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()).replace(" ", "_"))
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))
if not os.path.exists(C.log_dir):
    os.makedirs(C.log_dir, exist_ok=True)
exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"
