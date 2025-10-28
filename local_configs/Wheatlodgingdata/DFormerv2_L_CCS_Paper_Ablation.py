"""
DFormerv2-Large with CCS Shape Prior - Paper Ablation Study
基于CVPR 2025论文的消融实验配置

消融实验设计：
1. 基线：不使用CCS
2. 中心数量：3, 5, 7
3. 温度参数：0.5, 1.0, 2.0
4. 变分权重：0.05, 0.1, 0.2
5. 形状损失权重：0.05, 0.1, 0.2
6. 中心学习策略：固定vs学习
7. 半径学习策略：固定vs学习
"""

from .._base_.datasets.Wheatlodgingdata import *

""" Settings for network """
C.backbone = "DFormerv2_L"
C.pretrained_model = "/root/DFormer/checkpoints/pretrained/DFormerv2_Large_pretrained.pth"
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

""" CCS Shape Prior Settings - 消融实验参数 """
# 基础CCS配置
C.use_ccs = True
C.ccs_num_centers = 5
C.ccs_temperature = 1.0
C.ccs_variational_weight = 0.1
C.ccs_shape_lambda = 0.1
C.ccs_learnable_centers = True
C.ccs_learnable_radius = True

# 消融实验标识
C.experiment_type = "paper_ablation"
C.ablation_variant = "default"  # 在训练脚本中动态设置

"""Train Config - 消融实验优化配置"""
C.lr = 2e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 2
C.nepochs = 150
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8
C.train_scale_array = [0.75, 1, 1.25]
C.warm_up_epoch = 10

# 正则化
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.1
C.aux_rate = 0.0  # 消融实验不使用辅助头

# 早停机制
C.early_stopping = True
C.patience = 15
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

"""Path Config - 消融实验目录结构"""
import time
import os

# 消融实验根目录
ablation_root = "experiments/paper_ablation"
experiment_name = f"DFormerv2_L_CCS_Paper_{C.ablation_variant}"

# 添加缺失的配置属性
C.pad = False
C.eval_h = 500
C.eval_w = 500
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

C.log_dir = osp.abspath(osp.join(ablation_root, experiment_name, timestamp))
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoints"))

# 创建目录
if not os.path.exists(C.log_dir):
    os.makedirs(C.log_dir, exist_ok=True)

# 日志文件
exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = osp.join(C.log_dir, f"log_{exp_time}.log")
C.link_log_file = osp.join(C.log_dir, "log_last.log")
C.val_log_file = osp.join(C.log_dir, f"val_{exp_time}.log")
C.link_val_log_file = osp.join(C.log_dir, "val_last.log")

# 消融实验元数据
C.experiment_metadata = {
    'experiment_type': 'paper_ablation',
    'base_model': 'DFormerv2_Large',
    'dataset': 'Wheatlodgingdata',
    'paper_reference': 'Zhao et al. CVPR 2025',
    'ablation_variant': C.ablation_variant,
    'ccs_config': {
        'use_ccs': C.use_ccs,
        'num_centers': C.ccs_num_centers,
        'temperature': C.ccs_temperature,
        'variational_weight': C.ccs_variational_weight,
        'shape_lambda': C.ccs_shape_lambda,
        'learnable_centers': C.ccs_learnable_centers,
        'learnable_radius': C.ccs_learnable_radius
    },
    'training_config': {
        'lr': C.lr,
        'batch_size': C.batch_size,
        'epochs': C.nepochs,
        'optimizer': C.optimizer
    }
}



