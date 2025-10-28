"""
DFormerv2-Large with CCS Shape Prior - Paper Implementation
基于CVPR 2025论文的严谨实现配置

核心特点：
1. 严格遵循论文的数学公式
2. 使用变分对偶算法
3. 支持完整的消融实验
"""

from .._base_.datasets.Wheatlodgingdata import *

""" Settings for network """
C.backbone = "DFormerv2_L"
C.pretrained_model = "/root/DFormer/checkpoints/pretrained/DFormerv2_Large_pretrained.pth"
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

""" CCS Shape Prior Settings - 基于论文的参数 """
# CCS核心参数
C.use_ccs = True
C.ccs_num_centers = 5                    # 星形中心数量
C.ccs_temperature = 1.0                  # Softmax温度参数
C.ccs_variational_weight = 0.1           # 变分权重
C.ccs_shape_lambda = 0.1                 # 形状损失权重
C.ccs_learnable_centers = True           # 是否学习中心位置
C.ccs_learnable_radius = True            # 是否学习半径函数

# 实验标识
C.experiment_type = "paper_ccs"
C.implementation = "paper_based"

"""Train Config - 基于论文的训练配置"""
C.lr = 2e-5  # 预训练模型使用较小的学习率
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 1  # 减少batch size以节省内存
C.nepochs = 150   # 论文实验使用150轮
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 0  # 单GPU训练时设置为0避免多进程问题
C.train_scale_array = [0.75, 1, 1.25]
C.warm_up_epoch = 10

# 正则化
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# 辅助头配置
C.aux_rate = 0.0  # 禁用辅助头，专注于CCS
C.aux_index = 0   # 辅助头索引
C.drop_path_rate = 0.1

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

"""Path Config - 论文实验专用路径"""
import time
import os

# 论文实验根目录
paper_root = "experiments/paper_ccs"
experiment_name = "DFormerv2_L_CCS_Paper"
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

C.log_dir = osp.abspath(osp.join(paper_root, experiment_name, timestamp))
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

# 论文实验元数据
C.experiment_metadata = {
    'experiment_type': 'paper_ccs',
    'base_model': 'DFormerv2_Large',
    'dataset': 'Wheatlodgingdata',
    'paper_reference': 'Zhao et al. CVPR 2025',
    'ccs_config': {
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
    },
    'mathematical_formulation': {
        'star_shape_field': 'φ(x) = r(θ) - d(x, c)',
        'convex_combination': 'φ_CCS(x) = Σ_i α_i(x) · φ_i(x)',
        'variational_form': 'u* = softmax(f + μ · φ_CCS(x))',
        'shape_loss': 'L_shape = ∫_Ω φ_CCS(x) · (1 - u(x)) dx'
    }
}



