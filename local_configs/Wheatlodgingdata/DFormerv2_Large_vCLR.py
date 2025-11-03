"""
DFormerv2-Large with Multi-View Consistency Learning (v-CLR)
用于SCI论文实验的完整配置

实验设置：
1. Baseline: 标准DFormerv2-Large
2. with v-CLR: DFormerv2-Large + 多视图一致性学习

对比指标：
- mIoU
- Feature similarity
- Consistency rate
- Cross-view generalization
"""

from .._base_.datasets.Wheatlodgingdata import *
import time
import os

""" 模型配置 """
C.backbone = "DFormerv2_L"
C.pretrained_model = "/root/DFormer/checkpoints/pretrained/DFormerv2_Large_pretrained.pth"
C.decoder = "ham"
C.decoder_embed_dim = 512

""" v-CLR 实验配置 """
C.use_multi_view_consistency = True  # 启用多视图一致性学习
C.consistency_loss_weight = 0.1  # 一致性损失权重
C.alignment_loss_weight = 0.05  # 对齐损失权重
C.num_views = 2  # 生成的视图数量
C.consistency_type = "cosine_similarity"  # 或 "mse", "contrastive"

""" 实验设置 """
C.experiment_name = "DFormerv2_vCLR"  # 实验名称
C.experiment_type = "multi_view"  # "baseline" 或 "multi_view"
C.enable_visualization = True  # 是否启用可视化
C.save_experiment_results = True  # 是否保存实验结果

""" 训练配置 """
C.lr = 2e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 4
C.nepochs = 200
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 12
C.train_scale_array = [0.75, 1, 1.25]
C.warm_up_epoch = 10
C.optimizer = "AdamW"  # 添加优化器配置

# 正则化
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.1
C.aux_rate = 0.0

""" 早停机制 """
C.early_stopping = True
C.patience = 20
C.min_delta = 0.001

""" 评估配置 """
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = True
C.eval_crop_size = [500, 500]
C.checkpoint_start_epoch = 20
C.checkpoint_step = 10

""" 路径配置 """
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
C.log_dir = osp.abspath(f"checkpoints/{C.dataset_name}_{C.backbone}_vCLR_{timestamp}")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))
C.visualization_dir = osp.abspath(osp.join(C.log_dir, "visualizations"))

if not os.path.exists(C.log_dir):
    os.makedirs(C.log_dir, exist_ok=True)
if not os.path.exists(C.visualization_dir):
    os.makedirs(C.visualization_dir, exist_ok=True)

exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_dir + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"
C.experiment_results_file = C.log_dir + "/experiment_results.json"

