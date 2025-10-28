"""
DFormerv2-Large with CCS Shape Prior - Quick Test
快速验证CCS模块的有效性

配置特点：
- 较少的训练轮数（20 epochs）
- 较小的batch size（1）
- 简化的CCS配置
"""

from .._base_.datasets.Wheatlodgingdata import *

""" Settings for network """
C.model_name = "DFormerv2_L_CCS_Quick"
C.backbone = "DFormerv2_L"
C.pretrained_model = "/root/DFormer/checkpoints/pretrained/DFormerv2_Large_pretrained.pth"
C.decoder = "ham"
C.decoder_embed_dim = 512
C.num_classes = 3

""" CCS Shape Prior Settings - 简化配置 """
C.use_ccs = True
C.ccs_num_centers = 3  # 减少中心数量
C.ccs_temperature = 1.0
C.ccs_variational_weight = 0.1
C.ccs_shape_lambda = 0.1
C.ccs_learnable_centers = True
C.ccs_learnable_radius = True

"""Train Config - 快速训练配置"""
C.optimizer = "AdamW"
C.lr = 2e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 1  # 减小batch size
C.nepochs = 20    # 减少训练轮数
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 4  # 减少worker数量
C.train_scale_array = [0.75, 1, 1.25]
C.warm_up_epoch = 2  # 减少warm up

# 正则化
C.drop_path_rate = 0.1
C.early_stopping = False  # 关闭early stopping
C.patience = 10
C.min_delta = 0.001

# 数据增强
C.train_h = 500
C.train_w = 500
C.scale_min = 0.5
C.scale_max = 2.0
C.ignore_index = 255

# 验证配置
C.eval_h = 500
C.eval_w = 500
C.eval_scale_array = [1.0]  # 简化验证
C.eval_flip = False  # 关闭flip验证

# 日志和检查点
C.log_dir = osp.abspath(osp.join(C.root_dir, "experiments", "quick_ccs", C.model_name + "_" + time.strftime("%Y%m%d_%H%M%S")))
C.log_file = osp.join(C.log_dir, "log_" + time.strftime("%Y_%m_%d_%H_%M_%S") + ".log")
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoints"))

# 添加缺失的配置属性
C.tb_dir = osp.join(C.log_dir, "tb")
C.val_log_file = osp.join(C.log_dir, "val_" + time.strftime("%Y_%m_%d_%H_%M_%S") + ".log")
C.link_log_file = osp.join(C.log_dir, "log_last.log")
C.link_val_log_file = osp.join(C.log_dir, "val_last.log")
C.log_dir_link = C.log_dir

# 其他必要属性
C.fix_bias = True
C.pad = False
C.aux_rate = 0.0  # 禁用辅助头，避免索引错误
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.aux_index = 0  # 修复aux_index错误
C.checkpoint_start_epoch = 1  # 添加缺失的检查点配置
C.checkpoint_step = 5

if not os.path.exists(C.log_dir):
    os.makedirs(C.log_dir, exist_ok=True)

print(f"Quick CCS Experiment: {C.model_name}")
print(f"CCS Centers: {C.ccs_num_centers}")
print(f"Training Epochs: {C.nepochs}")
print(f"Batch Size: {C.batch_size}")
print(f"Log Directory: {C.log_dir}")
