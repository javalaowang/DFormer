from .._base_.datasets.Wheatlodgingdata import *

""" Settings for network, this would be different for each kind of model"""
C.backbone = "DFormer-Small"  # 使用较小模型减少过拟合
C.pretrained_model = None
C.decoder = "ham"
C.decoder_embed_dim = 256  # 减少解码器复杂度
C.optimizer = "AdamW"

"""Train Config - 过拟合优化"""
C.lr = 1e-5  # 降低学习率从6e-5到1e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.05  # 增加权重衰减从0.01到0.05
C.batch_size = 2  # 减小batch size
C.nepochs = 150  # 减少训练轮数
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5]  # 增强数据增强
C.warm_up_epoch = 20  # 增加预热轮数

# 正则化技术
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.3  # 增加dropout从0.15到0.3
C.aux_rate = 0.0

# 早停机制
C.early_stopping = True
C.patience = 15  # 连续15轮验证性能不提升就停止
C.min_delta = 0.001  # 最小改善阈值

# 损失函数优化
C.loss_type = "focal"  # 使用Focal Loss处理类别不平衡
C.focal_alpha = 0.25
C.focal_gamma = 2.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = True
C.eval_crop_size = [500, 500]

"""Store Config - 更频繁保存"""
C.checkpoint_start_epoch = 20  # 更早开始保存
C.checkpoint_step = 5  # 更频繁保存

"""Path Config"""
C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone + "_improved")
C.log_dir = C.log_dir + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()).replace(" ", "_")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(
    osp.join(C.log_dir, "checkpoint")
)
if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir, exist_ok=True)
exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"


