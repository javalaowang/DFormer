"""
DFormer-Base with CCS Shape Prior
集成凸组合星形(CCS)约束的配置文件

使用方法:
    bash train.sh  (修改--config=local_configs.WheatLodging.DFormer_Base_CCS)
"""

from .._base_.datasets.WheatLodging import *

""" Settings for network """
C.backbone = "DFormer-Base"
C.pretrained_model = "checkpoints/pretrained/DFormer_Base.pth.tar"
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

""" CCS Shape Prior Settings """
C.use_ccs = True          # 启用CCS约束
C.num_centers = 5         # 星形中心数量 (3-7推荐)
C.ccs_lambda = 0.1        # CCS损失权重 (0.05-0.2)
C.learnable_centers = True  # 学习中心位置
C.ccs_temperature = 1.0   # Softmax温度

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 300
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.1
C.aux_rate = 0.4

"""Eval Config"""
C.eval_iter = 20
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = True
C.eval_crop_size = [480, 640]

"""Store Config"""
C.checkpoint_start_epoch = 100
C.checkpoint_step = 20

"""Path Config"""
C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone + "_CCS")
C.log_dir = C.log_dir + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()).replace(" ", "_")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir, exist_ok=True)

exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"

