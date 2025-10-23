from .._base_.datasets.Wheatlodgingdata import *

""" Settings for network, this would be different for each kind of model"""
C.backbone = "DFormer-Large"  # Remember change the path below.
C.pretrained_model = None
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 4  # 减小batch size因为图像较大(500x500)
C.nepochs = 300  # 减少epoch数，因为数据集较小
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8
C.train_scale_array = [0.75, 1, 1.25]  # 减少scale变化，因为图像已经较大
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.15
C.aux_rate = 0.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]  # [0.75, 1, 1.25] #
C.eval_flip = True  # False #
C.eval_crop_size = [500, 500]  # [height weight] 匹配图像尺寸

"""Store Config"""
C.checkpoint_start_epoch = 50  # 提前开始保存checkpoint
C.checkpoint_step = 10  # 更频繁保存

"""Path Config"""
C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone)
C.log_dir = C.log_dir + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()).replace(" ", "_")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(
    osp.join(C.log_dir, "checkpoint")
)  #'/mnt/sda/repos/2023_RGBX/pretrained/'#osp.abspath(osp.join(C.log_dir, "checkpoint"))
if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir, exist_ok=True)
exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"
