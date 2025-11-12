from .._base_.datasets.SUNRGBD import *

""" Settings for network, this would be different for each kind of model"""
C.backbone = "DFormerv2_L"  # Remember change the path below.
C.pretrained_model = "checkpoints/pretrained/SUNRGBD/SUNRGBD_DFormer_Large.pth"
C.decoder = "ham"
C.decoder_embed_dim = 1024
C.optimizer = "AdamW"

"""Train Config"""
C.lr = 8e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8  # Reduced from 16 to avoid OOM
C.nepochs = 300
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.27
C.aux_rate = 0.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [0.5, 0.75, 1, 1.25, 1.5]  # [0.75, 1, 1.25] # 0.5,0.75,1,1.25,1.5
C.eval_flip = True  # False #
C.eval_crop_size = [480, 480]  # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 200
C.checkpoint_step = 25

"""vCLR Config"""
# Enable multi-view consistency learning
C.use_multi_view_consistency = True

# Consistency loss weights
C.consistency_loss_weight = 0.1  # Weight for consistency loss
C.alignment_loss_weight = 0.05   # Weight for alignment loss

# View generation settings
C.num_views = 2  # Number of views to generate

# Consistency loss type: "cosine_similarity", "mse", "contrastive"
C.consistency_type = "cosine_similarity"

# Geometry constraint (if depth available)
C.use_geometry_constraint = True

# Experiment settings
C.experiment_name = "DFormerv2_vCLR_SUNRGBD"
C.enable_visualization = True
C.save_experiment_results = True

"""Path Config"""
C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone + "_vCLR")
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

