from .. import *

# Dataset config
"""Dataset Path"""
C.dataset_name = "Wheatlodgingdata"
C.dataset_path = osp.join(C.root_dir, "Wheatlodgingdata")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".jpg"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = False  # 标签从0开始，不需要转换
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, "HHA")
C.x_format = ".png"  # 需要将TIF转换为PNG
C.x_is_single_channel = True  # HHA是单通道
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 357
C.num_eval_imgs = 153
C.num_classes = 3
C.class_names = [
    "background",
    "wheat",
    "lodging",
]

"""Image Config"""
C.background = 255
C.image_height = 500
C.image_width = 500
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])
