import numpy as np
import scipy.signal
import torch
import albumentations as Augment

# ------- 基于albumentations的数据增强 ---------------
class BaseAugmentPipe(torch.nn.Module):
    def __init__(self, target_size, use_agument, crop_rate=0.8, object_type='object'):
        super().__init__()
        
        self.use_agument = use_agument
        assert object_type in ['object', 'texture']
        
        self.augment_fun = self.get_augmentations(target_size, crop_rate, object_type)
    
    def get_augmentations(self, target_size, crop_rate=0.8, object_type='object'):

        # 进行resize
        aug_resize      = Augment.Resize(height=target_size[0], width=target_size[1], p=1)
        
        # 进行水平垂直翻转
        aug_flip        = Augment.Compose([Augment.VerticalFlip(p=0.5), Augment.HorizontalFlip(p=0.5),])
        
        # 90度旋转
        aug_rotate90    = Augment.RandomRotate90(p=0.5)

        # 随机旋转
        aug_scaleRotate = Augment.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=0.5)
        
        # 亮度/对比度拉升
        aug_brtContrast = Augment.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5)

        # HueSaturationValue
        #aug_hueSat      = Augment.HueSaturationValue(p=0.5)

        # RGBShift
        #aug_rgbShift    = Augment.RGBShift(p=0.5)

        # ColorJitter
        aug_colorJitter = Augment.ColorJitter(p=0.5)

        # Sharpen
        #aug_sharpen     = Augment.Sharpen(p=0.5)

        # random crop
        # crop_size 必须是8的倍数
        crop_size = int(target_size[0] * crop_rate), int(target_size[1] * crop_rate) 
        crop_size = (crop_size[0] // 8) * 8, (crop_size[1] // 8) * 8
        aug_randomCrop  = Augment.RandomCrop(height=int(crop_size[0]), width=int(crop_size[1]), p=1)

        # 组合
        if self.use_agument:
            if object_type == 'object':
                augment_fun = Augment.Compose([aug_resize, 
                                               aug_flip, 
                                               aug_rotate90, 
                                               aug_scaleRotate, 
                                               aug_brtContrast,
                                               #aug_hueSat, aug_rgbShift,
                                               #aug_colorJitter, 
                                               #aug_sharpen,
                                               aug_randomCrop])
            else:
                augment_fun = Augment.Compose([aug_resize, 
                                               aug_flip, 
                                               aug_rotate90, 
                                               #aug_scaleRotate, 
                                               aug_brtContrast,
                                               #aug_hueSat, aug_rgbShift,
                                               #aug_colorJitter, 
                                               #aug_sharpen,
                                               aug_randomCrop])
        else:
            augment_fun  = Augment.Compose([aug_resize, aug_randomCrop])
        return augment_fun
        
    def forward(self, image, mask=None):
        if mask is None:
            processed = self.augment_fun(image = image)
            return processed['image']
        else:
            processed = self.augment_fun(image = image, mask=mask)
            return processed['image'], processed['mask']
