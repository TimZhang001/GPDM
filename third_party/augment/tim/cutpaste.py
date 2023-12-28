'''
* 当前版本: 1.0.0
* 作    者: Tim.Zhang
* 日    期: 2023.11.17
* 备    注: 1.基本的基于cutpaste的数据增强,包括cutpaste_normal, cutpaste_scar, cutpaste_union, cutpaste_3way
           2.暂时不支持基于Mask的CutPaste
           3.暂时不支持对特征图像的CutPaste
'''

import random
import math
from torchvision import transforms
import torch 
from PIL import Image

class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""
    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform
        
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness = colorJitter,
                                                      contrast   = colorJitter,
                                                      saturation = colorJitter,
                                                      hue        = colorJitter)
    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img     = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img
    
class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """
    def __init__(self, area_ratio=[0.02,0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio   = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        #TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]
        
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h
        
        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        aspect    = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
        
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))
        
        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box   = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)
        
        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))
        
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented  = img.copy()
        augmented.paste(patch, insert_box)
        
        return super().__call__(img, augmented)

class CutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """
    def __init__(self, width=[2,16], height=[10,25], rotation=[-45,45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation
    
    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        
        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)
        
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg,expand=True)
        
        #paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")
        
        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)
        
        return super().__call__(img, augmented)
    
class CutPasteUnion(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar   = CutPasteScar(**kwags)
    
    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)

class CutPaste3Way(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar   = CutPasteScar(**kwags)
    
    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar     = self.scar(img)
        
        return org, cutpaste_normal, cutpaste_scar

# ----- Tim.Zhang Add ----- #
class CutPasteTensor(object):
    """Randomly copy one patche from the tensor image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """
    def __init__(self, area_ratio=[0.05,0.10], aspect_ratio=0.3, **kwags):
        super(CutPasteTensor, self).__init__(**kwags)
        self.area_ratio   = area_ratio
        self.aspect_ratio = aspect_ratio

    def do_cutpaste(self, img_tensor):
        #TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img_tensor.shape[1]
        w = img_tensor.shape[2]
        
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h
        
        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        aspect    = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
        
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))
        
        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box   = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img_tensor[:, box[1]:box[3], box[0]:box[2]]
        
  
        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))
        
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented  = img_tensor.clone()
        augmented[:, insert_box[1]:insert_box[3], insert_box[0]:insert_box[2]] = patch
        

        return augmented, box, insert_box

    def __call__(self, img_tensor):
        if len(img_tensor.shape) == 4:
            augmented_tensor = torch.zeros_like(img_tensor).to(img_tensor.device)
            box              = []
            insert_box       = []
            for i in range(img_tensor.shape[0]):
                augmented_tensor[i], box_, insert_box_ = self.do_cutpaste(img_tensor[i])
                box.append(box_)
                insert_box.append(insert_box_)
        else:
            augmented_tensor, box, insert_box = self.do_cutpaste(img_tensor)
        
        return augmented_tensor, box, insert_box
        

class CutPasteMaskTensor(object):
    """Randomly copy one patche from the image tensor and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """
    def __init__(self, area_ratio=[0.5, 2.0], min_ratio=0.01):
        super(CutPasteMaskTensor, self).__init__()
        self.area_ratio = area_ratio
        self.min_ratio  = min_ratio

    def __call__(self, img_tensor, mask_tensor, return_all=False):
        #TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img_tensor.shape[1]
        w = img_tensor.shape[2]
        
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h
        
        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        aspect    = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
        
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))
        
        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box   = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img_tensor[:, box[1]:box[3], box[0]:box[2]]
        
  
        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))
        
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented  = img_tensor.clone()
        augmented[:, insert_box[1]:insert_box[3], insert_box[0]:insert_box[2]] = patch
        
        if return_all:
            return augmented, img_tensor
        else:
            return augmented


def test_function1():
    CP_Augmenta = CutPasteNormal()
    input_image = Image.open('/home/zhangss/PHDPaper/06_OneShotSynthesis1/datasets/grid/image/broken_003.png')
    input_image = input_image.convert('RGB')

    org, cutpaste_normal = CP_Augmenta(input_image)
    org.save('/home/zhangss/PHDPaper/06_OneShotSynthesis1/datasets/grid/image/broken_003_org.png')
    cutpaste_normal.save('/home/zhangss/PHDPaper/06_OneShotSynthesis1/datasets/grid/image/broken_003_cutpaste_normal.png')


def test_function2():
    CP_Augmenta  = CutPasteNormal()
    input_tensor = torch.randn(256, 224, 224)

    org, cutpaste_normal = CP_Augmenta(input_tensor)


# main 测试程序
if __name__ == '__main__':
    test_function2()

