"""
This file computes metrics for a chosen checkpoint. By default, it computes SIFID (at lowest InceptionV3 scale),
LPIPS diversity, LPIPS distance to training data, mIoU (in case segmentation masks are used).
The results are saved at /${checkpoints_dir}/${exp_name}/metrics/
For SIFID, LPIPS_to_train, mIoU, and segm accuracy, the metrics are computed per each image.
LPIPS, mIoU and segm_accuracy are also computed for the whole dataset.
"""


import os
import argparse
import numpy as np
import pickle
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from OneShotMetrics import SIFID, LPIPS, LPIPS_to_train, mIoU
import matplotlib.pyplot as plt
import seaborn as sns

def parser_param():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',        type=str,  default='/home/zhangss/PHDPaper/04_SinFusion_01/lightning_logs/wood/256_256_nextnet')
    parser.add_argument('--sifid_all_layers', type=bool, default=False)
    parser.add_argument('--real_path',        type=str,  default="/home/zhangss/PHDPaper/04_SinFusion_01/images/mvtec/wood/image")
    parser.add_argument('--fake_path',        type=str,  default="/home/zhangss/PHDPaper/04_SinFusion_01/outputs/256_256_nextnet/wood")
    args = parser.parse_args()

    return args

def convert_sifid_dict(names_fake_image, sifid):
    ans = dict()
    if sifid is not None:
        for i, item in enumerate(names_fake_image):
            if isinstance(sifid, list):
                ans[item] = sifid[i]
            else:
                ans[item] = sifid
    return ans

def get_image_names(args):
    # --- Read options file from checkpoint --- #
    path_real_images = args.real_path
    path_feak_images = args.fake_path

    if not os.path.isdir(path_feak_images):
        raise ValueError("Generated images not found. Run the test script first. (%s)" % (path_feak_images))
    
    if not os.path.isdir(path_real_images):
        raise ValueError("Generated images not found. Run the test script first. (%s)" % (path_real_images))
    
    # --- Prepare files and images to compute metrics --- #
    names_fake_image = sorted(os.listdir(path_feak_images))
    list_fake_image  = list()
    for i in range(len(names_fake_image)):
        im_path          = os.path.join(path_feak_images, names_fake_image[i])
        im               = (Image.open(im_path).convert("RGB"))
        list_fake_image += [im]

    names_real_image = sorted(os.listdir(path_real_images))
    list_real_image  = list()
    im_res = (ToTensor()(list_fake_image[0]).shape[2], ToTensor()(list_fake_image[0]).shape[1])
    for i in range(len(names_real_image)):
        im_path          = os.path.join(path_real_images, names_real_image[i])
        im               = (Image.open(im_path).convert("RGB"))
        list_real_image += [im.resize(im_res, Image.BILINEAR)]

    return names_real_image, names_fake_image, list_real_image, list_fake_image, im_res

def write_sfid_to_file(save_fld_file, names_fake_image, sifid1, sifid2, sifid3, sifid4):
    
    for i in range(1, 4):
        if i == 1:
            tempfid = sifid1
        elif i == 2:
            tempfid = sifid2
        elif i == 3:
            tempfid = sifid3
        elif i == 4:
            tempfid = sifid4

        file_name = save_fld_file+"SIFID"+str(i)+".csv"
      
        # write to npy file and csv file
        if tempfid is not None:
            tempfid        = convert_sifid_dict(names_fake_image, tempfid)
            tempfid_values = np.array(list(tempfid.values()))
            tempfid_mean   = np.mean(tempfid_values)
            tempfid_std    = np.std(tempfid_values)
            tempfid_min    = np.min(tempfid_values)
            tempfid_max    = np.max(tempfid_values)
            tempfid_median = np.median(tempfid_values)
            
            #np.save(os.path.join(save_fld, str(args.epoch))+"SIFID1", sifid1)  
            with open(file_name, "w") as f:
                f.write("ave values, "    + format(tempfid_mean, ".6f") + "\n")
                f.write("std values, "    + format(tempfid_std, ".6f") + "\n")
                f.write("min values, "    + format(tempfid_min, ".6f") + "\n")
                f.write("max values, "    + format(tempfid_max, ".6f") + "\n")
                f.write("median values, " + format(tempfid_median, ".6f") + "\n")

                for k, v in tempfid.items():
                    f.write(str(k) + "," + format(v, ".6f") + "\n")
            print("--- Saved sifid metrics at %s ---" % (file_name))

            plot_hist_fig(tempfid_values, save_fld_file, hist_num = 128, title_name="SIFID"+str(i))
    
def write_lpips_to_file(save_fld_file, lpips, dist_to_tr, dist_to_tr_byimage):
    
    # write to npy file and csv file
    # np.save(os.path.join(save_fld_file, str(args.epoch))+"lpips",              lpips)
    # np.save(os.path.join(save_fld_file, str(args.epoch))+"dist_to_tr",         dist_to_tr)
    # np.save(os.path.join(save_fld_file, str(args.epoch))+"dist_to_tr_byimage", dist_to_tr_byimage)

    # write to csv file
    file_name = save_fld_file + "lpips.csv"   
    with open(file_name, "w") as f:
        # lpips values
        f.write("fake ave values, "    + format(lpips.item(), ".6f") + "\n")
        
        # dist_to_tr values
        f.write("fake to true ave values, " + format(dist_to_tr.item(), ".6f") + "\n")
        
        # dist_to_tr_byimage values
        tempfid_values = np.array(list(dist_to_tr_byimage.values()))
        tempfid_mean   = np.mean(tempfid_values)
        tempfid_std    = np.std(tempfid_values)
        tempfid_min    = np.min(tempfid_values)
        tempfid_max    = np.max(tempfid_values)
        tempfid_median = np.median(tempfid_values)
        f.write("ave values, "    + format(tempfid_mean, ".6f") + "\n")
        f.write("std values, "    + format(tempfid_std, ".6f") + "\n")
        f.write("min values, "    + format(tempfid_min, ".6f") + "\n")
        f.write("max values, "    + format(tempfid_max, ".6f") + "\n")
        f.write("median values, " + format(tempfid_median, ".6f") + "\n")
        for k, v in dist_to_tr_byimage.items():
            f.write(str(k) + "," + format(v, ".6f") + "\n")

        plot_hist_fig(tempfid_values, save_fld_file, hist_num = 128, title_name="lpips_dist_to_true")

def plot_hist_fig(values_list, save_path, hist_num = 128, title_name=""):

    # plot histogram
    save_name = os.path.join(save_path, title_name+".png")
    fig, ax   = plt.subplots(figsize=(8, 6))
    sns.distplot(values_list, bins=hist_num, kde=False, ax=ax)
    ax.set_title(title_name)
    plt.tight_layout()
    plt.show()
    fig.savefig(save_name)
    plt.close(fig)


if __name__ == "__main__":
    args = parser_param()
    print("--- Computing metrics for job %s  ---" %(args.root_path))

    # --- Get image names --- #
    names_real_image, names_fake_image, list_real_image, list_fake_image, im_res = get_image_names(args)

    # --- Compute the metrics --- #
    with torch.no_grad():
        # 计算FID(多张图像)或者SIFID(单张图像) 值越小代表真实性越高
        sifid1, sifid2, sifid3, sifid4 = SIFID(list_real_image, list_fake_image, args.sifid_all_layers)
        
        # 计算生成样本的LPIPS，值越大代表样本多样性越高
        lpips                          = LPIPS(list_fake_image)
        
        # 统计每张生成的样本距离训练集的距离（可能有多张样本），值越小代表样本真实性越高
        dist_to_tr, dist_to_tr_byimage = LPIPS_to_train(list_real_image, list_fake_image, names_fake_image)

    # --- Save the metrics under .${exp_name}/metrics --- #
    save_fld = os.path.join(args.root_path, "metrics")
    os.makedirs(save_fld, exist_ok=True)

    # ----------------------------------------------------------------------------
    write_sfid_to_file(save_fld, names_fake_image, sifid1, sifid2, sifid3, sifid4)

    # ----------------------------------------------------------------------------
    write_lpips_to_file(save_fld, lpips.cpu(), dist_to_tr.cpu(), dist_to_tr_byimage)
    
    print("--- Saved metrics at %s ---" % (save_fld))


