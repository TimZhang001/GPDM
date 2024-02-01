import os
import pandas as pd
import numpy as np
import argparse

def parser_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,  default='/home/zhangss/PHDPaper/05_GPDM/outputs/')
    args = parser.parse_args()

    return args

def main(root_path):

    # 定义数据集中的所有类型
    mvtec_path="/raid/zhangss/dataset/ADetection/mvtecAD/"
    categories=('carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper')

    # 结果字典 {category:{subcategory:{lpips:xx, sfid:xx,yy}}}
    result_dict = {}

    # 循环处理每个类型
    for category in categories:
        
        # 获取类别下的子类别
        subcategories = os.listdir(os.path.join(mvtec_path, category, 'test'))
        result_dict[category] = {}

        # 循环处理每个子类别
        for subcategory in subcategories:
            if subcategory == 'good':
                continue
            
            # 目录路径
            sub_path = os.path.join(root_path, category, subcategory)
            result_dict[category][subcategory] = {}

            # lpips文件 -----------------------------------------------------
            lpips_file = os.path.join(sub_path, 'metricslpips.csv')
            with open(lpips_file, 'r') as f:
                lines = f.readlines()
                line  = lines[0]
                lpips = line.split(',')[1]

                # 将结果转化为数字存入字典
                result_dict[category][subcategory]['lpips'] = float(lpips)
            
            # sfid文件 -----------------------------------------------------
            sfid_file = os.path.join(sub_path, 'metricsSIFID1.csv')
            with open(sfid_file, 'r') as f:
                lines    = f.readlines()
                line     = lines[0]
                sfid_val = line.split(',')[1]

                # 将结果转化为数字存入字典
                result_dict[category][subcategory]['sfid_val'] = float(sfid_val)

            # fid文件 -----------------------------------------------------
            fid_file = os.path.join(sub_path, 'metricsFID1.csv')
            with open(fid_file, 'r') as f:
                lines    = f.readlines()
                line     = lines[0]
                fid_val  = line.split(',')[1]

                # 将结果转化为数字存入字典
                result_dict[category][subcategory]['fid_val'] = float(fid_val)

            # pixel diversity文件 -----------------------------------------------------
            pd_file = os.path.join(sub_path, 'metricspixel_div.csv')
            with open(pd_file, 'r') as f:
                lines    = f.readlines()
                line     = lines[0]
                pd_val  = line.split(',')[1]

                # 将结果转化为数字存入字典
                result_dict[category][subcategory]['pixel_div_val'] = float(pd_val)

            # no-reference文件 -----------------------------------------------------
            nr_file = os.path.join(sub_path, 'metricsno_reference.csv')
            with open(nr_file, 'r') as f:
                lines     = f.readlines()
                line      = lines[0]
                niqe_val  = line.split(',')[1]

                line      = lines[1]
                musiq_val = line.split(',')[1]
                
                line      = lines[2]
                is_val    = line.split(',')[1]

                # 将结果转化为数字存入字典
                result_dict[category][subcategory]['niqe_val']  = float(niqe_val)
                result_dict[category][subcategory]['musiq_val'] = float(musiq_val)
                result_dict[category][subcategory]['is_val']    = float(is_val)


    # -----------------------------------------
    # 将结果result_dict输出为result.txt文件格式如下：
    #                      sfid          lpips
    # carpet   carpet_01   0.123±0.123   0.123
    #          carpet_02   0.123±0.123   0.123
    #          .......     ...........   ......
    # bottle   bottle_01   0.123±0.123   0.123
    with open(os.path.join(root_path, 'single_result.txt'), 'w') as f:
        f.write('{:<10} {:<20} {:<16} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n'.format('', '', 'sfid', 'fid', 'niqe', 'musiq', 'is', 'pixel_div', 'lpips'))
        for category in categories:
            subcategories = list(result_dict[category].keys())
            for i, subcategory in enumerate(subcategories):
                sfid_val  = result_dict[category][subcategory]['sfid_val']
                fid_val   = result_dict[category][subcategory]['fid_val']
        
                niqe_val  = result_dict[category][subcategory]['niqe_val']
                musiq_val = result_dict[category][subcategory]['musiq_val']
                is_val    = result_dict[category][subcategory]['is_val']
                
                pixel_val = result_dict[category][subcategory]['pixel_div_val']
                lpips     = result_dict[category][subcategory]['lpips']

                if i == 0:
                    f.write('{:<10} {:<20} {:<.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}\n'.format(category, subcategory, sfid_val, fid_val, niqe_val, musiq_val, is_val, pixel_val, lpips))
                else:
                    f.write('{:<10} {:<20} {:<.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}\n'.format('',       subcategory, sfid_val, fid_val, niqe_val, musiq_val, is_val, pixel_val, lpips))


    # -----------------------------------------
    # 将结果result_dict输出为result.txt文件格式如下：
    # carpet     carpet_01        carpet_02       carpet_03     .......
    #           sfid    lpips   sfid    lpips    sfid    lpips
    #          0.1234   0.1234  0.1234  0.1234  0.1234   0.1234
    # ........
    with open(os.path.join(root_path, 'single_result_csv.txt'), 'w') as f:
        #f.write('{:<20} {:<25} {:<15} {:<10}\n'.format('', '', 'sfid', 'lpips'))
        for category in categories:
            subcategories = list(result_dict[category].keys())
            f.write('{:<15}'.format(category))
            for subcategory in subcategories:
                f.write('{:<25},'.format(subcategory))
            f.write('\n')

            f.write('{:<15}'.format(''))
            for subcategory in subcategories:
                f.write('{:<.4f}, {:<8.4f}, {:<8.4f}, {:<8.4f}, {:<8.4f}, {:<8.4f},{:<8.4f},'.format(
                    result_dict[category][subcategory]['sfid_val'],
                    result_dict[category][subcategory]['fid_val'],
                    result_dict[category][subcategory]['niqe_val'],
                    result_dict[category][subcategory]['musiq_val'],
                    result_dict[category][subcategory]['is_val'],
                    result_dict[category][subcategory]['pixel_div_val'],    
                    result_dict[category][subcategory]['lpips']
                ))
            f.write('\n')
            #f.write('-'*100+'\n\n')



if __name__ == "__main__":
    args = parser_param()
    print("--- Computing metrics for job %s  ---" %(args.root_path))

    main(root_path=args.root_path)