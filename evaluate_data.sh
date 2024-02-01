#!/bin/bash

# 定义数据集中的所有类型
mvtec_path="/raid/zhangss/dataset/ADetection/mvtecAD/"
categories='carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper'

# 循环处理每个类型
for category in $categories; do
    
    base_paths="/home/zhangss/PHDPaper/05_GPDM/outputs/${category}"

    # 寻找cur_paths包含的所有子目录
    sub_dirs=$(ls "${base_paths}")

    # 循环处理每个子目录
    for sub_dir in ${sub_dirs}; do

        # print sub_dir
        echo "${base_paths}/${sub_dir}"
        
        # 真实样本的路径
        src_paths="${mvtec_path}/${category}/test/${sub_dir}/000.png"

        # 设置其他路径
        root_paths="${base_paths}/${sub_dir}/" 
        real_paths="${base_paths}/${sub_dir}/sample_true" 
        fake_paths="${base_paths}/${sub_dir}/sample_eval"

        # 如果real_paths不存在，先创建
        if [ ! -d "${real_paths}" ]; then
            mkdir -p "${real_paths}"
        fi

        # 将真实样本复制到real_paths
        cp "${src_paths}" "${real_paths}"

        # 如果fake_paths不存在，先创建
        if [ ! -d "${fake_paths}" ]; then
            mkdir -p "${fake_paths}"
        fi

        # 将root_paths下的所有.png文件移动到fake_paths
        #mv "${root_paths}"/*.png "${fake_paths}"     
        
        # 调用处理脚本进行评估
        python third_party/Metric/oneshot_evaluate.py --root_path "${root_paths}" --real_path "${real_paths}" --fake_path "${fake_paths}" --gpu_id 6

    done

    echo "----------------------------------------------"

done

#--------------进行结果的统计和输出----------------
root_path="/home/zhangss/PHDPaper/05_GPDM/outputs"
python third_party/Metric/stat_evaluate_data.py --root_path "${root_path}"