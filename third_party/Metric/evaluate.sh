
# few_shot
# 1. 进行数据的预处理 
python third_party/Metric/fewshot_dataset_tool.py --source images/mvtec/wood/image --dest images/mvtec/wood/wood.zip --width 256 --height 256
python third_party/Metric/fewshot_dataset_tool.py --source outputs/256_256_nextnet/wood --dest outputs/256_256_nextnet/wood.zip --width 256 --height 256

# 2. 计算指标
python third_party/Metric/fewshot_calc_metrics.py --real_data images/mvtec/wood/wood.zip --gen_data outputs/256_256_nextnet/wood.zip --run_dir lightning_logs/wood/256_256_nextnet

# one_shot
python third_party/Metric/oneshot_evaluate.py --root_path images/mvtec/wood/image --real_path images/mvtec/wood/wood.zip --fake_path outputs/256_256_nextnet/wood.zip

