# python test.py --pretrained CLIPreid


# ===========Baseline testing================
nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset duke \
--format_tag tensor \
--train_format base \
--train_stage klstage \
--resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/transreid_cvae_baseline/2024-04-12/baseline_fp32_l4vaeleakRelu_64z_yuke_mlpflow_mse \
--gpu 0 \
--eval \
--saved_name baseline_fp32_l4vaeleakRelu_64z \
--use_centroid \
--only_x_input \
> test_64.log 2>&1 & 
