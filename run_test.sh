# python test.py --pretrained CLIPreid


# ===========Baseline Stage 2 training================
nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset duke \
--format_tag tensor \
--train_format base \
--train_stage klstage \
--resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_cvae_ce_trip_cesmooth/2024-04-01/baseline_fp32_l4vaeleakRelu-lre4_12z_yuke_mlpflow_mse \
--gpu 0 \
--eval \
--saved_name baseline_fp32_l4vaeleakRelu-lre4_12z \
--use_centroid \
--only_x_input \
> test_12.log 2>&1 & 
