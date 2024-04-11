# nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage reidstage \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_cvae_ce_trip_cesmooth/2024-03-27/fp32_l4vaeleakRelu_yukeflowkl-MultivG-wUlkrelu-UindexC-12z_TripnoReID_clamp_yuke_mlpflow_mse \
# --gpu 0 \
# --saved_name fp32_l4vaeleakRelu_yukeflowkl-MultivG-wUlkrelu-UindexC-12z_TripnoReID_clamp \
# --vae_type cvae \
# --flow_type yuke_mlpflow \
# --recon_loss mse \
# --reid_loss crossentropy \
# --only_cvae_kl \
# --use_centroid \
# --gaussian MultivariateNormal \
# > train_reid_b+recon+trip_n_ce.log 2>&1 & 


#  choices=['crossentropy', 'crossentropylabelsmooth']
#  choices=['Normal', 'MultivariateNormal']
# --use_centroid \
# --only_cvae_kl \
# --only_x_input \


# # ===========Stage 1 traning================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage klstage \
# --gpu 0 \
# --saved_name fp32_l4vaeleakRelu-lre4_yukeflowkl_J-MG-noU-UindexC-12z_TrinoReID_clamp \
# --vae_type cvae \
# --flow_type yuke_mlpflow \
# --recon_loss mse \
# --reid_loss crossentropy \
# --use_centroid \
# --only_cvae_kl \
# --gaussian MultivariateNormal \
# > train_MG_J_Tri.log 2>&1 & 


# # ===========Baseline Stage 1 training================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage klstage \
# --gpu 0 \
# --saved_name baseline_fp32_l4vaeleakRelu-lre4_64z \
# --vae_type cvae \
# --flow_type yuke_mlpflow \
# --recon_loss mse \
# --reid_loss crossentropylabelsmooth \
# --use_centroid \
# --only_x_input \
# --only_cvae_kl \
# --gaussian MultivariateNormal \
# > train_baseline_64.log 2>&1 & 


# ===========Baseline Stage 2 training================
nohup python -u main.py --cfg ./configs/clipreid_cvae_stage2.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset duke \
--format_tag tensor \
--train_format base \
--train_stage reidstage \
--resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_cvae_baseline/2024-04-04/baseline_fp32_l4vaeleakRelu-lre4_64z_yuke_mlpflow_mse \
--gpu 0 \
--saved_name baseline_fp32_l4vaeleakRelu-lre4_64z_05w_dp02_bn \
--vae_type cvae \
--flow_type yuke_mlpflow \
--recon_loss mse \
--reid_loss crossentropylabelsmooth \
--use_centroid \
--only_x_input \
--only_cvae_kl \
--gaussian MultivariateNormal \
> train_baseline_stage2_64_05w_dp02_bn.log 2>&1 & 




# python main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --gpu 0 \
# --amp \
# --saved_name l4vaeRelu_yukeflow-noU-12z_noReID_clamp \
# --vae_type cvae \
# --flow_type yuke_mlpflow \
# --recon_loss mse \


# --only_cvae_kl \
# --only_x_input
# --use_centroid


# python main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --gpu 0 \
# --saved_name fp32_l4vaeleakRelu_yukeflowkl-NormalG-wU-Uindex-12z_noReID_clamp \
# --vae_type cvae \
# --flow_type yuke_mlpflow \
# --recon_loss mse \
# --only_cvae_kl 