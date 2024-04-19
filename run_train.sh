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
nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset duke \
--format_tag tensor \
--train_format base \
--train_stage klstage \
--gpu 0 \
--saved_name fp32_l4vaeleakRelu-lre4_MG+RealNVP_64z_UindexC \
--vae_type cvae \
--flow_type RealNVP \
--recon_loss mse \
--reid_loss crossentropylabelsmooth \
--use_centroid \
--gaussian MultivariateNormal \
> train_clipreid_Uc_RealNVP.log 2>&1 & 





# ===========Baseline Stage 1 training================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage klstage \
# --gpu 0 \
# --saved_name baseline_fp32_l4vaeleakRelu_64z_norm \
# --vae_type cvae \
# --flow_type yuke_mlpflow \
# --recon_loss mse \
# --reid_loss crossentropylabelsmooth \
# --use_centroid \
# --only_x_input \
# --only_cvae_kl \
# --gaussian MultivariateNormal \
# > train_clipreid_baseline_64_norm.log 2>&1 & 


# # ===========Baseline Stage 2 training================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_stage2.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage reidstage \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_cvae_baseline/2024-04-12/baseline_fp32_l4vaeleakRelu_64z_yuke_mlpflow_mse \
# --gpu 0 \
# --saved_name baseline_fp32_l4vaeleakRelu_64z \
# --vae_type cvae \
# --flow_type yuke_mlpflow \
# --recon_loss mse \
# --reid_loss crossentropylabelsmooth \
# --use_centroid \
# --only_x_input \
# --only_cvae_kl \
# --gaussian MultivariateNormal \
# > train_clipreid_baseline_64_stage2.log 2>&1 & 

