nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset duke \
--format_tag tensor \
--train_format base \
--gpu 0 \
--saved_name fp32_l4vaeleakRelu_yukeflowkl-NormalG-wU-Uindex-12z_noReID_clamp \
--vae_type cvae \
--flow_type yuke_mlpflow \
--recon_loss mse \
--only_cvae_kl \
--use_centroid \
> train_Ucentro.log 2>&1 & 


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