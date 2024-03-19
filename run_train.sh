# nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --gpu 0 \
# --amp \
# --saved_name kl_claude_theta \
# --flow_type Planar \
# --recon_loss pearson \
# > train.log 2>&1 & 


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


python main.py --cfg ./configs/clipreid_cvae_kl.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset duke \
--format_tag tensor \
--train_format base \
--gpu 0 \
--amp \
--saved_name fp16_l4vaeRelu_yukeflow-noU-12z_noReID_clamp \
--vae_type cvae \
--flow_type yuke_mlpflow \
--recon_loss mse \
