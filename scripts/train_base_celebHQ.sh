
cd /home/zhengwei/github/CVAE

# CUDA_VISIBLE_DEVICES=1
# # # ===========Stage 1 traning================
# nohup python -u main.py --cfg ./configs/base_face/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/my_data/datasets \
# --output /data/zhengwei/CVAE/outputs \
# --dataset celebHQ \
# --format_tag tensor \
# --train_format base \
# --train_stage klNocls_stage \
# --gpu 1 \
# --saved_name baseCelebHQ_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_noCLS \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > nohup_logs/train_baseCelebHQ_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_noCLS.log 2>&1 &

# # # ===========Stage 1 traning================
# nohup python -u main.py --cfg ./configs/base_face/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/my_data/datasets \
# --output /data/zhengwei/CVAE/outputs \
# --dataset celebHQ \
# --format_tag tensor \
# --train_format base \
# --train_stage klNocls_stage \
# --gpu 1 \
# --saved_name baseCelebHQ_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_noCLS \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --use_two_encoder \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > nohup_logs/train_baseCelebHQ_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_noCLS.log 2>&1 &


CUDA_VISIBLE_DEVICES=3
# ===========Base ReID+CLS Stage traning================
nohup python -u main.py --cfg ./configs/base_face/clipreid_cvae_kl.yaml \
--root /home/zhengwei/my_data/datasets \
--output /data/zhengwei/CVAE/outputs \
--dataset celebHQ \
--format_tag tensor \
--train_format base \
--train_stage reid+cls_stage \
--resume /data/zhengwei/CVAE/outputs/celebHQ/clipreid_simplevae_base/2024-09-22/baseCelebHQ_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_noCLS \
--gpu 3 \
--saved_name baseCelebHQ_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_L1280+bnTuneVersion+CLS_Trip+Center+Ce \
--vae_type SinpleVAE \
--recon_loss mse \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
> nohup_logs/train_2nd_baseCelebHQ_SimpleVAE_128+64z_1e4_60+120_KLtotalZ_L1280+bnTuneVersion+CLS_Trip+Center+Ce.log 2>&1 &

# ===========Base ReID+CLS Stage traning================
nohup python -u main.py --cfg ./configs/base_face/clipreid_cvae_kl.yaml \
--root /home/zhengwei/my_data/datasets \
--output /data/zhengwei/CVAE/outputs \
--dataset celebHQ \
--format_tag tensor \
--train_format base \
--train_stage reid+cls_stage \
--resume /data/zhengwei/CVAE/outputs/celebHQ/clipreid_simplevae_base/2024-09-22/baseCelebHQ_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_noCLS \
--gpu 3 \
--saved_name baseCelebHQ_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_L1280+bnTuneVersion+CLS_Trip+Center+Ce \
--vae_type SinpleVAE \
--recon_loss mse \
--use_two_encoder \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
> nohup_logs/train_2nd_baseCelebHQ_SimpleVAE+2E_128+64z_1e4_60+120_KLtotalZ_L1280+bnTuneVersion+CLS_Trip+Center+Ce.log 2>&1 &

