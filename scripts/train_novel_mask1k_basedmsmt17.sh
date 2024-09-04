
cd /home/zhengwei/github/CVAE


#  choices=['crossentropy', 'crossentropylabelsmooth']
# [novel, novel_train_from_scratch]
# [klstage, klNocls_stage, reidstage]


# --use_two_encoder  # for model
# --use_NCE # for loss

# CUDA_VISIBLE_DEVICES=2
# # # ===========Stage 1 traning================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/my_data/datasets \
# --output /data/zhengwei/CVAE/outputs \
# --dataset market1k \
# --format_tag tensor \
# --train_format novel \
# --resume /data/zhengwei/CVAE/outputs/msmt17/clipreid_simplevae_base/2024-08-31/baseMSMT17_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_noCLS/2nd_stage/2024-09-02/baseMSMT17_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_L1280+distCLS_Trip+Center+Ce \
# --train_stage klNocls_stage \
# --gpu 2 \
# --saved_name novelMask1K_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_noCLS \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > nohup_logs/tune_novelMask1K_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_noCLS.log 2>&1 &

# # # ===========Stage 1 traning================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/my_data/datasets \
# --output /data/zhengwei/CVAE/outputs \
# --dataset market1k \
# --format_tag tensor \
# --train_format novel \
# --resume /data/zhengwei/CVAE/outputs/msmt17/clipreid_simplevae_base/2024-08-31/baseMSMT17_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_noCLS/2nd_stage/2024-09-02/baseMSMT17_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_L1280+distCLS_Trip+Center+Ce \
# --train_stage klNocls_stage \
# --gpu 2 \
# --saved_name novelMask1K_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_noCLS \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --use_two_encoder \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > nohup_logs/tune_novelMask1K_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_noCLS.log 2>&1 &

CUDA_VISIBLE_DEVICES=2
# ===========Base ReID+CLS Stage traning================
nohup python -u main.py --cfg ./configs/clipreid_cvae_stage2.yaml \
--root /home/zhengwei/my_data/datasets \
--output /data/zhengwei/CVAE/outputs \
--dataset market1k \
--format_tag tensor \
--train_format novel \
--train_stage reid+cls_stage \
--resume  /data/zhengwei/CVAE/outputs/market1k/clipreid_tuned_person/novel/2024-09-04/novelMask1K_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_noCLS \
--gpu 2 \
--saved_name novelMask1K_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_L1280-D-LCLS_Trip+Center+Ce \
--vae_type SinpleVAE \
--recon_loss mse \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
> nohup_logs/tune_2nd_novelMask1K_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_L1280-D-LCLS_Trip+Center+Ce.log 2>&1 &


nohup python -u main.py --cfg ./configs/clipreid_cvae_stage2.yaml \
--root /home/zhengwei/my_data/datasets \
--output /data/zhengwei/CVAE/outputs \
--dataset market1k \
--format_tag tensor \
--train_format novel \
--train_stage reid+cls_stage \
--resume  /data/zhengwei/CVAE/outputs/market1k/clipreid_tuned_person/novel/2024-09-04/novelMask1K_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_noCLS \
--gpu 2 \
--saved_name novelMask1K_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_L1280-D-LCLS_Trip+Center+Ce \
--vae_type SinpleVAE \
--recon_loss mse \
--use_two_encoder \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
> nohup_logs/tune_2nd_novelMask1K_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_L1280-D-LCLS_Trip+Center+Ce.log 2>&1 &

