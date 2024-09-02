
cd /home/zhengwei/github/CVAE


# # # ===========Stage 1 traning================
# nohup python -u main.py --cfg ./configs/base_msmt17/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/my_data/datasets \
# --output /data/zhengwei/CVAE/outputs \
# --dataset msmt17 \
# --format_tag tensor \
# --train_format base \
# --train_stage klNocls_stage \
# --gpu 2 \
# --saved_name baseMSMT17_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_noCLS \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > nohup_logs/train_baseMSMT17_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_noCLS.log 2>&1 &


CUDA_VISIBLE_DEVICES=2
# ===========Base ReID+CLS Stage traning================
nohup python -u main.py --cfg ./configs/base_msmt17/clipreid_cvae_kl.yaml \
--root /home/zhengwei/my_data/datasets \
--output /data/zhengwei/CVAE/outputs \
--dataset msmt17 \
--format_tag tensor \
--train_format base \
--train_stage reid+cls_stage \
--resume  /data/zhengwei/CVAE/outputs/msmt17/clipreid_simplevae_base/2024-08-31/baseMSMT17_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_noCLS \
--gpu 2 \
--saved_name baseMSMT17_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_L1280+CLS_Trip+Center+Ce \
--vae_type SinpleVAE \
--recon_loss mse \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
> nohup_logs/train_2nd_baseMSMT17_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_L1280+CLS_Trip+Center+Ce.log 2>&1 &


# # ===========Base CLS Stage traning================
# nohup python -u main.py --cfg ./configs/base_msmt17/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/my_data/datasets \
# --output /data/zhengwei/CVAE/outputs \
# --dataset msmt17 \
# --format_tag tensor \
# --train_format base \
# --train_stage CLSstage \
# --resume  /data/zhengwei/CVAE/outputs/msmt17/clipreid_simplevae_base/2024-08-31/baseMSMT17_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_noCLS \
# --gpu 1 \
# --saved_name baseMSMT17_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_L1280_disCLS+ce \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --use_two_encoder \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > nohup_logs/train_2nd_baseMSMT17_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_L1280_disCLS+ce.log 2>&1 & 


# ===========Base ReID Stage traning================
# nohup python -u main.py --cfg ./configs/base_duke/clipreid_cvae_stage2_tripwrt.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage reidstage \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae_base/2024-06-20/fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_noCLS \
# --gpu 0 \
# --saved_name 2nd_SimpleVAE+2E_128+64z_warm5+1e4+d1e4_10+30_KLtotalZ_L1280_ReIDZnew+center+tripwrt \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --use_two_encoder \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > 2nd_train_base_clipreid_SimpleVAE+2E_128+64z_warm5+1e4+d1e4_10+30_KLtotalZ_L1280_ReIDZnew+center+tripwrt.log 2>&1 & 


# /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae_base/2024-06-20/fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_noCLS/reid_crossentropy/2024-07-30/2nd_SimpleVAE+2E_128+64z_warm5+1e3+d1e4_10+30_KLtotalZ_L1280_ReIDZnew+center+tripwrt
# ===========Base CLS Stage traning================
# nohup python -u main.py --cfg ./configs/base_duke/clipreid_cvae_stage2.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage reidstage \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae_base/2024-06-20/fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_noCLS/reid_crossentropy/2024-07-30/2nd_SimpleVAE+2E_128+64z_warm5+1e3+d1e4_10+30_KLtotalZ_L1280_ReIDZnew+center+tripwrt \
# --gpu 0 \
# --saved_name 3rd_SimpleVAE+2E_128+64z_warm5+1e3_60+120_KLtotalZ_L1280_ReIDZnew+center+tripwrt_distCLS+ce \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --use_two_encoder \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > 3rd_train_base_clipreid_SimpleVAE+2E_128+64z_warm5+1e3_60+120_KLtotalZ_L1280_ReIDZnew+center+tripwrt_distCLS+ce.log 2>&1 & 



#  choices=['crossentropy', 'crossentropylabelsmooth']
# [novel, novel_train_from_scratch]
# [klstage, klNocls_stage, reidstage]


# --use_two_encoder  # for model
# --use_NCE # for loss

