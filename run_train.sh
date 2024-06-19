
# # ===========Stage 1 traning================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage klstage \
# --gpu 0 \
# --saved_name fp32_SimpleVAE+2E_256+128z_1e3+3_60+120_SingleCls_ce_KLtotalZ \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > train_base_clipreid_SimpleVAE+2E_256+128z_1e3+3_60+120_SingleCls_ce_KLtotalZ.log 2>&1 & 



#  choices=['crossentropy', 'crossentropylabelsmooth']
# [novel, novel_train_from_scratch]
# [klstage, klNocls_stage, reidstage]

# for single Encoder
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae/2024-05-16/fp32_SimpleVAE_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ
# or
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae/2024-06-07/fp32_SimpleVAE_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ

# for two Encdoers
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae/2024-05-25/fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ \
# --use_two_encoder  # for model
# --use_NCE # for loss

# scratch with no resume
# ===========Novel Stage 1 traning================
nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset market1k \
--format_tag tensor \
--train_format novel \
--resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae/2024-06-07/fp32_SimpleVAE_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ \
--train_stage klstage \
--gpu 0 \
--saved_name TunedF_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_cls \
--vae_type SinpleVAE \
--recon_loss mse \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
> tune_TunedF_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_cls.log 2>&1 & 



# for single encoder
# /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae/novel/2024-06-13/fp32_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_nocls
 
# for two encoders
#  /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae/novel/2024-06-13/fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_nocls

# # ===========Novel ReID Stage traning================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset market1k \
# --format_tag tensor \
# --train_format novel \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae/novel/2024-06-13/fp32_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_nocls \
# --train_stage reidstage \
# --gpu 0 \
# --saved_name fp32_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_S2+cls+text512  \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --use_NCE \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > tune_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_S2+cls+text512.log 2>&1 & 




# ===========Baseline Stage 1 training================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_stage2.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage klstage \
# --gpu 0 \
# --saved_name baseline_fp32_l4vae+Relu+bn_64z_1e3 \
# --vae_type cvae \
# --flow_type yuke_mlpflow \
# --recon_loss mse \
# --reid_loss crossentropylabelsmooth \
# --use_centroid \
# --only_x_input \
# --gaussian MultivariateNormal \
# > train_clipreid_baseline_64_vae+Relu+bn_1e3.log 2>&1 & 


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



# ===========Novel Stage 1 testing================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_stage2.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset market1k \
# --format_tag tensor \
# --train_format novel \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae/novel/2024-05-17/fp32_SimpleVAE_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ \
# --train_stage klstage \
# --gpu 0 \
# --saved_name fp32_SimpleVAE_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ  \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# --eval \
# > test_SimpleVAE_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ.log 2>&1 &