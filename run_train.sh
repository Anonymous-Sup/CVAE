
# # ===========Stage 1 traning================
# nohup python -u main.py --cfg ./configs/base_duke/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage klstage \
# --gpu 0 \
# --saved_name SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_CLS+Trip+Center \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --use_two_encoder \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > train_base_clipreid_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_CLS+Trip+Center.log 2>&1 & 

# # ===========Base ReID Stage traning================
# nohup python -u main.py --cfg ./configs/base_duke/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage reidstage \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae_base/2024-06-20/fp32_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_noCLS \
# --gpu 0 \
# --saved_name SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_s2_[MLP_relu_nobn+CLS]+Trip+Center \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > train_base_clipreid_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_s2_[MLP_relu_nobn+CLS]+Trip+Center.log 2>&1 & 




# For the base model training with CE and Classifier
# for single Encoder
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae/2024-06-07/fp32_SimpleVAE_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ
# for two Encdoers
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae/2024-05-25/fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ \


# For the base model training with NCE and Text Emebediing
# for single Encoder
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae_base/2024-06-20/fp32_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_NCE+TEXT \
# for two Encdoers
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae_base/2024-06-20/fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_NCE+TEXT \


# For the base model training with CE and Text Emebediing
# for single Encoder
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae_base/2024-06-20/fp32_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_CE+TEXT \
# for two Encdoers
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae_base/2024-06-20/fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_CE+TEXT \


#  choices=['crossentropy', 'crossentropylabelsmooth']
# [novel, novel_train_from_scratch]
# [klstage, klNocls_stage, reidstage]


# --use_two_encoder  # for model
# --use_NCE # for loss

# scratch with no resume
# ===========Novel Stage 1 traning================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset market1k \
# --format_tag tensor \
# --train_format novel \
# --train_stage reidstage \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_tune/novel/2024-07-04/Tuned_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_noCLS \
# --gpu 0 \
# --saved_name tune_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_s2_ceText+ce+trip+center \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --use_two_encoder \
# --use_NCE \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > tune_Sketch_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_s2_ceText+ce+trip+center.log 2>&1 & 


# for cls 
# for single encoder
# /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_tune/novel/2024-06-19/TunedF_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_nocls 
 
# for two encoders
#  /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_tune/novel/2024-06-19/TunedF_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_nocls


# for CE with Text
# for single
# /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_tune/novel/2024-06-21/BaseCET_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_noCls
# for two encoders
# /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_tune/novel/2024-06-21/BaseCET_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_noCls

# for NCE
# for single
# /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_tune/novel/2024-06-22/BaseNCE_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_nocls
# for two encoders
# /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_tune/novel/2024-06-22/BaseNCE_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_nocls

# # ===========Novel ReID Stage traning================
# nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset market1k \
# --format_tag tensor \
# --train_format novel \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_tune/novel/2024-07-04/Tuned_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_noCLS \
# --train_stage reidstage \
# --gpu 0 \
# --saved_name 2ndstage_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_s2_MLP1280+CLS_ce+trip+center \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --use_two_encoder \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > 2ndstage_sketch_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_s2_MLP1280+CLS_ce+trip+center.log 2>&1 & 


# # ===========Novel SYSUMM01 KL Stage traning================
# nohup python -u main.py --cfg ./configs/tune_sysu/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset sysu_mm01 \
# --format_tag tensor \
# --train_format novel \
# --resume  /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae_base/2024-06-20/fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_noCLS \
# --train_stage klNocls_stage \
# --gpu 0 \
# --saved_name tune_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_nocls \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --use_two_encoder \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > tune_sysumm01_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_nocls.log 2>&1 & 


# # # ===========Novel SYSUMM01 ReID Stage traning================
nohup python -u main.py --cfg ./configs/tune_sysu/clipreid_cvae_kl.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset sysu_mm01 \
--format_tag tensor \
--train_format novel \
--resume  /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/sysu_mm01/clipreid_simplevae_tune/novel/2024-07-12/tune_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_nocls \
--train_stage reidstage \
--gpu 0 \
--saved_name 2ndstage_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_s2_L1280+CLS_ce+trip+center \
--vae_type SinpleVAE \
--recon_loss mse \
--use_two_encoder \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
> 2ndstage_sysumm01_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_s2_L1280+CLS_ce+trip+center.log 2>&1 & 



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