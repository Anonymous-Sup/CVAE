
cd /home/zhengwei/github/CVAE


#  choices=['crossentropy', 'crossentropylabelsmooth']
# [novel, novel_train_from_scratch]
# [klstage, klNocls_stage, reidstage]


# --use_two_encoder  # for model
# --use_NCE # for loss

# CUDA_VISIBLE_DEVICES=1
# # # ===========Stage 1 traning================
# nohup python -u main.py --cfg ./configs/tuberlin_categray/clipreid_cvae_kl_novel_true5wKs.yaml \
# --root /home/zhengwei/my_data/datasets \
# --output /data/zhengwei/CVAE/outputs \
# --dataset tuberlin \
# --format_tag tensor \
# --train_format novel \
# --resume /data/zhengwei/CVAE/outputs/tuberlin/clipreid_simplevae_base/2024-09-18/baseTuberlin_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_noCLS/2nd_stage/2024-09-18/baseTUBerlin_SimpleVAE_128+64z_1e4_60+120_KLtotalZ_L1280+bnTuneVersion+CLS_Trip+Center+Ce \
# --train_stage klNocls_stage \
# --gpu 1 \
# --saved_name novelTUBerlin_SimpleVAE_128+64z_1e3_60+120_KLtotalZ+lessC_true5w5s_noCLS_forLinear \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > nohup_logs/tune_novelTUBerlin_SimpleVAE_128+64z_1e3_60+120_KLtotalZ+lessC_true5w5s_noCLS_forLinear.log 2>&1 &

# # # ===========Stage 1 traning================
# nohup python -u main.py --cfg ./configs/tuberlin_categray/clipreid_cvae_kl_novel_true5wKs.yaml \
# --root /home/zhengwei/my_data/datasets \
# --output /data/zhengwei/CVAE/outputs \
# --dataset tuberlin \
# --format_tag tensor \
# --train_format novel \
# --resume /data/zhengwei/CVAE/outputs/tuberlin/clipreid_simplevae_base/2024-09-18/baseTuberlin_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ_noCLS/2nd_stage/2024-09-18/baseTUBerlin_SimpleVAE+2E_128+64z_1e4_60+120_KLtotalZ_L1280+bnTuneVersion+CLS_Trip+Center+Ce \
# --train_stage klNocls_stage \
# --gpu 1 \
# --saved_name novelTUBerlin_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ+lessC_true5w5s_noCLS_forLinear \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --use_two_encoder \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# > nohup_logs/tune_novelTUBerlin_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ+lessC_true5w5s_noCLS_forLinear.log 2>&1 &

CUDA_VISIBLE_DEVICES=1
# ========== ReID+CLS Stage traning================
nohup python -u main.py --cfg ./configs/tuberlin_categray/clipreid_cvae_kl_novel.yaml \
--root /home/zhengwei/my_data/datasets \
--output /data/zhengwei/CVAE/outputs \
--dataset tuberlin \
--format_tag tensor \
--train_format novel \
--train_stage reid+cls_stage \
--resume /data/zhengwei/CVAE/outputs/tuberlin/clipreid_simplevae_base/novel/2024-09-19/novelTUBerlin_SimpleVAE_128+64z_1e3_60+120_KLtotalZ+lessC_all5w5s_noCLS_forLinear \
--gpu 1 \
--saved_name novelTUBerlin_SimpleVAE_128+64z_1e3_60_KLtotalZ+lessC_all5w5s_L1280+bnTuneVersion+distCLS_Trip+Center+Ce \
--vae_type SinpleVAE \
--recon_loss mse \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
> nohup_logs/tune_2nd_novelTUBerlin_SimpleVAE_128+64z_1e3_60_KLtotalZ+lessC_all5w5s_L1280+bnTuneVersion+distCLS_Trip+Center+Ce.log 2>&1 &


nohup python -u main.py --cfg ./configs/tuberlin_categray/clipreid_cvae_kl_novel.yaml \
--root /home/zhengwei/my_data/datasets \
--output /data/zhengwei/CVAE/outputs \
--dataset tuberlin \
--format_tag tensor \
--train_format novel \
--train_stage reid+cls_stage \
--resume  /data/zhengwei/CVAE/outputs/tuberlin/clipreid_simplevae_base/novel/2024-09-19/novelTUBerlin_SimpleVAE+2E_128+64z_1e3_60+120_KLtotalZ+lessC_all5w5s_noCLS_forLinear \
--gpu 1 \
--saved_name novelTUBerlin_SimpleVAE+2E_128+64z_1e3_60_KLtotalZ+lessC_all5w5s_L1280+bnTuneVersion+distCLS_Trip+Center+Ce \
--vae_type SinpleVAE \
--recon_loss mse \
--use_two_encoder \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
> nohup_logs/tune_2nd_novelTUBerlin_SimpleVAE+2E_128+64z_1e3_60_KLtotalZ+lessC_all5w5s_L1280+bnTuneVersion+distCLS_Trip+Center+Ce.log 2>&1 &

