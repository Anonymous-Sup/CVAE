cd /home/zhengwei/github/CVAE
# python test.py --pretrained CLIPreid

# /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_tune/novel/2024-07-14/2ndstage_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_s2_L1280+CLS_ce+trip+center
# /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_tune_wReID/novel/2024-08-03/sketch_3rdstage_SimpleVAE_128+64z_1e3_60+120_KLtotalZ_L1280_NewE+Znew_center+trip_disCLS+ce

# [market1k, duke, sysu_mm01]
# ===============For Regular testing================
nohup python -u main.py --cfg ./configs/sketchy_categray/clipreid_cvae_kl_novel.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset sketchy \
--format_tag tensor \
--train_format novel \
--train_stage reid+cls_stage \
--resume  /data/zhengwei/CVAE/outputs/sketchy/clipreid_simplevae_base/novel/2024-09-18/novelSketchy_SimpleVAE+2E_128+64z_1e4_60+120_KLtotalZ_all5w5s_L1280+bnTuneVersion+CLS_Trip+Center+Ce \
--gpu 0 \
--vae_type SinpleVAE \
--recon_loss mse \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
--eval \
--use_two_encoder \
--saved_name test \
> nohup_logs/test_sketchy_2E_all5w5s.log 2>&1 & 



# # The following are the commands for testing the models with polts
# # ===========Baseline testing================
# nohup python -u main.py --cfg ./configs/ablation_studies/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset duke \
# --format_tag tensor \
# --train_format base \
# --train_stage klstage \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae/2024-05-25/fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ \
# --gpu 0 \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# --eval \
# --use_two_encoder \
# --saved_name fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ \
# > test_duke_plots_2E.log 2>&1 & 

# # ===============For Novel testing================
# nohup python -u main.py --cfg ./configs/ablation_studies/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset market1k \
# --format_tag tensor \
# --train_format base \
# --train_stage klstage \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_ablations/novel/2024-07-03/ablation_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_cls_alldata \
# --gpu 0 \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# --eval \
# --saved_name ablation_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_cls_alldata \
# > test_novel_plots_E.log 2>&1 & 


# # ===============For Ablation testing================
# # /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae/novel/2024-06-01/2sketch_fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ_scratch_onlyRGB
# # /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae/novel/2024-06-01/2sketch_fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ_scratch_onlysketch

# nohup python -u main.py --cfg ./configs/ablation_studies/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset market1k \
# --format_tag tensor \
# --train_format novel_train_from_scratch \
# --train_stage klstage \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_ablations/novel/2024-07-09/train_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_cls_onlyRGB+Gaussian \
# --gpu 0 \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# --eval \
# --saved_name train_SimpleVAE_128+64z_1e3+3_60+120_KLtotalZ_cls_onlyRGB+Gaussian \
# > test_Gaussian.log 2>&1 & 


# nohup python -u main.py --cfg ./configs/ablation_studies/clipreid_cvae_kl.yaml \
# --root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
# --dataset market1k \
# --format_tag tensor \
# --train_format base \
# --train_stage klstage \
# --resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae_ablations/novel/2024-07-03/ablation_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_cls_OnlySketch \
# --gpu 0 \
# --vae_type SinpleVAE \
# --recon_loss mse \
# --reid_loss crossentropy \
# --gaussian MultivariateNormal \
# --eval \
# --use_two_encoder \
# --saved_name ablation_SimpleVAE+2E_128+64z_1e3+3_60+120_KLtotalZ_cls_OnlySketch \
# > test_sketchs_plots_2E.log 2>&1 & 