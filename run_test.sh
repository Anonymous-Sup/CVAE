# python test.py --pretrained CLIPreid


# ===========Baseline testing================
nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset duke \
--format_tag tensor \
--train_format base \
--train_stage klstage \
--resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/duke/clipreid_simplevae/2024-05-25/fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ \
--gpu 0 \
--vae_type SinpleVAE \
--recon_loss mse \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
--eval \
--saved_name fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ \
> test_duke_plots.log 2>&1 & 

# ===============For Novel testing================
nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset market1k \
--format_tag tensor \
--train_format base \
--train_stage klstage \
--resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae/novel/2024-05-31/2sketch_fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ_scratch \
--gpu 0 \
--vae_type SinpleVAE \
--recon_loss mse \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
--eval \
--saved_name 2sketch_fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ_scratch \
> test_novel_plots.log 2>&1 & 


# ===============For Ablation testing================
# /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae/novel/2024-06-01/2sketch_fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ_scratch_onlyRGB
# /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae/novel/2024-06-01/2sketch_fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ_scratch_onlysketch

nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset market1k \
--format_tag tensor \
--train_format base \
--train_stage klstage \
--resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae/novel/2024-06-01/2sketch_fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ_scratch_onlyRGB \
--gpu 0 \
--vae_type SinpleVAE \
--recon_loss mse \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
--eval \
--saved_name 2sketch_fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ_scratch_onlyRGB \
> test_rgb_plots.log 2>&1 & 


nohup python -u main.py --cfg ./configs/clipreid_cvae_kl.yaml \
--root /home/zhengwei/Desktop/Zhengwei/Projects/datasets \
--dataset market1k \
--format_tag tensor \
--train_format base \
--train_stage klstage \
--resume /home/zhengwei/Desktop/Zhengwei/Projects/CVAE/outputs/market1k/clipreid_simplevae/novel/2024-06-01/2sketch_fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ_scratch_onlysketch \
--gpu 0 \
--vae_type SinpleVAE \
--recon_loss mse \
--reid_loss crossentropy \
--gaussian MultivariateNormal \
--eval \
--saved_name 2sketch_fp32_SimpleVAE+2E_128+64z_1e3+3_60+120_SingleCls_ce_KLtotalZ_scratch_onlysketch \
> test_sketchs_plots.log 2>&1 & 