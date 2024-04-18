# python keams.py --type gallery --pretrain_type AGWRes50 --n_clusters 129
# python keams.py --type query --pretrain_type AGWRes50 --n_clusters 129
# python keams.py --type train_all --pretrain_type AGWRes50 --n_clusters 129


# for Market-Sketch-1K dataset only
python keams_sketch.py --type train_rgb --pretrain_type CLIPreidNew --n_clusters 129
python keams_sketch.py --type train_sketch --pretrain_type CLIPreidNew --n_clusters 129
python keams_sketch.py --type query --pretrain_type CLIPreidNew --n_clusters 129

