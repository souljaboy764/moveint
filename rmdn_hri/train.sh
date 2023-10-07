# CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/mdn_3/trial0 --src data/data_raw.npz --model MixtureDensityNet --num-components 3 --std-reg 1e-6 --hidden-sizes 25 25 --loss nll >> logs/mdn_3/trial0/log.txt &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/mdn_3/trial1 --src data/data_raw.npz --model MixtureDensityNet --num-components 3 --std-reg 1e-6 --hidden-sizes 25 25 --loss nll >> logs/mdn_3/trial1/log.txt &
# CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/mdn_3/trial2 --src data/data_raw.npz --model MixtureDensityNet --num-components 3 --std-reg 1e-6 --hidden-sizes 25 25 --loss nll >> logs/mdn_3/trial2/log.txt &
# CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --results logs/mdn_3/trial3 --src data/data_raw.npz --model MixtureDensityNet --num-components 3 --std-reg 1e-6 --hidden-sizes 25 25 --loss nll >> logs/mdn_3/trial3/log.txt &


CUDA_VISIBLE_DEVICES=0 nohup python3 rmdn_hri/train_rmdvae.py --results logs/rmdvae_diagcov_expdistcontrastive/trial0/ --epochs 400 >> logs/rmdvae_diagcov_expdistcontrastive/trial0/log.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 rmdn_hri/train_rmdvae.py --results logs/rmdvae_diagcov_expdistcontrastive/trial1/ --epochs 400 >> logs/rmdvae_diagcov_expdistcontrastive/trial1/log.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 rmdn_hri/train_rmdvae.py --results logs/rmdvae_diagcov_expdistcontrastive/trial2/ --epochs 400 >> logs/rmdvae_diagcov_expdistcontrastive/trial2/log.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 rmdn_hri/train_rmdvae.py --results logs/rmdvae_diagcov_expdistcontrastive/trial3/ --epochs 400 >> logs/rmdvae_diagcov_expdistcontrastive/trial3/log.txt &