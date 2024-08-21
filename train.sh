# Training configs

CUDA_VISIBLE_DEVICES=0 nohup python3 train_moveint.py --results logs/corrected/nuisi_pepper/moveint_combined_3comps/trial0/ --ckpt logs/corrected/bp_pepper/moveint_combined_3comps/trial3/models/final_399.pth --epochs 400 --num-components 3 --dataset NuiSIPepper --hidden-sizes 40 20 --latent-dim 5 >> logs/nuisi_pepper/moveint_combined_3comps/trial0/log.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train_moveint.py --results logs/corrected/nuisi_pepper/moveint_combined_3comps/trial1/ --ckpt logs/corrected/bp_pepper/moveint_combined_3comps/trial3/models/final_399.pth --epochs 400 --num-components 3 --dataset NuiSIPepper --hidden-sizes 40 20 --latent-dim 5 >> logs/nuisi_pepper/moveint_combined_3comps/trial1/log.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train_moveint.py --results logs/corrected/nuisi_pepper/moveint_combined_3comps/trial2/ --ckpt logs/corrected/bp_pepper/moveint_combined_3comps/trial3/models/final_399.pth --epochs 400 --num-components 3 --dataset NuiSIPepper --hidden-sizes 40 20 --latent-dim 5 >> logs/nuisi_pepper/moveint_combined_3comps/trial2/log.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train_moveint.py --results logs/corrected/nuisi_pepper/moveint_combined_3comps/trial3/ --ckpt logs/corrected/bp_pepper/moveint_combined_3comps/trial3/models/final_399.pth --epochs 400 --num-components 3 --dataset NuiSIPepper --hidden-sizes 40 20 --latent-dim 5 >> logs/nuisi_pepper/moveint_combined_3comps/trial3/log.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 train_moveint.py --results logs/corrected/bp_yumi/moveint_combined_3comps/trial0/ --epochs 400 --num-components 3 --dataset BuetepageYumi--hidden-sizes 40 20 >> logs/corrected/bp_yumi/moveint_combined_3comps/trial0/log.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train_moveint.py --results logs/corrected/bp_yumi/moveint_combined_3comps/trial1/ --epochs 400 --num-components 3 --dataset BuetepageYumi--hidden-sizes 40 20 >> logs/corrected/bp_yumi/moveint_combined_3comps/trial1/log.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train_moveint.py --results logs/corrected/bp_yumi/moveint_combined_3comps/trial2/ --epochs 400 --num-components 3 --dataset BuetepageYumi--hidden-sizes 40 20 >> logs/corrected/bp_yumi/moveint_combined_3comps/trial2/log.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train_moveint.py --results logs/corrected/bp_yumi/moveint_combined_3comps/trial3/ --epochs 400 --num-components 3 --dataset BuetepageYumi--hidden-sizes 40 20 >> logs/corrected/bp_yumi/moveint_combined_3comps/trial3/log.txt &


CUDA_VISIBLE_DEVICES=0 nohup python3 train_moveint.py --results logs/alap_kobo/moveint_combined_3comps/trial0/ --epochs 400 --num-components 3 --dataset HandoverKobo--hidden-sizes 40 20 >> logs/alap_kobo/moveint_combined_3comps/trial0/log.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train_moveint.py --results logs/alap_kobo/moveint_combined_3comps/trial1/ --epochs 400 --num-components 3 --dataset HandoverKobo--hidden-sizes 40 20 >> logs/alap_kobo/moveint_combined_3comps/trial1/log.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train_moveint.py --results logs/alap_kobo/moveint_combined_3comps/trial2/ --epochs 400 --num-components 3 --dataset HandoverKobo--hidden-sizes 40 20 >> logs/alap_kobo/moveint_combined_3comps/trial2/log.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train_moveint.py --results logs/alap_kobo/moveint_combined_3comps/trial3/ --epochs 400 --num-components 3 --dataset HandoverKobo--hidden-sizes 40 20 >> logs/alap_kobo/moveint_combined_3comps/trial3/log.txt &


CUDA_VISIBLE_DEVICES=0 nohup python3 train_moveint.py --results logs/corrected/kobo_leftin_bi_nowindow/moveint_combined_3comps/trial0/ --epochs 400 --num-components 3 --dataset HandoverKobo--hidden-sizes 40 20 >> logs/corrected/kobo_leftin_bi_nowindow/moveint_combined_3comps/trial0/log.txt &