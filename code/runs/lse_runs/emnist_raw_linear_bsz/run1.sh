CUDA_VISIBLE_DEVICES=0 python main_semi_ot.py --config config/emnist/linear/lse_bandit_hpc_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.05 --disable_weight_decay
CUDA_VISIBLE_DEVICES=0 python main_semi_ot.py --config config/emnist/linear/lse_bandit_hpc_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.5 --disable_weight_decay
CUDA_VISIBLE_DEVICES=0 python main_semi_ot.py --config config/emnist/linear/lse_bandit_hpc_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 5 --disable_weight_decay
CUDA_VISIBLE_DEVICES=0 python main_semi_ot.py --config config/emnist/linear/lse_bandit_hpc_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.05 --disable_weight_decay
CUDA_VISIBLE_DEVICES=0 python main_semi_ot.py --config config/emnist/linear/lse_bandit_hpc_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.5 --disable_weight_decay
CUDA_VISIBLE_DEVICES=0 python main_semi_ot.py --config config/emnist/linear/lse_bandit_hpc_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 5 --disable_weight_decay
