python main_semi_ot.py --config config/emnist/linear/lse_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda -1 --reward_flip 0.5

#No Weight Decay:

python main_semi_ot.py --config config/emnist/linear/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 888 --reward_flip 0.5 --disable_weight_decay