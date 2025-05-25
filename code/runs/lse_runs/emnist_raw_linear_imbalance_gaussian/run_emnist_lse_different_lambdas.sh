python main_semi_ot.py --config config/emnist/linear/lse_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda -1

#No Weight Decay:

python main_semi_ot.py --config config/emnist/linear/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:1 --raw_image --linear --lse_lambda 888 --disable_weight_decay --gaussian_imbalance 5