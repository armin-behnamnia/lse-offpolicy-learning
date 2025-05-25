python main_semi_ot.py --config config/emnist/linear/lse_bandit.yaml --tau 1.0 --ul 0 --device cuda:1 --raw_image --linear --lse_lambda -1 --unbalance 5 0.5

#No Weight Decay:

python main_semi_ot.py --config config/emnist/linear/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:1 --raw_image --linear --lse_lambda 888 --unbalance 5 0.5 --disable_weight_decay

#with data_repeat:

python main_semi_ot.py --config config/emnist/linear/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:1 --raw_image --linear --lse_lambda 888 --unbalance 5 0.8 --data_repeat 2 --disable_weight_decay