python main_semi_ot.py --config config/emnist/linear/powermean_bandit.yaml --tau 1.0 --ul 0 --device cuda:1 --raw_image --linear --power_mean_lambda 0.5 --unbalance 5 0.5

#With no weight decay:

python main_semi_ot.py --config config/emnist/linear/powermean_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:1 --raw_image --linear --power_mean_lambda 0.5 --unbalance 5 0.5 --disable_weight_decay

#With data_repeat:

python main_semi_ot.py --config config/emnist/linear/powermean_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:1 --raw_image --linear --power_mean_lambda 0.5 --unbalance 5 0.8 --data_repeat 2 --disable_weight_decay