python main_semi_ot.py --config config/fmnist/linear/powermean_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.5

#With no weight decay:

python main_semi_ot.py --config config/fmnist/linear/powermean_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.5 --disable_weight_decay