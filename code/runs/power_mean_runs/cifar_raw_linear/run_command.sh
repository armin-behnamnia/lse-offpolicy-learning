python main_semi_ot.py --config config/cifar/linear_raw/powermean_bandit.yaml --tau 0.04 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.1

#with no weight decay:

python main_semi_ot.py --config config/cifar/linear_raw/powermean_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.1 --disable_weight_decay

#with feature:

python main_semi_ot.py --config config/cifar/linear/powermean_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0 --disable_weight_decay --feature_size 2048