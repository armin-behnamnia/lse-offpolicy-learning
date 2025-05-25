python main_semi_ot.py --config config/cifar/linear/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.01 --feature_size 2048

#with no weight decay:

python main_semi_ot.py --config config/cifar/linear/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.01 --disable_weight_decay --feature_size 2048 --gamma_noise_beta 1.0

