python main_semi_ot.py --config config/emnist/linear/lse_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda -888 --gamma_noise_beta 1.0

#with no weight decay:

python main_semi_ot.py --config config/emnist/linear/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda -888 --gamma_noise_beta 1.0 --disable_weight_decay

