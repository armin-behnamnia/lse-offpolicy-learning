python main_semi_ot.py --config config/emnist/linear/lse_bandit_AR_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda -1

#no weight decay:

python main_semi_ot.py --config config/emnist/linear/lse_bandit_AR_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 7 --disable_weight_decay


#special:

python main_semi_ot.py --config config/emnist/linear/special.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 7 --disable_weight_decay
