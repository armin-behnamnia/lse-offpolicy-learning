tau=$1
device=$2
# python main_semi_ot.py --config config/cifar100/linear/lse_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda -0.1 --disable_weight_decay
# python main_semi_ot.py --config config/cifar100/linear/lse_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda -0.01 --disable_weight_decay
python main_semi_ot.py --config config/cifar100/linear/lse_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda 0.01 --disable_weight_decay --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/lse_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda 0.1 --disable_weight_decay --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/lse_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda 1 --disable_weight_decay --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/lse_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda 10 --disable_weight_decay --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/lse_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda 100 --disable_weight_decay --feature_size 2048