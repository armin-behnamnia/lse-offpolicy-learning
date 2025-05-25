tau=$1
device=$2
python main_semi_ot.py --config config/cifar100/linear/powermean_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0 --disable_weight_decay --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/powermean_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.1 --disable_weight_decay --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/powermean_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.3 --disable_weight_decay --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/powermean_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.5 --disable_weight_decay --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/powermean_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.8 --disable_weight_decay --feature_size 2048