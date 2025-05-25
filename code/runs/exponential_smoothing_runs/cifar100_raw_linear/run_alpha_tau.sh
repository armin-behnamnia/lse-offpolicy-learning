tau=$1
device=$2
python main_semi_ot.py --config config/cifar100/linear/exponential_smoothing_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --exs_alpha 0.1 --disable_weight_decay --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/exponential_smoothing_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --exs_alpha 0.4 --disable_weight_decay --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/exponential_smoothing_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --exs_alpha 0.7 --disable_weight_decay --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/exponential_smoothing_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:${device} --raw_image --linear --exs_alpha 1 --disable_weight_decay --feature_size 2048