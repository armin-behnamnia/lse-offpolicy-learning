# python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL2K.yaml --tau 0.1 --ul 1 --device cuda:1 --wd 0.003 --kl2 2.719 --ignore_unlabeled --raw_image --linear --feature_size 2048
# python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL2K.yaml --tau 0.1 --ul 4 --device cuda:1 --wd 0.0156 --kl2 2.433 --ignore_unlabeled --raw_image --linear --feature_size 2048
# python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL2K.yaml --tau 0.1 --ul 9 --device cuda:1 --wd 0.0024 --kl2 3.119 --ignore_unlabeled --raw_image --linear --feature_size 2048
# python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL2K.yaml --tau 0.1 --ul 49 --device cuda:1 --wd 0.0141 --kl2 0.4807 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL2K.yaml --tau 0.05 --ul 1 --device cuda:1 --wd 0.0006 --kl2 1.91 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL2K.yaml --tau 0.05 --ul 4 --device cuda:1 --wd 0.0012 --kl2 0.55 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL2K.yaml --tau 0.05 --ul 9 --device cuda:1 --wd 0.0048 --kl2 4.25 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL2K.yaml --tau 0.05 --ul 49 --device cuda:1 --wd 0.0295 --kl2 1.58 --ignore_unlabeled --raw_image --linear --feature_size 2048
