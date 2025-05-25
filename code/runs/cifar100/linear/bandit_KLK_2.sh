# python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KLK.yaml --tau 0.1 --ul 1 --device cuda:1 --wd 0.0034 --kl 1.49 --ignore_unlabeled --raw_image --linear --feature_size 2048
# python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KLK.yaml --tau 0.1 --ul 4 --device cuda:1 --wd 0.0451 --kl 1.082 --ignore_unlabeled --raw_image --linear --feature_size 2048
# python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KLK.yaml --tau 0.1 --ul 9 --device cuda:1 --wd 0.003 --kl 1.673 --ignore_unlabeled --raw_image --linear --feature_size 2048
# python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KLK.yaml --tau 0.1 --ul 49 --device cuda:1 --wd 0.0003 --kl 0.2742 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KLK.yaml --tau 0.05 --ul 1 --device cuda:1 --wd 0.0022 --kl 2.57 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KLK.yaml --tau 0.05 --ul 4 --device cuda:1 --wd 0.0004 --kl 1.97 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KLK.yaml --tau 0.05 --ul 9 --device cuda:1 --wd 0.0006 --kl 1.52 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KLK.yaml --tau 0.05 --ul 49 --device cuda:1 --wd 0.0157 --kl 0.47 --ignore_unlabeled --raw_image --linear --feature_size 2048
