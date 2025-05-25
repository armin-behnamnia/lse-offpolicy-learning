python main_semi_ot.py --config config/new_cifar/linear/ips_bandit_KL2K.yaml --tau 1.0 --ul 1 --device cuda:0 --wd 0.0019 --kl2 0.32 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/new_cifar/linear/ips_bandit_KL2K.yaml --tau 1.0 --ul 4 --device cuda:0 --wd 0.0056 --kl2 0.31 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/new_cifar/linear/ips_bandit_KL2K.yaml --tau 1.0 --ul 9 --device cuda:0 --wd 0.01 --kl2 0.68 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/new_cifar/linear/ips_bandit_KL2K.yaml --tau 1.0 --ul 49 --device cuda:0 --wd 0.0004 --kl2 0.0004 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/new_cifar/linear/ips_bandit_KL2K.yaml --tau 0.2 --ul 1 --device cuda:0 --wd 0.0008 --kl2 0.28 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/new_cifar/linear/ips_bandit_KL2K.yaml --tau 0.2 --ul 4 --device cuda:0 --wd 0.0113 --kl2 0.63 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/new_cifar/linear/ips_bandit_KL2K.yaml --tau 0.2 --ul 9 --device cuda:0 --wd 0.0041 --kl2 0.46 --ignore_unlabeled --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/new_cifar/linear/ips_bandit_KL2K.yaml --tau 0.2 --ul 49 --device cuda:0 --wd 0.0002 --kl2 1.84 --ignore_unlabeled --raw_image --linear --feature_size 2048
