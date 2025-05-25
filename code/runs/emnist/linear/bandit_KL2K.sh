python main_semi_ot.py --config config/emnist/linear/ips_bandit_KL2K.yaml --tau 1.0 --ul 1 --device cuda:1 --wd 0.0057 --kl2 0.1040 --ignore_unlabeled --raw_image --linear 
python main_semi_ot.py --config config/emnist/linear/ips_bandit_KL2K.yaml --tau 1.0 --ul 4 --device cuda:1 --wd 0.0025 --kl2 0.1551 --ignore_unlabeled --raw_image --linear
python main_semi_ot.py --config config/emnist/linear/ips_bandit_KL2K.yaml --tau 1.0 --ul 9 --device cuda:1 --wd 0.0009 --kl2 0.1618 --ignore_unlabeled --raw_image --linear
python main_semi_ot.py --config config/emnist/linear/ips_bandit_KL2K.yaml --tau 1.0 --ul 49 --device cuda:1 --wd 0.0037 --kl2 1.987 --ignore_unlabeled --raw_image --linear
python main_semi_ot.py --config config/emnist/linear/ips_bandit_KL2K.yaml --tau 0.2 --ul 1 --device cuda:1 --wd 0.003 --kl2 0.0671 --ignore_unlabeled --raw_image --linear
python main_semi_ot.py --config config/emnist/linear/ips_bandit_KL2K.yaml --tau 0.2 --ul 4 --device cuda:1 --wd 0.0009 --kl2 0.0623 --ignore_unlabeled --raw_image --linear
python main_semi_ot.py --config config/emnist/linear/ips_bandit_KL2K.yaml --tau 0.2 --ul 9 --device cuda:1 --wd 0.0006 --kl2 0.0735 --ignore_unlabeled --raw_image --linear
python main_semi_ot.py --config config/emnist/linear/ips_bandit_KL2K.yaml --tau 0.2 --ul 49 --device cuda:1 --wd 0.0042 --kl2 1.621 --ignore_unlabeled --raw_image --linear
