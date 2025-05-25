python main_semi_ot.py --config config/emnist/deep/bandit_KL2K.yaml --tau 0.1 --ul 1 --device cuda:0 --wd 0.0006 --kl2 0.02 --ignore_unlabeled
python main_semi_ot.py --config config/emnist/deep/bandit_KL2K.yaml --tau 0.1 --ul 4 --device cuda:0 --wd 0.0001 --kl2 0.18 --ignore_unlabeled
python main_semi_ot.py --config config/emnist/deep/bandit_KL2K.yaml --tau 0.1 --ul 9 --device cuda:0 --wd 0.0007 --kl2 0.23 --ignore_unlabeled
python main_semi_ot.py --config config/emnist/deep/bandit_KL2K.yaml --tau 0.1 --ul 49 --device cuda:0 --wd 0.0008 --kl2 3.0 --ignore_unlabeled
python main_semi_ot.py --config config/emnist/deep/bandit_KL2K.yaml --tau 0.05 --ul 1 --device cuda:0 --wd 0.0008 --kl2 0.02 --ignore_unlabeled
python main_semi_ot.py --config config/emnist/deep/bandit_KL2K.yaml --tau 0.05 --ul 4 --device cuda:0 --wd 0.0003 --kl2 0.007 --ignore_unlabeled
python main_semi_ot.py --config config/emnist/deep/bandit_KL2K.yaml --tau 0.05 --ul 9 --device cuda:0 --wd 0.0001 --kl2 0.007 --ignore_unlabeled
python main_semi_ot.py --config config/emnist/deep/bandit_KL2K.yaml --tau 0.05 --ul 49 --device cuda:0 --wd 0.0029 --kl2 1.4 --ignore_unlabeled
