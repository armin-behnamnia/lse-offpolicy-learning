python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 1.0 --ul 1 --device cuda:0 --wd 0.004 --kl2 0.1620 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 1.0 --ul 4 --device cuda:0 --wd 0.003 --kl2 0.007 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 1.0 --ul 9 --device cuda:0 --wd 0.00007 --kl2 0.0397 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 1.0 --ul 49 --device cuda:0 --wd 0.003 --kl2 0.5313 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.2 --ul 1 --device cuda:0 --wd 0.00007 --kl2 0.1095 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.2 --ul 4 --device cuda:0 --wd 0.0004 --kl2 0.1132 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.2 --ul 9 --device cuda:0 --wd 0.001 --kl2 0.4117 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.2 --ul 49 --device cuda:0 --wd 0.0007 --kl2 0.9708 --ignore_unlabeled
