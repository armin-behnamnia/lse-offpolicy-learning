# python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.1 --ul 1 --device cuda:0 --wd 0.00005 --kl2 0.1993 --ignore_unlabeled
# python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.1 --ul 4 --device cuda:0 --wd 0.0002 --kl2 1.039 --ignore_unlabeled
# python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.1 --ul 9 --device cuda:0 --wd 0.00004 --kl2 0.9857 --ignore_unlabeled
# python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.1 --ul 49 --device cuda:0 --wd 0.00009 --kl2 1.697 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.05 --ul 1 --device cuda:0 --wd 0.0006 --kl2 0.03 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.05 --ul 4 --device cuda:0 --wd 0.0002 --kl2 0.06 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.05 --ul 9 --device cuda:0 --wd 0.0006 --kl2 0.28 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KL2K.yaml --tau 0.05 --ul 49 --device cuda:0 --wd 0.0001 --kl2 0.17 --ignore_unlabeled
