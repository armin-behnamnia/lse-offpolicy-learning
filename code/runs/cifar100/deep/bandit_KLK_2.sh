# python main_semi_ot.py --config config/cifar100/deep/bandit_KLK.yaml --tau 0.1 --ul 1 --device cuda:0 --wd 0.0002 --kl 0.372 --ignore_unlabeled
# python main_semi_ot.py --config config/cifar100/deep/bandit_KLK.yaml --tau 0.1 --ul 4 --device cuda:0 --wd 0.0046 --kl 0.2155 --ignore_unlabeled
# python main_semi_ot.py --config config/cifar100/deep/bandit_KLK.yaml --tau 0.1 --ul 9 --device cuda:0 --wd 0.0018 --kl 0.1265 --ignore_unlabeled
# python main_semi_ot.py --config config/cifar100/deep/bandit_KLK.yaml --tau 0.1 --ul 49 --device cuda:0 --wd 0.0014 --kl 1.029 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KLK.yaml --tau 0.05 --ul 1 --device cuda:0 --wd 0.0002 --kl 0.01 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KLK.yaml --tau 0.05 --ul 4 --device cuda:0 --wd  --kl  --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KLK.yaml --tau 0.05 --ul 9 --device cuda:0 --wd 0.00002 --kl 0.25 --ignore_unlabeled
python main_semi_ot.py --config config/cifar100/deep/bandit_KLK.yaml --tau 0.05 --ul 49 --device cuda:0 --wd 0.004 --kl 0.15 --ignore_unlabeled
