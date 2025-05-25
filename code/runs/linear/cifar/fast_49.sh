tau=$1
python main_semi_ot.py --config config/cifar/linear/ips_bandit.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 49
python main_semi_ot.py --config config/cifar/linear/ips_bandit_KL.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 49
python main_semi_ot.py --config config/cifar/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 49
python main_semi_ot.py --config config/cifar/linear/ips_bandit_KLK.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 49 --ignore_unlabeled
python main_semi_ot.py --config config/cifar/linear/ips_bandit_KL2K.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 49 --ignore_unlabeled
