tau=$1
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 1 --ignore_unlabeled
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 4 --ignore_unlabeled
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 9 --ignore_unlabeled
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 49 --ignore_unlabeled
