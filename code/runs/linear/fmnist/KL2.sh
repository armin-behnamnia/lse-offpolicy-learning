tau=$1
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 0
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 1
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 4
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 9
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 49
