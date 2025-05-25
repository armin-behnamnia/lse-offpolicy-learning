tau=$1
python main_semi_ot.py --config config/cifar/linear/ips_bandit.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 0
python main_semi_ot.py --config config/cifar/linear/ips_bandit_KL.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 0
python main_semi_ot.py --config config/cifar/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau ${tau} --ul 0
