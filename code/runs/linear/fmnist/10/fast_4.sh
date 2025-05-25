python main_semi_ot.py --config config/fmnist/linear/ips_bandit.yaml --device cuda:0 --feature_size 512 --tau 0.1 --ul 4
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL.yaml --device cuda:0 --feature_size 512 --tau 0.1 --ul 4
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL2.yaml --device cuda:0 --feature_size 512 --tau 0.1 --ul 4
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KLK.yaml --device cuda:0 --feature_size 512 --tau 0.1 --ul 4 --ignore_unlabeled
python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KL2K.yaml --device cuda:0 --feature_size 512 --tau 0.1 --ul 4 --ignore_unlabeled