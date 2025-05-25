tau=$1
ul=$2
python main_semi_ot.py --device cuda:0 --config config/fmnist/tau${tau}/ul${ul}/ips_bandit.yaml --feature_size 512
python main_semi_ot.py --device cuda:0 --config config/fmnist/tau${tau}/ul${ul}/ips_bandit_KL.yaml --feature_size 512
python main_semi_ot.py --device cuda:0 --config config/fmnist/tau${tau}/ul${ul}/ips_bandit_KL2.yaml --feature_size 512
python main_semi_ot.py --device cuda:0 --config config/fmnist/tau${tau}/ul${ul}/ips_KL.yaml --feature_size 512
python main_semi_ot.py --device cuda:0 --config config/fmnist/tau${tau}/ul${ul}/ips_KL2.yaml --feature_size 512
