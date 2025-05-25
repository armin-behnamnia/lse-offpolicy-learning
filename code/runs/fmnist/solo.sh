tau=$1
ul=$2
python main_semi_ot.py --device cuda:0 --config config/fmnist/tau${tau}/ul${ul}/KL.yaml
python main_semi_ot.py --device cuda:0 --config config/fmnist/tau${tau}/ul${ul}/KL2.yaml
python main_semi_ot.py --device cuda:0 --config config/fmnist/tau${tau}/ul${ul}/bandit.yaml
