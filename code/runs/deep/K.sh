dataset=$1
tau=$2
ul=$3
python main_semi_ot.py --config config/${dataset}/deep/bandit_KLK.yaml --device cuda:0 --tau ${tau} --ul ${ul} --ignore_unlabeled
python main_semi_ot.py --config config/${dataset}/deep/bandit_KL2K.yaml --device cuda:0 --tau ${tau} --ul ${ul} --ignore_unlabeled