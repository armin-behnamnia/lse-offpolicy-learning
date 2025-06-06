dataset=$1
tau=$2
python main_semi_ot.py --config config/${dataset}/linear/ips_bandit_KL.yaml --device cuda:0 --linear --tau ${tau} --ul 49 --raw_image
python main_semi_ot.py --config config/${dataset}/linear/ips_bandit_KL2.yaml --device cuda:0 --linear --tau ${tau} --ul 49 --raw_image
python main_semi_ot.py --config config/${dataset}/linear/ips_bandit_KLK.yaml --device cuda:0 --linear --tau ${tau} --ul 49 --ignore_unlabeled --raw_image
python main_semi_ot.py --config config/${dataset}/linear/ips_bandit_KL2K.yaml --device cuda:0 --linear --tau ${tau} --ul 49 --ignore_unlabeled --raw_image