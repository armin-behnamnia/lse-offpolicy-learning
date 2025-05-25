dataset=$1
tau=$2
python main_semi_ot.py --config config/${dataset}/linear/DEEPLOG_ips_bandit_KL.yaml --device cuda:0 --linear --tau ${tau} --ul 0 --raw_image
python main_semi_ot.py --config config/${dataset}/linear/DEEPLOG_ips_bandit_KL2.yaml --device cuda:0 --linear --tau ${tau} --ul 0 --raw_image
