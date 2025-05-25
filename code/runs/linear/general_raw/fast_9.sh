dataset=$1
tau=$2
<<<<<<< HEAD
# python main_semi_ot.py --config config/${dataset}/linear_wwd/ips_bandit.yaml --device cuda:0 --linear --tau ${tau} --ul 9 --raw_image
=======
>>>>>>> 887332039f18f1b659e56ef0e44ac0c4b80e50e1
python main_semi_ot.py --config config/${dataset}/linear/ips_bandit_KL.yaml --device cuda:0 --linear --tau ${tau} --ul 9 --raw_image
python main_semi_ot.py --config config/${dataset}/linear/ips_bandit_KL2.yaml --device cuda:0 --linear --tau ${tau} --ul 9 --raw_image
python main_semi_ot.py --config config/${dataset}/linear/ips_bandit_KLK.yaml --device cuda:0 --linear --tau ${tau} --ul 9 --ignore_unlabeled --raw_image
python main_semi_ot.py --config config/${dataset}/linear/ips_bandit_KL2K.yaml --device cuda:0 --linear --tau ${tau} --ul 9 --ignore_unlabeled --raw_image