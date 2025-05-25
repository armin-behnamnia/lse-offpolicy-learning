dataset=$1
tau=$2
python create_bandit_dataset.py --linear --tau ${tau} --ul 0 --dataset ${dataset} --config config/${dataset}/h0/linear.yaml
python create_bandit_dataset.py --linear --tau ${tau} --ul 1 --dataset ${dataset} --config config/${dataset}/h0/linear.yaml
python create_bandit_dataset.py --linear --tau ${tau} --ul 4 --dataset ${dataset} --config config/${dataset}/h0/linear.yaml
python create_bandit_dataset.py --linear --tau ${tau} --ul 9 --dataset ${dataset} --config config/${dataset}/h0/linear.yaml
python create_bandit_dataset.py --linear --tau ${tau} --ul 49 --dataset ${dataset} --config config/${dataset}/h0/linear.yaml
