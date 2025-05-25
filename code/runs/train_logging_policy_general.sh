dataset=$1
device=$2
python train_logging_policy.py --device ${device} --config config/${dataset}/supervised.yaml --tau 1.0 --proportion 1.0 --dataset ${dataset} --linear
