python main_semi_ot.py --config config/rct/linear/ops_bandit.yaml --tau 0.1 --ul 0 --device cuda:0 --raw_image --linear --ops_lambda 0.01 --disable_weight_decay
python main_semi_ot.py --config config/rct/linear/ops_bandit.yaml --tau 0.1 --ul 0 --device cuda:0 --raw_image --linear --ops_lambda 0.1  --disable_weight_decay
python main_semi_ot.py --config config/rct/linear/ops_bandit.yaml --tau 0.1 --ul 0 --device cuda:0 --raw_image --linear --ops_lambda 1.0  --disable_weight_decay
