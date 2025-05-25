python main_semi_ot.py --config config/rct/linear/lse_bandit_AR.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.01 --disable_weight_decay
python main_semi_ot.py --config config/rct/linear/lse_bandit_AR.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.1 --disable_weight_decay
python main_semi_ot.py --config config/rct/linear/lse_bandit_AR.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 1 --disable_weight_decay
