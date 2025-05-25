device=$1
python main_semi_rec.py --config config/kuairec/lse_bandit_AR_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda 0.01 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/lse_bandit_AR_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda 0.1 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/lse_bandit_AR_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda 1 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/lse_bandit_AR_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda 10 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/lse_bandit_AR_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --lse_lambda 100 --disable_weight_decay
