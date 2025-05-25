device=$1
python main_semi_rec.py --config config/kuairec/exponential_smoothing_bandit_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --exs_alpha 0.1 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/exponential_smoothing_bandit_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --exs_alpha 0.4 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/exponential_smoothing_bandit_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --exs_alpha 0.7 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/exponential_smoothing_bandit_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --exs_alpha 1.0 --disable_weight_decay
