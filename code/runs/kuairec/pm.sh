device=$1
python main_semi_rec.py --config config/kuairec/powermean_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.1 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/powermean_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.5 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/powermean_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.8 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/powermean_bandit_no_wd_ns3.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.1 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/powermean_bandit_no_wd_ns3.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.5 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/powermean_bandit_no_wd_ns3.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.8 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/powermean_bandit_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.1 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/powermean_bandit_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.5 --disable_weight_decay
python main_semi_rec.py --config config/kuairec/powermean_bandit_no_wd_ns5.yaml --tau 1.0 --ul 0 --device cuda:${device} --raw_image --linear --power_mean_lambda 0.8 --disable_weight_decay
