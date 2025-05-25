python main_semi_ot.py --config config/emnist/linear/exponential_smoothing_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --exs_alpha 0.3

#with no weight decay:

python main_semi_ot.py --config config/emnist/linear/exponential_smoothing_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --exs_alpha 0.3 --disable_weight_decay