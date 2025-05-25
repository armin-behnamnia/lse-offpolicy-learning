tau=$1
python main_semi_ot.py --config config/emnist/deep/bandit_KL2.yaml --tau ${tau} --ul 0 --device cuda:0
python main_semi_ot.py --config config/emnist/deep/bandit_KL2.yaml --tau ${tau} --ul 1 --device cuda:0
python main_semi_ot.py --config config/emnist/deep/bandit_KL2.yaml --tau ${tau} --ul 4 --device cuda:0
python main_semi_ot.py --config config/emnist/deep/bandit_KL2.yaml --tau ${tau} --ul 9 --device cuda:0
python main_semi_ot.py --config config/emnist/deep/bandit_KL2.yaml --tau ${tau} --ul 49 --device cuda:0