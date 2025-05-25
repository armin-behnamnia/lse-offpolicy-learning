tau=$1
python preprocess_fmnist_from_model.py --config code/config/h0_0.4/bandit.yaml --tau ${tau} --ul 0 --feature
python preprocess_fmnist_from_model.py --config code/config/h0_0.4/bandit.yaml --tau ${tau} --ul 1 --feature
python preprocess_fmnist_from_model.py --config code/config/h0_0.4/bandit.yaml --tau ${tau} --ul 4 --feature
python preprocess_fmnist_from_model.py --config code/config/h0_0.4/bandit.yaml --tau ${tau} --ul 9 --feature
python preprocess_fmnist_from_model.py --config code/config/h0_0.4/bandit.yaml --tau ${tau} --ul 49 --feature
