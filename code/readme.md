### Here an example of the full command syntax of each script is provided.

`python train_logging_policy.py --config config/fmnist/h0/linear_raw.yaml -p 1.0 -d cuda:0 -t 0.05 --dataset fmnist --linear --raw_image`
This code trains a logging policy using the config file "config/fmnist/h0/linear_raw.yaml" in device "cuda:0" with inverse temperature 0.05 for FashionMNIST (use "cifar" for CIFAR-10 dataset) dataset as a linear model on raw flattened images.

`python create_bandit_dataset.py --config config/fmnist/h0/linear_raw.yaml --linear --tau 0.1 --ul 49 --dataset fmnist --raw_image`
This code generates a bandit dataset using the config file "config/fmnist/h0/linear_raw.yaml" and from a linear logging policy that is trained with inverse temperature 0.05 for FashionMNIST dataset on raw flattened images, where the ratio of logged-missing-reward to logged-known reward samples is 49.

`python main_semi_ot.py --config config/fmnist/linear/ips_bandit_KLK.yaml --tau 0.1 --ul 49 --device cuda:0 --linear --raw_image`
This code trains our methods based on the config file "config/fmnist/h0/linear_raw.yaml" and from a linear logging policy with temperature 0.1 on raw flattened images, where the ratio of logged-missing-reward to logged-known reward samples is 49.
