import torch
import numpy as np
from tqdm import tqdm
import os
from utils import *
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from utils import create_tensors


def readfile(path, hyper_params):
    if "feature_size" in hyper_params:
        feature_size = hyper_params["feature_size"]
    else:
        raise ValueError("Feature size not set.")
    print("FEATURE_SIZE =", feature_size)

    data = np.load(path, allow_pickle=True)
    logged_data, gt = data["logged_data"], data["gt"].item()
    print("Loaded data address and shape: ", path, logged_data.shape)
    assert (
        logged_data.shape[1] == feature_size + 5
    ), "Data is unlabeled, but fully labeled loader is used."

    x, ids, action, prop, reward, labeled = (
        logged_data[:, :feature_size],
        logged_data[:, feature_size],
        logged_data[:, feature_size + 1],
        logged_data[:, feature_size + 2],
        logged_data[:, feature_size + 3],
        logged_data[:, feature_size + 4],
    )
    return x, ids, action, prop, reward, labeled, gt


class RecDataset:
    def __init__(self, data_path, base_path, hyper_params):
        self.items = np.load(os.path.join(base_path, "item_features.npz"))["arr_0"]
        # self.type = type
        (
            self.x,
            self.user_ids,
            self.action,
            self.prop,
            self.reward,
            self.labeled,
            self.item_rewards_list,
        ) = readfile(data_path, hyper_params)
        # self.user_features = np.load(os.path.join(base_path, "user_features.npz"))[
        #     "arr_0"
        # ]
        # print(self.item_rewards_list)
        print(type(self.item_rewards_list))
        print(len(self.x), len(self.user_ids))

    def __len__(self):
        return len(self.x)

    @property
    def item_matrix(self):
        return self.items

    # def _get_train_idx(self, idx):
    #     assert self.type == "train"
    #     return (
    #         self.x[idx],
    #         self.items[self.action[idx]],
    #         self.prop[idx],
    #         self.reward[idx],
    #         self.labeled[idx],
    #     )

    # def _get_eval_idx(self, idx):
    #     assert self.type != "train"
    #     item_rewards = self.item_rewards_list[idx]
    #     items = item_rewards["items"]
    #     rewards = item_rewards["rewards"]
    #     return self.user_features[self.user_ids[idx]], self.items[items], rewards

    def __getitem__(self, idx):
        x = self.x[idx]
        _id = self.user_ids[idx]
        action = self.action[idx]
        rewards = self.item_rewards_list[_id]["rewards"]
        arms = self.item_rewards_list[_id]["items"]
        prop = self.prop[idx]
        labeled = self.labeled[idx]
        return (
            torch.tensor(x).float(),
            _id,
            action,
            prop,
            self.reward[idx],
            labeled,
            [torch.tensor(rewards).float(), torch.tensor(arms).long()],
        )
        # if self.type == "train":
        #     return self._get_train_idx(idx)
        # else:
        #     return self._get_eval_idx(idx)

    def collate_fn(self, data):
        x, ids, action, prop, reward, labeled, eval_data = zip(*data)
        x = torch.stack(x)
        ids = torch.tensor(ids).long()
        action = torch.tensor(action).long()
        prop = torch.tensor(prop).float()
        reward = torch.tensor(reward).float()
        labeled = torch.tensor(labeled).long()
        all_rewards, all_actions = zip(*eval_data)
        batch_idx = torch.tensor(
            [
                j
                for j, action_array in enumerate(all_actions)
                for _ in range(len(action_array))
            ]
        ).long()
        all_rewards = torch.cat(all_rewards)
        all_actions = torch.cat(all_actions)
        return (
            x,
            ids,
            action,
            prop,
            reward,
            labeled,
            all_rewards,
            all_actions,
            batch_idx,
        )


def load_data_fast(hyper_params):
    store_folder = hyper_params.dataset["name"]
    path = f"../data/{store_folder}/bandit_data"
    path += "_sampled_" + str(hyper_params["num_sample"])
    bese_store_folder = store_folder.split("_")[0]
    base_path = f"../data/{bese_store_folder}/"
    train_dataset = RecDataset(
        path + "_train.npz", base_path, hyper_params=hyper_params
    )
    train_loader = TorchDataLoader(
        train_dataset,
        num_workers=4,
        batch_size=hyper_params["batch_size"],
        collate_fn=train_dataset.collate_fn,
    )
    val_dataset = RecDataset(path + "_val.npz", base_path, hyper_params=hyper_params)
    val_loader = TorchDataLoader(
        val_dataset,
        num_workers=4,
        batch_size=hyper_params["batch_size"],
        collate_fn=train_dataset.collate_fn,
    )
    test_dataset = RecDataset(path + "_test.npz", base_path, hyper_params=hyper_params)
    test_loader = TorchDataLoader(
        test_dataset,
        num_workers=4,
        batch_size=hyper_params["batch_size"],
        collate_fn=train_dataset.collate_fn,
    )
    return train_loader, val_loader, test_loader
