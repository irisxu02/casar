import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from tqdm import tqdm
import json


def load_frame(path: str, frame: int):
    """
    Loads a hand, object, and label for a single frame

    Args:
        path    : Path to h2o dataset video segment
        frame   : Frame number to load

    Returns:
        hand    (3 x 21 x 2) numpy array of (x, y, z) points corresponding
                    to 21 joints on left and right hand
        obj     (3 x 21) numpy array of (x, y, z) points corresponding
                    to 21 points of object bounding box
        label   8-element one-hot vector denoting object label
    """
    hand_pose = np.loadtxt(os.path.join(path, f"cam4/hand_pose/{frame:06d}.txt"))
    obj_pose = np.loadtxt(os.path.join(path, f"cam4/obj_pose/{frame:06d}.txt"))

    # Ignore first element of each hand, actual points start at second element
    left = hand_pose[1:64].reshape((3, 21), order="F")
    right = hand_pose[65:128].reshape((3, 21), order="F")

    # Combine into 3 x 21 x 2 matrix
    hand = np.stack((left, right), axis=2)

    # First element is object label, actual points start at second element
    obj = obj_pose[1:].reshape((3, 21), order="F")

    # First element is object label
    label = np.zeros(8)
    label[obj_pose[0].astype(int) - 1] = 1

    return hand, obj, label


# Mapping from dataset split to path to corresponding index file
split_index_mapping = {
    "train": "action_labels/action_train.txt",
    "test": "action_labels/action_test.txt",
    "val": "action_labels/action_val.txt",
}


class H2OContactDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str,
        eta_c: float = 0.02,
        eta_d: float = 0.2,
        use_cache: bool = True,
    ):
        self._eta_c_sq = eta_c**2
        self._eta_d_sq = eta_d**2
        self._use_cache = use_cache

        loaded_paths = set()
        self._elements = []

        self._base_dir = os.path.abspath(dataset_path)
        index_file = os.path.join(self._base_dir, split_index_mapping[split])

        with open(index_file, "r") as file:
            # Skip first line (the header)
            next(file)
            for line in file:
                tokens = line.split(" ")

                # Only load unique video segments
                if tokens[1] in loaded_paths:
                    continue
                loaded_paths.add(tokens[1])

                if split == "test":
                    # Test dataset doesn't have action_labels column
                    start_frame = int(tokens[4])
                    end_frame = int(tokens[5])
                else:
                    start_frame = int(tokens[5])
                    end_frame = int(tokens[6])

                for frame in range(start_frame, end_frame + 1):
                    self._elements.append((tokens[1], frame))

        if self._use_cache:
            self._cache: list[None | tuple] = [None for _ in self._elements]

    def __len__(self):
        return len(self._elements)

    def _get_item(self, idx):
        rel_path, frame = self._elements[idx]
        path = os.path.join(self._base_dir, rel_path)

        hand, obj, label = load_frame(path, frame)
        pi = np.concatenate((hand.flatten(), obj.flatten(), label))

        # Compute contact and distant points
        is_contact = []
        is_distant = []
        for h in range(42):
            if h < 21:
                hand_points = hand[:, h, 0]
            else:
                hand_points = hand[:, h - 21, 1]

            min_dist_sq = np.inf
            for o in range(21):
                obj_points = obj[:, o]

                dist_sq = np.sum((hand_points - obj_points) ** 2)
                min_dist_sq = min(min_dist_sq, dist_sq)

            is_contact.append(min_dist_sq <= self._eta_c_sq)
            is_distant.append(min_dist_sq >= self._eta_d_sq)

        ci = np.array(is_contact).astype(int)
        di = np.array(is_distant).astype(int)
        qi = np.concatenate((ci, di))
        return torch.from_numpy(pi), torch.from_numpy(qi)

    def __getitem__(self, idx):
        if self._use_cache:
            if self._cache[idx] == None:
                self._cache[idx] = self._get_item(idx)
            return self._cache[idx]
        return self._get_item(idx)


class H2OSkeletonDataset(Dataset):
    def __init__(
        self, dataset_path: str, split: str, N: int = 32, use_cache: bool = True
    ):
        self._N = N
        self._use_cache = use_cache

        self._elements = []

        self._base_dir = os.path.abspath(dataset_path)
        index_file = os.path.join(self._base_dir, split_index_mapping[split])

        with open(index_file, "r") as file:
            # Skip first line (the header)
            next(file)
            for line in file:
                tokens = line.split(" ")

                if split == "test":
                    # Test dataset doesn't have action_labels column
                    action_label = -1
                    start_act = int(tokens[2])
                    end_act = int(tokens[3])
                else:
                    action_label = int(tokens[2])
                    start_act = int(tokens[3])
                    end_act = int(tokens[4])

                self._elements.append((tokens[1], action_label, start_act, end_act))

        if self._use_cache:
            self._cache: list[None | tuple] = [None for _ in self._elements]

    def __len__(self):
        return len(self._elements)

    def _get_item(self, idx):
        rel_path, action_label, start_act, end_act = self._elements[idx]
        path = os.path.join(self._base_dir, rel_path)

        # Sample N frames from action sequence
        # Rounding will handle repeat strategy
        frames = np.round(np.linspace(start_act, end_act, self._N)).astype(np.int32)
        xjs = []
        for frame in frames:
            hand, obj, label = load_frame(path, frame)
            xj = np.concatenate((hand.flatten(), obj.flatten(), label))
            xjs.append(xj)

        xi = np.stack(xjs)

        yi = torch.zeros(32)
        if action_label != -1:
            yi[action_label - 1] = 1
        return torch.from_numpy(xi), yi

    def __getitem__(self, idx):
        if self._use_cache:
            if self._cache[idx] == None:
                self._cache[idx] = self._get_item(idx)
            return self._cache[idx]
        return self._get_item(idx)


class MLP(nn.Module):
    """MLP with 2 hidden layers"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class ContactAwareModule(nn.Module):
    """
    Network f: Predicts contact-map from skeleton data
    Input: (batch, num_frames, 197)
    Output: (batch, num_frames, 84)
    """

    def __init__(self, hidden_dim=256):
        super(ContactAwareModule, self).__init__()
        self.mlp = MLP(input_dim=197, hidden_dim=hidden_dim, output_dim=84)

    def forward(self, skeleton_seq):
        batch_size, num_frames, _ = skeleton_seq.shape
        skeleton_flat = skeleton_seq.view(batch_size * num_frames, -1)
        contact_map_flat = self.mlp(skeleton_flat)
        contact_map = contact_map_flat.view(batch_size, num_frames, -1)
        return contact_map


class ActionRecognitionModule(nn.Module):
    def __init__(self, num_frames=32, num_classes=36, hidden_dim=5000):
        super(ActionRecognitionModule, self).__init__()
        self.num_frames = num_frames
        input_dim = num_frames * (197 + 84)  # Flattened sequence
        self.mlp = MLP(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes
        )

    def forward(self, skeleton_seq, contact_map):
        """
        Args:
            skeleton_seq: (batch, num_frames, 197)
            contact_map: (batch, num_frames, 84)
        Returns:
            action_logits: (batch, num_classes)
        """
        combined = torch.cat(
            [skeleton_seq, contact_map], dim=-1
        )  # (batch, num_frames, 281)
        # Flatten temporal dimension
        combined_flat = combined.view(
            combined.shape[0], -1
        )  # (batch, num_frames * 281)
        action_logits = self.mlp(combined_flat)
        return action_logits


class CaSAR(nn.Module):
    def __init__(self, num_frames=32, num_classes=36):
        super(CaSAR, self).__init__()
        self.contact_aware_module = ContactAwareModule(hidden_dim=256)
        self.action_recognition_module = ActionRecognitionModule(
            num_frames=num_frames, num_classes=num_classes, hidden_dim=5000
        )

    def forward(self, skeleton_seq):
        """
        Args:
            skeleton_seq: (batch, num_frames, 197)
        Returns:
            contact_map: (batch, num_frames, 84)
            action_logits: (batch, num_classes)
        """
        contact_map = self.contact_aware_module(skeleton_seq)
        action_logits = self.action_recognition_module(skeleton_seq, contact_map)

        return contact_map, action_logits


epsilon = 1e-8


def focal_loss(qi, qi_h, alpha=0.5, gamma=4):
    """
    Implementation of focal loss defined in equation (3)
    """
    # Term 1: -alpha * qi * (1 - qi_hat)^gamma * log(qi_hat)
    t1 = alpha * qi * torch.pow((1 - qi_h), gamma) * torch.log(qi_h + epsilon)

    # Term 2: -(1 - alpha) * (1 - qi) * (qi_hat)^gamma * log(1 - qi_hat)
    t2 = (1 - alpha) * (1 - qi) * torch.pow(qi_h, gamma) * torch.log(1 - qi_h + epsilon)

    loss = t1 + t2
    return -loss.mean()
