import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from tqdm import tqdm
import json


class H2ODataset(Dataset):
    """
    TODO: implement dataset loader
    - Uniformly sample frames
    - Calculate contact vs distant points (ηc = 2 cm, ηd = 20 cm)
    - Load the annotations, map the object and action classes
    """


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
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes)
    
    def forward(self, skeleton_seq, contact_map):
        """
        Args:
            skeleton_seq: (batch, num_frames, 197)
            contact_map: (batch, num_frames, 84)
        Returns:
            action_logits: (batch, num_classes)
        """
        combined = torch.cat([skeleton_seq, contact_map], dim=-1)  # (batch, num_frames, 281)
        # Flatten temporal dimension
        combined_flat = combined.view(combined.shape[0], -1)  # (batch, num_frames * 281)
        action_logits = self.mlp(combined_flat)
        return action_logits

class CaSAR(nn.Module):   
    def __init__(self, num_frames=32, num_classes=36):
        super(CaSAR, self).__init__()
        self.contact_aware_module = ContactAwareModule(hidden_dim=256)
        self.action_recognition_module = ActionRecognitionModule(
            num_frames=num_frames,
            num_classes=num_classes,
            hidden_dim=5000
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
