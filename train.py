import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from modules import CaSAR, H2ODataset

def load_hyperparameters(filepath):
    with open(filepath, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams

def train_f(model, dataloader, optimizer, scheduler, device, num_epochs=100):
    pass

def train_g(model, dataloader, optimizer, scheduler, device, num_epochs=600):
    # make sure to freeze f first
    pass

def evaluate(model, dataloader, device):
    model.eval()
    pass

def main():
    # TODO: implement main training loop
    
    # TODO: could load hyperparams from config
    # config_path = 'config.yaml'
    # hyperparameters = load_hyperparameters(config_path)

    DATA_PATH = 'h2o_CASA'
    BATCH_SIZE = 32 # not specified in paper, so may need to adjust this
    NUM_FRAMES = 32
    NUM_CLASSES = 36 # maybe this should be 37, since there's label 0 for background
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # TODO: Need to figure out how to split data, since I don't think we have a test split from the raw dataset
    train_dataset = H2ODataset(DATA_PATH, split='train', num_frames=NUM_FRAMES)
    val_dataset = H2ODataset(DATA_PATH, split='val', num_frames=NUM_FRAMES)
    #test_dataset = H2ODataset(DATA_PATH, split='test', num_frames=NUM_FRAMES)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    criterion_f = ops.sigmoid_focal_loss() # double check that this matches the def in the paper
    # may have to implement focal loss ourselves?
    
    criterion_g = nn.BCELoss()
    
    optimizer_f = optim.Adam(f.parameters(), lr=1e-4)
    optimizer_g = optim.Adam(g.parameters(), lr=1e-5)
    
    scheduler_f = optim.lr_scheduler.StepLR(optimizer_f, step_size=20, gamma=0.7)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=200, gamma=0.7)
    
    model = CaSAR(num_frames=NUM_FRAMES, num_classes=NUM_CLASSES).to(device)
  
    
    # train_f
    # train_g
    
    # TODO: evaluate and save model
    torch.save(model.state_dict(), 'casar_model.pth')

if __name__ == '__main__':
    main()
