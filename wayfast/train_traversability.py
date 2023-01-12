#!/usr/bin/env python

import os
import torch
import numpy as np
from tqdm import tqdm
from tqdm.auto import trange
from tqdm.contrib import tenumerate

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.resnet_depth_unet import ResnetDepthUnet
from utils.dataloader import TraversabilityDataset

import matplotlib.pyplot as plt

class Object(object):
    pass

params = Object()
# dataset parameters
params.data_path        = r'data'
params.csv_path         = os.path.join(params.data_path, 'data.csv')
params.preproc          = True  # Vertical flip augmentation
params.depth_mean       = 3.5235
params.depth_std        = 10.6645

# training parameters
params.seed             = 230
params.epochs           = 50
params.batch_size       = 4
params.learning_rate    = 1e-4
params.weight_decay     = 1e-5

# model parameters
params.pretrained = True
params.load_network_path = None 
params.input_size       = (424, 240)
params.output_size      = (424, 240)
params.output_channels  = 1
params.bottleneck_dim   = 256


torch.manual_seed(params.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(params.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


net = ResnetDepthUnet(params)

# use to load a previously trained network
if params.load_network_path is not None:
    print('Loading saved network from {}'.format(params.load_network_path))
    net.load_state_dict(torch.load(params.load_network_path))

print("Let's use", torch.cuda.device_count(), "GPUs!")
net = torch.nn.DataParallel(net).to(device)

test = net(torch.rand([2, 3, params.input_size[1], params.input_size[0]]).to(device), torch.rand([2, 1, params.input_size[1], params.input_size[0]]).to(device))
print('test.shape:', test.shape)

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

dataset = TraversabilityDataset(params, transform)

train_size, val_size = int(0.8*len(dataset)), np.ceil(0.2*len(dataset)).astype('int')
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader    = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2)
test_loader     = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2)

print('Total loaded %d images' % len(dataset))
print('Loaded %d train images' % train_size)
print('Loaded %d valid images' % val_size)

data = train_dataset[0]

criterion = torch.nn.L1Loss(reduction='none')
optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

best_val_loss = np.inf
train_loss_list = []
val_loss_list = []
for epoch in trange(params.epochs, desc='Training'):
    net.train()
    train_loss = 0.0
     
    for i, data in tenumerate(train_loader, desc='Inner'):
        data = (item.to(device).type(torch.float32) for item in data)
        color_img, depth_img, path_img, mu_img, nu_img, weight = data

        pred = net(color_img, depth_img)

        label = mu_img

        loss = weight*criterion(pred*path_img, label)
        loss = torch.mean(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)
        
    if (epoch) % 10 == 0:
        tqdm.write(f'Epoch [{epoch+1}/{params.epochs}], Loss: {train_loss}')
        tqdm.write(f'Learning Rate for this epoch: {optimizer.param_groups[0]["lr"]}')
    
    # evaluate the network on the test data
    with torch.no_grad():
        val_loss = 0.0
        net.eval()
        for i, data in enumerate(test_loader):
            data = (item.to(device).type(torch.float32) for item in data)
            color_img, depth_img, path_img, mu_img, nu_img, weight = data

            pred = net(color_img, depth_img)

            label = mu_img

            loss = weight*criterion(pred*path_img, label)
            loss = torch.mean(loss)

            val_loss += loss.item()
        val_loss /= len(test_loader)
        val_loss_list.append(val_loss)

    if (epoch + 1) % 5 == 0:
        plt.figure(figsize = (14,14))
        plt.subplot(1, 3, 1)
        plt.imshow(color_img[0].permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 3, 2)
        plt.imshow(255*pred[0,0,:,:].detach().cpu().numpy(), vmin=0, vmax=255)
        plt.show(block=False)
    
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        tqdm.write(f'Updating best validation loss: {best_val_loss}')
        torch.save(net.module.state_dict(),'checkpoints/best_predictor_depth.pth')

    torch.save(net.module.state_dict(),'checkpoints/predictor_depth.pth')