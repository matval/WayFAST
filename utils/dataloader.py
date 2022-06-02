import os
import random

import cv2
import numpy as np
import pandas as pd

import torch.utils.data as DataLoader

class TraversabilityDataset(DataLoader.Dataset):
    def __init__(self, params, transform):
        print("Initializing dataset")
        self.root           = params.data_path
        self.transform      = transform
        self.output_size    = params.output_size
        self.image_size     = params.input_size
        self.depth_mean     = params.depth_mean
        self.depth_std      = params.depth_std
        self.bin_width      = 0.2
        
        # Read lines in csv file
        self.data = pd.read_csv(params.csv_path)

        # Prepare data and get max and min values
        self.color_fname, self.depth_fname, self.path_fname, self.mu_fname, self.nu_fname = self.prepare(self.data)

        self.weights, self.bins = self.prepare_weights()

        # Print depth statistics
        self.get_depth_stats()

        self.preproc = params.preproc

    def __getitem__(self, idx):
        # Get current data
        color_fname = self.color_fname[idx]
        depth_fname = self.depth_fname[idx]
        path_fname = self.path_fname[idx]
        mu_fname = self.mu_fname[idx]
        nu_fname = self.nu_fname[idx]

        color_img = cv2.imread(os.path.join(self.root, color_fname),-1)
        depth_img = cv2.imread(os.path.join(self.root, depth_fname),-1)
        path_img = cv2.imread(os.path.join(self.root, path_fname),-1)
        mu_img = cv2.imread(os.path.join(self.root, mu_fname),-1)
        nu_img = cv2.imread(os.path.join(self.root, nu_fname),-1)

        if self.preproc:
            color_img, depth_img, path_img, mu_img, nu_img = self.random_flip(color_img, depth_img, path_img, mu_img, nu_img)

        # because pytorch pretrained model uses RGB
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img = cv2.resize(color_img, self.image_size, interpolation = cv2.INTER_AREA)
        color_img = self.transform(color_img)

        # Convert depth to meters
        depth_img = cv2.resize(depth_img, self.image_size, interpolation = cv2.INTER_AREA)
        depth_img = np.uint16(depth_img)
        depth_img = depth_img*10**-3
        # Normalize depth image
        depth_img = (depth_img-self.depth_mean)/self.depth_std
        depth_img = np.expand_dims(depth_img, axis=2)
        depth_img = np.transpose(depth_img, (2, 0, 1))

        mu_img = cv2.resize(mu_img, self.output_size, interpolation = cv2.INTER_AREA)
        nu_img = cv2.resize(nu_img, self.output_size, interpolation = cv2.INTER_AREA)
        path_img = cv2.resize(path_img, self.output_size, interpolation = cv2.INTER_AREA)

        # expand first dimension
        mu_img = np.expand_dims(mu_img, 0)
        nu_img = np.expand_dims(nu_img, 0)
        path_img = np.expand_dims(path_img, 0)

        mu_img = mu_img/255.0
        nu_img = nu_img/255.0
        path_img = (path_img/255.0).astype(bool)

        weight_idxs = np.digitize(mu_img, self.bins[:-1]) - 1
        weight = self.weights[weight_idxs]*path_img

        return color_img, depth_img, path_img, mu_img, nu_img, weight

    def __len__(self):
        return len(self.data)

    def prepare(self, data):
        color_fname_list = []
        depth_fname_list = []
        path_fname_list = []
        mu_fname_list = []
        nu_fname_list = []

        for color_fname, depth_fname, path_fname, mu_fname, nu_fname, _, _, _, _, _, _ in data.iloc:
            # Append values to lists
            color_fname_list.append(color_fname)
            depth_fname_list.append(depth_fname)
            path_fname_list.append(path_fname)
            mu_fname_list.append(mu_fname)
            nu_fname_list.append(nu_fname)

        return color_fname_list, depth_fname_list, path_fname_list, mu_fname_list, nu_fname_list

    def random_flip(self, color_img, depth_img, path_img, mu_img, nu_img):
        # Augment data with a random horizontal image flip
        if random.random() < 0.5:
            color_img_lr = np.fliplr(color_img).copy()
            depth_img_lr = np.fliplr(depth_img).copy()
            path_img_lr = np.fliplr(path_img).copy()
            mu_img_lr = np.fliplr(mu_img).copy()
            nu_img_lr = np.fliplr(nu_img).copy()
            return color_img_lr, depth_img_lr, path_img_lr, mu_img_lr, nu_img_lr

        return color_img, depth_img, path_img, mu_img, nu_img

    def prepare_weights(self):
        labels_data = []
        for idx in range(len(self.mu_fname)):
            mu_fname = self.mu_fname[idx]
            path_fname = self.path_fname[idx]

            mu_img = cv2.imread(os.path.join(self.root, mu_fname),-1)
            mu_img = cv2.resize(mu_img, self.output_size, interpolation = cv2.INTER_AREA)
            mu_img = mu_img/255.0

            path_img = cv2.imread(os.path.join(self.root, path_fname),-1)
            path_img = cv2.resize(path_img, self.output_size, interpolation = cv2.INTER_AREA)
            path_img = (path_img/255.0).astype(bool)
   
            data_image = mu_img[path_img]
            labels_data.extend(data_image.flatten().tolist())

        # Draw the plot
        values, bins = np.histogram(labels_data, bins = int(1/self.bin_width), range=(0,1), density=True)

        return (1-values*self.bin_width), bins

    def get_depth_stats(self):
        psum = 0.0
        psum_sq = 0.0
        for idx in range(len(self.depth_fname)):
            depth_fname = self.depth_fname[idx]

            depth_img = cv2.imread(os.path.join(self.root, depth_fname),-1)
            depth_img = depth_img*1e-3
            psum += np.sum(depth_img)
            psum_sq += np.sum(depth_img**2)

        count = len(self.depth_fname)*depth_img.shape[0]*depth_img.shape[1]
        total_mean = psum/count
        total_std = np.sqrt(psum_sq / count - (total_mean ** 2))

        print('Depth mean:', total_mean)
        print('Depth std:', total_std)