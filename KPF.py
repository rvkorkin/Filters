# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:02:47 2022

@author: RKorkin
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:09:58 2022

@author: RKorkin
"""
import torch
import torch.nn as nn
from dataset import LocalizationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from ModelParams import ModelParams
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from copy import deepcopy
from filters import KalmanParticleFilter
import time


np.random.seed(ModelParams().random_seed)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

num_tracks = 10000
track_data = np.array(pd.read_csv('trajectories.csv', header=None))
world = np.loadtxt('environment.csv', delimiter=',')
all_data = dict()

track_len = track_data.shape[0] // num_tracks
track = np.zeros((num_tracks, track_len, track_data.shape[1]))
for i in range(num_tracks):
    track[i] = track_data[i*track_len:(i+1)*track_len]

del track_data
eval_test_numbers = np.random.choice(num_tracks, size=num_tracks//5, replace=False)
eval_numbers = eval_test_numbers[:len(eval_test_numbers)//2]
test_numbers = eval_test_numbers[len(eval_test_numbers)//2:]
train_numbers = np.setdiff1d(np.arange(num_tracks), eval_test_numbers)

train_data = dict()
eval_data = dict()
test_data = dict()

train_data['tracks'] = track[train_numbers]
train_data['map'] = world

eval_data['tracks'] = track[eval_numbers]
eval_data['map'] = world

test_data['tracks'] = track[test_numbers]
test_data['map'] = world


train_dataset = LocalizationDataset(train_data)
eval_dataset = LocalizationDataset(eval_data)
test_dataset = LocalizationDataset(test_data)

params = ModelParams()

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, pin_memory=True, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=params.batch_size, pin_memory=True, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=True)

params = ModelParams()

arr = np.zeros((track_len, 4))
plt.close('all')


int_loss = dict()
fin_loss = dict()
for current_length in np.arange(10, 60, 10):
    int_loss[current_length] = []
    fin_loss[current_length] = []

t1 = time.time()
for data in test_loader:
    curr_loss = 0
    model = KalmanParticleFilter(world, eff = 0.25).to(device).double()
    _, measurement, location, motion = data
    measurement = measurement.to(device).double().squeeze(0)
    location = location.to(device).double().squeeze(0)
    motion = motion.to(device).double().squeeze(0)
    '''
    coeff = 1
    d_loc = coeff * torch.randn(model.pNumber, 3).to(device).double()
    d_loc[:, -1] *= np.pi
    d_loc[:, -1][d_loc[:, -1]<0] = 0
    d_loc[:, -1][d_loc[:, -1]>2*np.pi] = 2*np.pi
    model.x = location[0].unsqueeze(0) + d_loc
    model.x[:, :2][model.x[:, :2]<0] = 0
    model.x[:, :2][model.x[:, :2]>model.map_size] = model.map_size
    model.P[:, 0, 0] = model.P[:, 1, 1] = coeff**2
    '''
    for step in range(current_length):
        model.draw(location[step], step, to_do=True, period=1, save_fig=False)
        model.predict_update(motion[step], measurement[step])
        predicted_coord = (model.x * torch.exp(model.w)).sum(axis=0)
        loss = ((predicted_coord[:2] - location[step][:2])**2).sum()
        arr[step][:2] = predicted_coord[:2]
        arr[step][2:] = location[step][:2]
        curr_loss += loss / current_length
        if (step + 1) % 10 == 0:
            int_loss[step+1].append(deepcopy(curr_loss*current_length/(step+1)))
            fin_loss[step+1].append(loss**0.5)
    break
for current_length in np.arange(10, 110, 10):
    int_loss[current_length] = np.array(int_loss[current_length])
    fin_loss[current_length] = np.array(fin_loss[current_length])
    print(np.array(int_loss[current_length]).mean(), int_loss[current_length].std(), fin_loss[current_length].mean(), fin_loss[current_length].std())

t2 = time.time()
print('time for ', model.pNumber, ' particles is: ', t2-t1, ' sec')
