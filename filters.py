# -*- coding: utf-8 -*-
#https://github.com/jelfring/particle-filter-tutorial/tree/8d1304ec23e634353d108a2795f6c492b330344b
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
#from confidence_ellipse import get_ellipse
from cov_ellipse import get_ellipse

torch.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class KalmanFilter(nn.Module):
    def __init__(self, world, params=ModelParams()):
        super(KalmanFilter, self).__init__()
        '''
        y = z - d -- innovation vector
        S --  innovation vector covriance matrix
        F -- Jacobian of state transition
        P -- state covariance matrix
        Q -- the covariance matrix for the process noise
        H -- measurements Jacobian
        R -- process measurment covariance matrix
        K -- Kalman Gain
        '''
        self.uniform_noise = torch.distributions.uniform.Uniform(low=-1, high=1)
        self.pNumber = params.pNumber
        self.map_size = params.map_size
        self.measurement_num = params.measurement_num
        self.speed = params.speed
        self.sensor_noise = params.sensor_noise
        self.speed_noise = params.speed_noise
        self.theta_noise = params.theta_noise
        self.world = world
        self.P = torch.eye(3).to(device).double()
        self.P[0, 0] = self.P[1, 1] = self.map_size**2 / 3
        self.P[2, 2] = np.pi**2 / 3
        self.R = torch.eye(self.measurement_num).to(device).double() * (self.sensor_noise**2)
        self.x = torch.tensor([[self.map_size / 2, self.map_size / 2, np.pi]]).to(device).double()
        self.F = torch.tensor([[1, 0, -self.speed * torch.sin(self.x[:, 2])], [0, 1, self.speed * torch.cos(self.x[:, 2])], [0, 0, 1]]).to(device).double()
        self.z = torch.zeros(self.measurement_num, 1).to(device).double()
        self.H = torch.zeros(self.measurement_num, 3).to(device).double()
        self.Q = torch.zeros(3, 3).to(device).double()
        sensors = torch.zeros(0, 2).to(device).double()

        self.world = world
        for  y, line in enumerate(self.world):
            for x, block in enumerate(line):
                if block == 2:
                    nb_y = self.map_size - y - 1
                    sensors = torch.cat((sensors, torch.from_numpy(np.array([[x+0.5, nb_y+0.5]]))), axis=0)
        self.sensors = sensors

    def distance(self):
        D = torch.cdist(self.x[:, :-1], self.sensors, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        return D

    def distance_to_sensors(self):
        distances = self.distance()
        distances_sorted, indices = torch.topk(distances, self.measurement_num, axis=1, largest=False)
        return distances_sorted, indices

    def predict(self, delta):
        self.x += delta
        self.x[:, :2][self.x[:, :2] < 0] = 0
        self.x[:, :2][self.x[:, :2] > self.map_size] = self.map_size

        self.F = torch.tensor([[1, 0, -self.speed * torch.sin(self.x[:, 2])], [0, 1, self.speed * torch.cos(self.x[:, 2])], [0, 0, 1]]).to(device).double()

        sx2 = (self.speed_noise**2 / 3) * torch.cos(self.x[:, 2])**2 + (self.speed * (2 * np.pi * self.theta_noise)**2) * torch.sin(self.x[:, 2])**2
        sy2 = (self.speed_noise**2 / 3) * torch.sin(self .x[:, 2])**2 + (self.speed * (2 * np.pi * self.theta_noise)**2) * torch.cos(self.x[:, 2])**2
        sxy = ((self.speed_noise**2 / 3) / 2 + (self.speed * 2 * np.pi * self.theta_noise)**2) * torch.sin(2 * self.x[:, 2])
        sxphi = -self.speed * ((2 * np.pi * self.theta_noise)**2) * torch.sin(self.x[:, 2])
        syphi = self.speed * ((2 * np.pi * self.theta_noise)**2) * torch.cos(self.x[:, 2])
        sphi2 = (2 * np.pi * self.theta_noise)**2
        self.Q = torch.tensor([[sx2, sxy, sxphi], [sxy, sy2, syphi], [sxphi, syphi, sphi2]]).to(device).double()
        self.P = self.F @ (self.P @ self.F.transpose(1, 0)) + self.Q

    def update(self, z):
        d, indices = self.distance_to_sensors()
        self.H[:, 0] = (self.x[:, 0]-self.sensors[indices, 0]) / d
        self.H[:, 1] = (self.x[:, 1]-self.sensors[indices, 1]) / d
        self.H[:, 2] = 0
        y = (z.unsqueeze(0) - d).permute(1,0)
        S = (self.H @ (self.P @ self.H.transpose(1, 0))) + self.R
        K = (self.P @ (self. H.transpose(1, 0) @ torch.inverse(S)))
        self.x = self.x + (K @ y).transpose(1, 0)
        self.P = (torch.eye(3).to(device).double() - K @ self.H) @ self.P
    
    def draw(self, location, step, to_do=False, period=5, save_fig=False):
        if to_do and (step + 1) % period == 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            for i in range(self.world.shape[0]):
                yy = self.world.shape[0] - i - 1
                for j in range(self.world.shape[1]):
                    xx = j
                    if self.world[i, j] == 1.0:
                        r = mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='gray', alpha=0.5)
                        ax.add_patch(r)
                    if self.world[i, j] == 2.0:
                        r = mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='black', alpha=0.5)
                        ax.add_patch(r)
                        #el = mpl.patches.Ellipse((xx+0.5, yy+0.5), 0.2, 0.2, facecolor='black')
                        #ax.add_patch(el)
                        ax.scatter(xx+0.5, yy+0.5, s=1000, color='orange', marker='X')
            #ax.scatter(self.map_size-self.x[:, 0], self.x[:, 1], color='green', s=100, alpha=0.5, label='predicted location')
            plt.xlim(0, self.map_size)
            plt.ylim(0, self.map_size)
            #plt.xlabel('x coordinate', fontsize=18)
            #plt.ylabel('y coordinate', fontsize=18)
            #plt.title('time step ' + str(step+1), fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=18)

            z = ax.scatter(self.map_size-location[0], location[1], color='red', alpha=0.5, s=2000, marker='*', label='true location')
            x0, y0, cov = self.map_size-self.x[:, 0].to('cpu').detach().numpy(), self.x[:, 1].to('cpu').detach().numpy(), self.P.to('cpu').detach().numpy()[:2, :2].T
            el = get_ellipse(x0, y0, cov)
            ax.add_patch(el)
            ax.legend([el, z], ['predicted location', 'true location'], loc='lower left', fontsize=18)
            
            if save_fig:
                plt.savefig(r'D:\\work\python_env\PF\KF_empty\\KF_' + str(step+1)+'.pdf')


class ParticleFilter(nn.Module):
    def __init__(self, world, eff = 0.25, params=ModelParams()):
        self.uniform_noise = torch.distributions.uniform.Uniform(low=-1, high=1)
        super(ParticleFilter, self).__init__()
        self.pNumber = params.pNumber
        self.resamp_alpha = eff
        self.effNumber = eff * self.pNumber
        self.map_size = params.map_size
        self.measurement_num = params.measurement_num
        self.speed = params.speed
        self.sensor_noise = params.sensor_noise
        self.speed_noise = params.speed_noise
        self.theta_noise = params.theta_noise
        self.world = world
        self.measurement_noise_roughening = params.measurement_noise_roughening
        self.x = torch.zeros(self.pNumber, 3).to(device).double()
        self.x = self.init_particles()
        self.p = torch.log(1 / torch.tensor(self.pNumber)) * torch.ones(self.pNumber, 1).to(device).double()
        sensors = torch.zeros(0, 2).to(device).double()
        for  y, line in enumerate(self.world):
            for x, block in enumerate(line):
                if block == 2:
                    nb_y = self.map_size - y - 1
                    sensors = torch.cat((sensors, torch.from_numpy(np.array([[x+0.5, nb_y+0.5]]))), axis=0)
        self.sensors = sensors

    def inside(self, x, y):
        if x < 0 or y < 0 or x > self.map_size or y > self.map_size:
            return False
        return True

    def free(self, x, y):
        if not self.inside(x, y):
            return False
        return self.world[self.map_size - int(y) - 1][int(x)] == 0

    def init_particles(self):
        for i in range(self.pNumber):
            while True:
                x1 = self.map_size * torch.distributions.uniform.Uniform(low=0, high=1).sample(([2])).to(device).double()
                if self.free(x1[0], x1[1]):
                    self.x[i][:2] = x1
                    self.x[i][2:] = 2 * np.pi * torch.distributions.uniform.Uniform(low=0, high=1).sample([1]).to(device).double()
                    break
        return self.x

    def distance(self):
        D = torch.cdist(self.x[:, :-1], self.sensors, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        return D

    def distance_to_sensors(self):
        distances = self.distance()
        distances_sorted, _ = torch.topk(distances, self.measurement_num, axis=1, largest=False)
        return distances_sorted

    def predict(self, delta):
        direction =  deepcopy(self.x[:, 2:])
        delta_direction = 2 * np.pi * self.theta_noise * torch.randn(self.pNumber, 1)
        direction += delta_direction
        dv = self.speed_noise * torch.randn(self.pNumber, 1)
        d_speed_x = dv * torch.cos(direction)
        d_speed_y = dv * torch.sin(direction)
        self.x += delta
        self.x[:, :1] += d_speed_x
        self.x[:, 1:2] += d_speed_y
        self.x[:, 2:] += delta_direction
        self.x[:, :2][self.x[:, :2] < 0] = 0
        self.x[:, :2][self.x[:, :2] > self.map_size] = self.map_size

    def N_eff(self):
        return 1 / (torch.exp(self.p)**2).sum()

    def soft_resampling(self, probs):
        particles = deepcopy(self.x)
        probs = deepcopy(self.p)
        resamp_prob = self.resamp_alpha * torch.exp(probs) + (1 - self.resamp_alpha) * 1 / self.pNumber
        resamp_prob = resamp_prob.view(self.pNumber, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1), num_samples=self.pNumber, replacement=True)
        indices = indices.transpose(1, 0).contiguous()
        flatten_indices = indices.view(-1, 1).squeeze()
        particles_new = particles[flatten_indices]
        
        prob_new = torch.exp(probs.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 - self.resamp_alpha) / self.pNumber)
        prob_new = torch.log(prob_new).view(self.pNumber, -1, 1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)
        self.x = particles_new
        self.p = prob_new

    def hard_resampling(self, probs):
        if self.N_eff() <= self.effNumber:
            indices = torch.multinomial(probs.flatten(), num_samples=self.pNumber, replacement=True)
            self.x = self.x[indices]
            self.p = torch.log(1 / torch.tensor(self.pNumber)) * torch.ones(self.pNumber, 1).to(device).double()
            
    def update(self, z):
        eps = self.measurement_noise_roughening
        predicted_dist = self.distance_to_sensors()
        diff = (predicted_dist - z)
        log_weight = - (diff[:, :2]**2).sum(axis=1) / (2 * (self.sensor_noise**2 + eps**2))
        p_new = self.p + log_weight.unsqueeze(1)
        probs = torch.softmax(p_new, axis=0)
        self.p = torch.log_softmax(p_new, axis=0)
        self.hard_resampling(probs)
        #self.soft_resampling(probs)

    def draw(self, location, step, to_do=False, period=5, save_fig=False):
        if to_do and (step + 1) % period == 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            for i in range(self.world.shape[0]):
                yy = self.world.shape[0] - i - 1
                for j in range(self.world.shape[1]):
                    xx = j
                    if self.world[i, j] == 1.0:
                        r = mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='gray', alpha=0.5)
                        ax.add_patch(r)
                    if self.world[i, j] == 2.0:
                        r = mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='black', alpha=0.5)
                        ax.add_patch(r)
                        #el = mpl.patches.Ellipse((xx+0.5, yy+0.5), 0.2, 0.2, facecolor='black')
                        #ax.add_patch(el)
                        ax.scatter(xx+0.5, yy+0.5, s=1000, color='orange', marker='X')
            ax.scatter(self.map_size-self.x[:, 0], self.x[:, 1], color='green', s=10000*self.p.exp(), alpha=0.5, label='predicted location')
            ax.scatter(self.map_size-location[0], location[1], color='red', alpha=0.5, s=1000, marker='*', label='true location')
            plt.xlim(0, self.map_size)
            plt.ylim(0, self.map_size)
            #plt.xlabel('x coordinate', fontsize=18)
            #plt.ylabel('y coordinate', fontsize=18)
            plt.legend(loc='lower left', fontsize=18)
            #plt.title('time step ' + str(step+1), fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=18)
            if save_fig:
                plt.savefig(r'D:\\work\python_env\PF\SPF_empty\\SPF_' + str(step+1)+'.pdf')


class KalmanParticleFilter(nn.Module):
    def __init__(self, world, eff = 0.25, params=ModelParams()):
        super(KalmanParticleFilter, self).__init__()
        self.uniform_noise = torch.distributions.uniform.Uniform(low=-1, high=1)
        self.pNumber = params.pNumber
        self.effNumber = eff * self.pNumber
        self.resamp_alpha = eff
        self.map_size = params.map_size
        self.measurement_num = params.measurement_num
        self.speed = params.speed
        self.sensor_noise = params.sensor_noise
        self.speed_noise = params.speed_noise
        self.theta_noise = params.theta_noise
        self.world = world
        #self.measurement_noise_roughening = params.measurement_noise_roughening
        self.measurement_noise_roughening = 0.1
        self.x = torch.zeros(self.pNumber, 3).to(device).double()
        self.x = self.init_particles()
        self.w = torch.log(1 / torch.tensor(self.pNumber)) * torch.ones(self.pNumber, 1).to(device).double()
        sensors = torch.zeros(0, 2).to(device).double()
        self.z = torch.zeros(self.measurement_num, 1).to(device).double()
        for  y, line in enumerate(self.world):
            for x, block in enumerate(line):
                if block == 2:
                    nb_y = self.map_size - y - 1
                    sensors = torch.cat((sensors, torch.from_numpy(np.array([[x+0.5, nb_y+0.5]]))), axis=0)
        self.sensors = sensors
        self.R = torch.eye(self.measurement_num).to(device).double() * self.sensor_noise**2
        self.P = torch.zeros(self.pNumber, 3, 3).to(device).double()
        for i in range(self.pNumber):
            self.P[i] = torch.eye(3).to(device).double()
            self.P[i, 0, 0] = self.P[i, 1, 1] = self.map_size**2 #/ self.pNumber
            self.P[i, 2, 2] = np.pi**2 / 3

    def inside(self, x, y):
        if x < 0 or y < 0 or x > self.map_size or y > self.map_size:
            return False
        return True

    def free(self, x, y):
        if not self.inside(x, y):
            return False
        return self.world[self.map_size - int(y) - 1][int(x)] == 0

    def init_particles(self):
        for i in range(self.pNumber):
            while True:
                x1 = self.map_size * torch.distributions.uniform.Uniform(low=0, high=1).sample(([2])).to(device).double()
                if self.free(x1[0], x1[1]):
                    self.x[i][:2] = x1
                    self.x[i][2:] = 2 * np.pi * torch.distributions.uniform.Uniform(low=0, high=1).sample([1]).to(device).double()
                    break
        return self.x

    def distance(self):
        D = torch.cdist(self.x[:, :-1], self.sensors, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        return D

    def distance_to_sensors(self):
        distances = self.distance()
        distances_sorted, indices = torch.topk(distances, self.measurement_num, axis=1, largest=False)
        return distances_sorted, indices
    
    def batch_inverse(self, S):
        eye = S.new_ones(S.size(-1)).diag().expand_as(S)
        b_inv = torch.linalg.solve(S, eye)
        #b_inv1, _ = torch.solve(eye, S)
        return b_inv

    def soft_resampling(self, probs):
        particles = deepcopy(self.x)
        probs = deepcopy(self.w)
        covariance = deepcopy(self.P)
        resamp_prob = self.resamp_alpha * torch.exp(probs) + (1 - self.resamp_alpha) * 1 / self.pNumber
        resamp_prob = resamp_prob.view(self.pNumber, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1), num_samples=self.pNumber, replacement=True)
        indices = indices.transpose(1, 0).contiguous()
        flatten_indices = indices.view(-1, 1).squeeze()
        particles_new = particles[flatten_indices]
        covariance_new = covariance[flatten_indices]
        prob_new = torch.exp(probs.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 - self.resamp_alpha) / self.pNumber)
        prob_new = torch.log(prob_new).view(self.pNumber, -1, 1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)
        self.x = particles_new
        self.w = prob_new
        self.P = covariance_new

    def hard_resampling(self, probs):
        if self.N_eff() <= self.effNumber:
            indices = torch.multinomial(probs.flatten(), num_samples=self.pNumber, replacement=True)
            self.x = self.x[indices]
            self.w = torch.log(1 / torch.tensor(self.pNumber)) * torch.ones(self.pNumber, 1).to(device).double()
            self.P = self.P[indices]

    def predict_update(self, delta, z_true):
        self.x += delta
        self.x[:, :2][self.x[:, :2] < 0] = 0
        self.x[:, :2][self.x[:, :2] > self.map_size] = self.map_size

        tmp = torch.stack([torch.ones(self.pNumber), torch.zeros(self.pNumber), -self.speed * torch.sin(self.x[:, 2]), torch.zeros(self.pNumber), torch.ones(self.pNumber), self.speed * torch.cos(self.x[:, 2]), torch.zeros(self.pNumber), torch.zeros(self.pNumber), torch.ones(self.pNumber)])
        self.F = tmp.permute(1, 0).view(self.pNumber, 3, 3)
       

        sx2 = (self.speed_noise**2 / 3) * torch.cos(self.x[:, 2])**2 + (self.speed * (2 * np.pi * self.theta_noise)**2) * torch.sin(self.x[:, 2])**2
        sy2 = (self.speed_noise**2 / 3) * torch.sin(self .x[:, 2])**2 + (self.speed * (2 * np.pi * self.theta_noise)**2) * torch.cos(self.x[:, 2])**2
        sxy = ((self.speed_noise**2 / 3) / 2 + (self.speed * 2 * np.pi * self.theta_noise)**2) * torch.sin(2 * self.x[:, 2])
        sxphi = -self.speed * ((2 * np.pi * self.theta_noise)**2) * torch.sin(self.x[:, 2])
        syphi = self.speed * ((2 * np.pi * self.theta_noise)**2) * torch.cos(self.x[:, 2])
        sphi2 = (2 * np.pi * self.theta_noise)**2 * torch.ones(self.pNumber)
        tmp = torch.stack([sx2, sxy, sxphi, sxy, sy2, syphi, sxphi, syphi, sphi2]).to(device).double()
        self.Q =  tmp.permute(1, 0).view(self.pNumber, 3, 3)
        self.P = self.F @ (self.P @ self.F.permute(0, 2, 1)) + self.Q

        self.z, indices = self.distance_to_sensors()
        eps = self.measurement_noise_roughening
        coords = self.sensors[indices]
        y = (z_true.unsqueeze(0) - self.z)

        H = torch.zeros(self.pNumber, self.measurement_num, 3).to(device).double()
        H[:, :, 0] = (self.x[:, 0].unsqueeze(1) - coords[:, :, 0]) / self.z
        H[:, :, 1] = (self.x[:, 1].unsqueeze(1) - coords[:, :, 1]) / self.z
        H[:, :, 2] = 0
        S = H @ self.P @ H.permute(0, 2, 1) + self.R
        K = self.P @ (H.permute(0, 2, 1) @ self.batch_inverse(S))
        self.x = self.x + torch.bmm(K, y.unsqueeze(2)).squeeze(2)
        self.P = (torch.eye(3, 3).unsqueeze(0) - torch.bmm(K, H)) @ self.P

        self.z, _ = self.distance_to_sensors()
        diff = (z_true - self.z)
        log_weight = - (diff[:, :2]**2).sum(axis=1) / (2 * (self.sensor_noise**2 + eps**2))
        p_new = self.w + log_weight.unsqueeze(1)
        probs = torch.softmax(p_new, axis=0)
        self.w = torch.log_softmax(p_new, axis=0)
        #self.soft_resampling(probs)
        self.hard_resampling(probs)

    def N_eff(self):
        return 1 / (torch.exp(self.w)**2).sum()      

    def draw(self, location, step, to_do=False, period=5, save_fig=False):
        if to_do and (step + 1) % period == 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            for i in range(self.world.shape[0]):
                yy = self.world.shape[0] - i - 1
                for j in range(self.world.shape[1]):
                    xx = j
                    if self.world[i, j] == 1.0:
                        r = mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='gray', alpha=0.5)
                        ax.add_patch(r)
                    if self.world[i, j] == 2.0:
                        r = mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='black', alpha=0.5)
                        ax.add_patch(r)
                        #el = mpl.patches.Ellipse((xx+0.5, yy+0.5), 0.2, 0.2, facecolor='black')
                        #ax.add_patch(el)
                        ax.scatter(xx+0.5, yy+0.5, s=1000, color='orange', marker='X')
            ax.scatter(self.map_size-self.x[:, 0], self.x[:, 1], color='green', s=10000*self.w.exp(), alpha=0.5, label='predicted location')
            ax.scatter(self.map_size-location[0], location[1], color='red', alpha=0.5, s=1000, marker='*', label='true location')
            plt.xlim(0, self.map_size)
            plt.ylim(0, self.map_size)
            plt.legend(loc='lower left', fontsize=18)
            #plt.xlabel('x coordinate', fontsize=18)
            #plt.ylabel('y coordinate', fontsize=18)
            #plt.title('time step ' + str(step+1), fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=18)
            if save_fig:
                plt.savefig(r'D:\\work\python_env\PF\KPF_empty\\KPF_' + str(step+1)+'.pdf')

