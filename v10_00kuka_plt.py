import time
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import gym
import pybullet as p
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torchvision.transforms as tran

import torch.multiprocessing as mp
import multiprocessing

from tensorboardX import SummaryWriter

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


"############## Board ##############"


# plt
class Board:
    def __init__(self):
        self.episodes = []
        #
        self.rewards = []
        #
        self.s_actor_loss = []
        self.s_critic_loss = []
        self.s_entropy_loss = []
        self.s_loss = []
        #
        self.e_actor_loss = []
        self.e_critic_loss = []
        self.e_entropy_loss = []
        self.e_loss = []

        plt.ion()
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['lines.linewidth'] = 0.5

    def pull_data(self, episodes, m_rewards, Sloss, Eloss):
        self.episodes.append(episodes)
        #
        self.rewards.append(m_rewards)
        #
        self.s_actor_loss.append(Sloss[1])
        self.s_critic_loss.append(Sloss[2])
        self.s_entropy_loss.append(Sloss[3])
        self.s_loss.append(Sloss[0])
        #
        self.e_actor_loss.append(Eloss[1])
        self.e_critic_loss.append(Eloss[2])
        self.e_entropy_loss.append(Eloss[3])
        self.e_loss.append(Eloss[0])

    def update(self):
        plt.ion()
        plt.clf()
        # Plt
        plt_actor_loss = plt.subplot(5, 1, 1)
        plt_critic_loss = plt.subplot(5, 1, 2)
        plt_entropy_loss = plt.subplot(5, 1, 3)
        plt_loss = plt.subplot(5, 1, 4)
        plt_rewards = plt.subplot(5, 1, 5)

        # Title
        plt_actor_loss.set_ylabel('actor_loss', fontsize=10)
        plt_critic_loss.set_ylabel('critic_loss', fontsize=10)
        plt_entropy_loss.set_ylabel('entropy_loss', fontsize=10)
        plt_loss.set_ylabel('loss', fontsize=10)
        plt_rewards.set_ylabel('rewards', fontsize=10)

        #
        plt_rewards.plot(self.episodes, self.rewards, 'g-')
        #
        # print('self.episodes', self.episodes)
        # print('self.s_actor_loss', self.s_actor_loss)
        plt_actor_loss.plot(self.episodes, self.s_actor_loss, 'b-')
        plt_critic_loss.plot(self.episodes, self.s_critic_loss, 'b-')
        plt_entropy_loss.plot(self.episodes, self.s_entropy_loss, 'b-')
        plt_loss.plot(self.episodes, self.s_loss, 'b-')
        #
        plt_actor_loss.plot(self.episodes, self.e_actor_loss, 'r-')
        plt_critic_loss.plot(self.episodes, self.e_critic_loss, 'r-')
        plt_entropy_loss.plot(self.episodes, self.e_entropy_loss, 'r-')
        plt_loss.plot(self.episodes, self.e_loss, 'r-')

        np.save('plt_data.npy', np.array([self.episodes, self.rewards,
                                          self.s_actor_loss, self.s_critic_loss, self.s_entropy_loss, self.s_loss,
                                          self.e_actor_loss, self.e_critic_loss, self.e_entropy_loss, self.e_loss]))

        plt.pause(0.2)

    def load(self):
        plt_data = np.load('plt_data.npy')
        print(plt_data)
        self.episodes = plt_data[0]
        self.rewards = plt_data[1]
        self.s_actor_loss = plt_data[2]
        self.s_critic_loss = plt_data[3]
        self.s_entropy_loss = plt_data[4]
        self.s_loss = plt_data[5]
        self.e_actor_loss = plt_data[6]
        self.e_critic_loss = plt_data[7]
        self.e_entropy_loss = plt_data[8]
        self.e_loss = plt_data[9]


    def plt_sleep(self):
        plt.ioff()
        plt.show()
        while True:
            plt.pause(8)


def plt_show():
    board = Board()
    board.load()
    board.update()
    board.plt_sleep()

if __name__ == '__main__':
    plt_show()
