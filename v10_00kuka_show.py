import time
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gym

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torchvision.transforms as tran

import torch.multiprocessing as mp
import multiprocessing

import pybullet as p
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces  #

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"############## Agent ##############"


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()
        # Finial Goal
        # conv
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # shared full
        self.shared_full = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # actor
        self.actor_full = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.actor_mu = nn.Sequential(  # action mean range -1 to 1
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        self.actor_sigma = nn.Sequential(
            nn.Linear(32, action_dim),
            nn.Softplus()
        )
        # critic
        self.critic_full = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        # share full
        conv = self.conv(state)
        shared_full = self.shared_full(conv.view(conv.size(0), -1))
        # actor
        action_mean = self.actor_mu(self.actor_full(shared_full))
        cov_mat = torch.diag_embed(self.actor_sigma(self.actor_full(shared_full)))

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(np.squeeze(state.cpu().data.numpy(), axis=0))
        memory.actions.append(np.squeeze(action.cpu().data.numpy(), axis=0))
        memory.logprobs.append(np.squeeze(action_logprob.cpu().data.numpy(), axis=0))

        return action.detach()

    def evaluate(self, state, action):
        # share full
        conv = self.conv(state)
        shared_full = self.shared_full(conv.view(conv.size(0), -1))
        # actor
        action_mean = self.actor_mu(self.actor_full(shared_full))
        cov_mat = torch.diag_embed(self.actor_sigma(self.actor_full(shared_full)))
        # critic
        state_value = self.critic_full(shared_full)

        # Evaluate action
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class Worker(mp.Process):#mp.Process#multiprocessing.Process
    def __init__(self, w_id, action_dim, n_round=64, n_maxSteps=20):
        super(Worker, self).__init__()
        """
        Element:
            w_id*   : classify
            w_ACnet : for action. in cuda. copied from learning net. old policy.
            sub_memory  : save data in class. can be pushed into PPO to train.
            
            n_round*: play n round. default n=64.
            n_maxSteps  : max steps every round
            action_dim  : env action dim. directly use.
            w_env   : every worker has independent env

        Function:
            run     : every worker plays 'n_round' in 'w_env' to collect data. 
        """
        # worker
        self.w_id = w_id
        self.sub_memory = Memory()
        # env
        self.n_round = n_round
        self.n_maxSteps = n_maxSteps
        self.action_dim = action_dim
        # self.w_env = KukaDiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=False,
        #                        maxSteps=n_maxSteps, actionRepeat=60, numObjects=5)
        # # Connect to p.DIRECT
        # self.w_env.cid = p.connect(p.DIRECT)
        # print('%s w_id=%s.  w_ACnet, sub_memory, w_env Built.  Physics engine connected(mode=DIRECT).'
        #       % (time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())), self.w_id))

    # Run n_round. Put data into sub_memory.
    def collect(self, mp_flage, mp_recoder, mp_lock,
                q_actions, q_states, q_logprobs,
                q_rewards, q_is_terminals):
        print('%s %s: Building env.' % (time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())),
                                           self.w_id))
        self.w_ACnet = torch.load('./PPO_continuous_kuka.pkl')
        # print('g_ACnet', g_ACnet)
        # print('self.w_ACnet', self.w_ACnet)
        self.w_env = KukaDiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=False,
                                          maxSteps=self.n_maxSteps, actionRepeat=80, numObjects=2)
        self.w_env.cid = p.connect(p.DIRECT)
        self.w_env.reset()
        print('%s %s: Built env. Phy-Engine Connected.' % (time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())),
                                                           self.w_id))
        # round
        for _ in range(self.n_round):
            self.w_env.reset()
            state = self.w_env._get_observation()
            # action step
            for t in range(self.n_maxSteps):
                # State
                state = trans_state(state)

                # action
                action = self.w_ACnet.act(state, self.sub_memory).cpu().data.numpy().flatten()
                state, reward, done, _ = self.w_env.step(action)

                # Saving reward and is_terminals
                self.sub_memory.rewards.append(reward)
                self.sub_memory.is_terminals.append(done)

                # Recode
                mp_recoder[0] += 1
                mp_recoder[1] += reward

                # without if update
                if done:
                    break

        mp_lock.acquire()
        q_actions.put(self.sub_memory.actions)
        q_states.put(self.sub_memory.states)
        q_logprobs.put(self.sub_memory.logprobs)
        q_rewards.put(self.sub_memory.rewards)
        q_is_terminals.put(self.sub_memory.is_terminals)
        mp_lock.release()
        print('Trans data done. Closing.')

        # bug? can not clear at here
        # self.sub_memory.clear_memory()
        self.w_env.cid = p.disconnect()
        self.w_env.close()

        mp_flage[self.w_id] = 0
        return


class PPO:
    def __init__(self, action_dim, lr, betas, gamma, eps_clip, k_epochs):
        # train parameter
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # AC net
        self.g_ACnet = ActorCritic(action_dim).to(device)
        self.g_ACnet.share_memory()
        self.optimizer = torch.optim.Adam(self.g_ACnet.parameters(), lr=lr, betas=betas)
        self.MseLoss = nn.MSELoss()

        # batch memory
        self.batch_memory = Memory()

    # Update ppo net
    def update(self):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        sum_loss = 0.
        # TD_error = r(S(t)) + v(S(t+1)) - v(S(t))
        # Target = r(S(t)) + v(S(t+1)). Only get Target here.
        for reward, is_terminal in zip(reversed(self.batch_memory.rewards), reversed(self.batch_memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = ((rewards - rewards.mean()) / (rewards.std() + 1e-5)).detach()
        print('Normalized Monte Carlo Rewards:', rewards)

        # Convert list to tensor
        old_states = torch.Tensor(self.batch_memory.states).to(device).detach()
        old_actions = torch.Tensor(self.batch_memory.actions).to(device).detach()
        old_logprobs = torch.Tensor(np.array((self.batch_memory.logprobs))).to(device).detach()


        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.g_ACnet.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            print('mean_loss:%s  actor_loss:%s  critic_loss:%s  entropy_loss:%s'
                  % (torch.mean(loss),
                     torch.mean(-torch.min(surr1, surr2)),
                     torch.mean(0.5 * self.MseLoss(state_values, rewards)),
                     torch.mean(- 0.01 * dist_entropy)))
            sum_loss += torch.mean(loss).detach().cpu().numpy()

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Print loss
        print('\tNet Update. sum_loss = %.4f' % sum_loss)
        sum_loss = 0.


"############## Camera ##############"
# Define 'torchvision transforms'
resize = tran.Compose([tran.ToPILImage(),
                       tran.Resize(40, interpolation=Image.CUBIC),
                       tran.ToTensor()])


# Get Screen
def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    # env.render(mode='human')
    screen = env._get_observation().transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


def trans_state(screen):
    screen = torch.from_numpy(np.ascontiguousarray(screen.transpose((2, 0, 1)), dtype=np.float32) / 255)
    return resize(screen).unsqueeze(0).to(device)


"############## train_agent ##############"


def test_agent():
    print('=================================')
    print('%s Ready to Play.' % (time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
    print('%s BREAK POINT: Click "Play" to Continue.' % (time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))

    # test use
    test_times = 10
    test_result = []

    # env
    episode = 250
    n_maxSteps = 10
    env = KukaDiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=False, maxSteps=20, isTest=False, numObjects=2)
    env.cid = p.connect(p.DIRECT)

    # load the model
    lr = 0.0003
    betas = (0.9, 0.999)
    gamma = 0.75
    eps_clip = 0.07
    k_epochs = 9
    action_dim = env.action_space.shape[0]
    ppo = PPO(action_dim, lr, betas, gamma, eps_clip, k_epochs)
    ppo.g_ACnet = torch.load('./model_v08_22kuka_2parallel.py_BestMean.pkl')

    # Recoder
    recoder = 0

    # test loop
    for _ in range(test_times):
        # evaluate the model
        for e in range(episode):
            env.reset()
            state = env._get_observation()
            # action step
            for t in range(n_maxSteps):
                # State
                state = trans_state(state)

                # action
                action = ppo.g_ACnet.act(state, ppo.batch_memory).cpu().data.numpy().flatten()
                state, reward, done, _ = env.step(action)

                # Recode
                recoder += reward

                # without if update
                if done:
                    break

        print('%s Total score in 100 episode: %s' % (time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())),
                                                     recoder))
        # Append test result
        test_result.append(recoder)
        recoder = 0

    print('%s Test result: %s' % (time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())), test_result))


if __name__ == '__main__':
    test_agent()
