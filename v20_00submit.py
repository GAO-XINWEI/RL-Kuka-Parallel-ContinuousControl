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



# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def plt_sleep(self):
        plt.ioff()
        plt.show()
        while True:
            plt.pause(8)


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
        # conv
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # shared full
        self.shared_full = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()  # ,
            # nn.Linear(128, 64),
            # nn.ReLU()
        )
        # actor
        self.actor_full = nn.Sequential(
            nn.Linear(128, 32),  # 64, 32
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
            nn.Linear(128, 32),  # 64, 32
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
        '''action dim'''
        # print('actor_full', actor_full)
        # print('action_mean', action_mean)  # action_mean tensor([[-0.0975]], device='cuda:0', grad_fn=<TanhBackward>)
        # print(self.actor_sigma(self.actor_full(self.shared_full(state))))
        # tensor([[0.7744]], device='cuda:0', grad_fn=<SoftplusBackward>)
        # print(torch.diag(self.actor_sigma(self.actor_full(self.shared_full(state)))))
        # If input is a vector (1-D tensor),
        # then returns a 2-D square tensor with the elements of input as the diagonal.
        # If input is a matrix (2-D tensor),
        # then returns a 1-D tensor with the diagonal elements of input.
        # cov_mat = torch.diag(self.actor_sigma(self.actor_full(self.shared_full(state))))
        # print(torch.diag_embed(self.actor_sigma(self.actor_full(self.shared_full(state)))))
        '''parameter passing check(preposition print)'''
        # conv_full = self.conv(state)
        # print('conv_full', conv_full)
        # print('conv_full', conv_full.view(conv_full.size(0), -1))
        ''''''
        # print('shared_full', shared_full)
        # print('actor_full', actor_full)
        # print('action_mean', action_mean)
        # print('cov_mat', cov_mat)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(np.squeeze(state.cpu().data.numpy(), axis=0))
        memory.actions.append(np.squeeze(action.cpu().data.numpy(), axis=0))
        memory.logprobs.append(np.squeeze(action_logprob.cpu().data.numpy(), axis=0))
        '''trans cpu check'''
        # print('memory.states', memory.states)
        # print('torch.tensor(memory.states)', torch.tensor(memory.states))
        # print('memory.action', memory.actions)
        # print('torch.tensor(memory.actions)', torch.tensor(memory.actions))
        # print('memory.logprobs', memory.logprobs)
        # print('torch.tensor(memory.logprobs)', torch.tensor(np.array(memory.logprobs).flatten()))

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
        # self.w_ACnet = ActorCritic(action_dim).to(device)
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
    def collect(self, RUN_PATH, mp_flage, mp_recoder, mp_lock,
                q_actions, q_states, q_logprobs,
                q_rewards, q_is_terminals):
        print('%s %s: Building env.' % (
            time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())),
            self.w_id))
        # self.w_ACnet = torch.load_state_dist(RUN_PATH)
        w_ACnet = torch.load(RUN_PATH)
        # print('self.w_ACnet', self.w_ACnet)
        self.w_env = KukaDiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=False,
                                          maxSteps=self.n_maxSteps, actionRepeat=80, numObjects=2)
        self.w_env.cid = p.connect(p.DIRECT)
        self.w_env.reset()
        # print('%s %s: Built env. Phy-Engine Connected.' % (
        #     time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())),
        #     self.w_id))
        # round
        for _ in range(self.n_round):
            self.w_env.reset()
            state = self.w_env._get_observation()
            # action step
            for t in range(self.n_maxSteps):
                # State
                state = trans_state(state)

                # action
                action = w_ACnet.act(state, self.sub_memory).cpu().data.numpy().flatten()
                state, reward, done, _ = self.w_env.step(action)
                '''check'''
                # print('state', state)
                # print('self.sub_memory.states', self.sub_memory.states)
                # print('reward', reward)
                # print('done', done)
                '''old class ppo version'''
                # action = ppo.select_action(states, self.sub_memory)
                # print('action', action)

                # Saving reward and is_terminals
                self.sub_memory.rewards.append(reward)
                self.sub_memory.is_terminals.append(done)

                # Recode
                mp_recoder[0] += 1
                mp_recoder[1] += reward

                # without if update
                if done:
                    mp_recoder[2] += 1
                    break

        mp_lock.acquire()
        q_actions.put(self.sub_memory.actions)
        q_states.put(self.sub_memory.states)
        q_logprobs.put(self.sub_memory.logprobs)
        q_rewards.put(self.sub_memory.rewards)
        q_is_terminals.put(self.sub_memory.is_terminals)
        mp_lock.release()
        # print('Trans data done. Closing.')

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
        # return
        Sloss = []
        Eloss = []
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        sum_loss = 0.
        # TD_error = r(S(t)) + v(S(t+1)) - v(S(t))
        # Target = r(S(t)) + v(S(t+1)). Only get Target here.
        for reward, is_terminal in zip(reversed(self.batch_memory.rewards), reversed(self.batch_memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward * 5 + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # Test: critic converge.
        # rewards = ((rewards - rewards.mean()) / (rewards.std() + 1e-5)).detach()
        rewards = rewards.detach()
        print('Normalized Monte Carlo Rewards:', rewards)

        # Convert list to tensor
        old_states = torch.Tensor(self.batch_memory.states).to(device).detach()
        old_actions = torch.Tensor(self.batch_memory.actions).to(device).detach()
        old_logprobs = torch.Tensor(np.array((self.batch_memory.logprobs))).to(device).detach()
        '''update tensor check point'''
        # print('memory.states', self.batch_memory.states)
        # print('old_states', old_states)
        # print('memory.actions', self.batch_memory.actions)
        # print('old_actions', old_actions)
        # print('memory.logprobs\t', self.batch_memory.logprobs)
        # print('old_logprobs\t', old_logprobs)

        # Optimize policy for K epochs:
        for i in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.g_ACnet.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss:
            advantages = rewards - state_values
            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1. - self.eps_clip, 1. + self.eps_clip) * advantages.detach()
            # actor_loss
            actor_loss = -1. * torch.min(surr1, surr2)
            # critic_loss
            critic_loss = .5 * self.MseLoss(state_values, rewards)
            # entropy_loss
            entropy_loss = - 0.01 * dist_entropy
            # total loss
            loss = actor_loss + critic_loss + entropy_loss
            # print('mean_loss:%s  actor_loss:%s  critic_loss:%s  entropy_loss:%s' % (torch.mean(loss),
            #                                                                         torch.mean(actor_loss),
            #                                                                         torch.mean(critic_loss),
            #                                                                         torch.mean(entropy_loss)))

            # tensorboard
            # print('train_episodes', train_episodes)
            if i == 0:
                Sloss.append(float(torch.mean(loss).detach().cpu().numpy()))
                Sloss.append(float(torch.mean(actor_loss).detach().cpu().numpy()))
                Sloss.append(float(torch.mean(critic_loss).detach().cpu().numpy()))
                Sloss.append(float(torch.mean(entropy_loss).detach().cpu().numpy()))
                print('mean_loss:%s  actor_loss:%s  critic_loss:%s  entropy_loss:%s' % (torch.mean(loss),
                                                                                        torch.mean(actor_loss),
                                                                                        torch.mean(critic_loss),
                                                                                        torch.mean(entropy_loss)))
            if i == (self.k_epochs - 1):
                Eloss.append(float(torch.mean(loss).detach().cpu().numpy()))
                Eloss.append(float(torch.mean(actor_loss).detach().cpu().numpy()))
                Eloss.append(float(torch.mean(critic_loss).detach().cpu().numpy()))
                Eloss.append(float(torch.mean(entropy_loss).detach().cpu().numpy()))
                print('mean_loss:%s  actor_loss:%s  critic_loss:%s  entropy_loss:%s' % (torch.mean(loss),
                                                                                        torch.mean(actor_loss),
                                                                                        torch.mean(critic_loss),
                                                                                        torch.mean(entropy_loss)))

            sum_loss += torch.mean(loss).detach().cpu().numpy()

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Print loss
        print('\tNet Update. sum_loss = %.4f' % (sum_loss))
        sum_loss = 0.

        return Sloss, Eloss


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


def train_agent():
    # Hyper-parameters
    # workers
    num_workers = 5  # mp.cpu_count()
    sub_round = 50  # every worker collects this env episodes # 25
    # env
    max_timesteps = 10  # max timesteps in one episode
    # train
    max_episodes = 5e6  # max training episodes
    dis_episodes = int(max_episodes / num_workers / sub_round)
    # gradient descent
    lr = 0.0003  # parameters for Adam optimizer # 0.0003
    betas = (0.9, 0.999)  # (0.9, 0.999)
    k_epochs = 10  # update policy for k epochs # 30 # decrease to 10
    # ppo-ac
    eps_clip = 0.04  # clip parameter for PPO  # 0.2 # decrease to 0.07
    gamma = 0.75  # discount factor  # 0.99 # decrease to 0.9

    print('=================================')
    print('torch.cuda.is_available():', torch.cuda.is_available())
    # Create environment
    env = KukaDiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=False,
                               maxSteps=20, actionRepeat=80, numObjects=2)
    env.cid = p.connect(p.DIRECT)
    print('%s Info Env Build.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
    # Reset Env
    env.reset()
    print('%s Info Env:' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))

    # Env Info
    # state_dim = env.observation_space.shape[0]
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    print('\tSize of screen:', screen_height, screen_width)
    action_dim = env.action_space.shape[0]
    print('\tSize of each action:', action_dim)
    # Verify env import
    # plt.figure()
    # plt.imshow(init_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')
    # plt.title('Screen')
    # plt.show()

    # Close env
    env.close()
    print('%s Info Env Closed.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
    print('=================================')

    # Local Recoder
    global train_episodes
    train_episodes = 0
    best_mean_reward = None
    # Board
    board = Board()

    # Generate PPO algorithm
    ppo = PPO(action_dim, lr, betas, gamma, eps_clip, k_epochs)
    print('%s PPO algorithm Built.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
    # model save
    RUN_PATH = './model_{}.pkl'.format(os.path.basename(__file__))
    BEST_PATH = './model_{}_BestMean.pkl'.format(os.path.basename(__file__))
    # torch.save(ppo.g_ACnet.state_dict(), RUN_PATH)
    torch.save(ppo.g_ACnet, RUN_PATH)

    # Generate workers
    workers = [Worker(w_id=i, action_dim=action_dim, n_round=sub_round, n_maxSteps=max_timesteps)
               for i in range(num_workers)]
    # [ppo.cover_weight(worker.w_ACnet) for worker in workers]
    print('%s Worker Built: num = %s.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())), num_workers))

    # Generate 'queue' and 'lock'
    mp_lock = mp.Lock()
    q_actions = mp.Queue()
    q_states = mp.Queue()
    q_logprobs = mp.Queue()
    q_rewards = mp.Queue()
    q_is_terminals = mp.Queue()
    print('%s Multiprocess: mp.queue and mp.lock built.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))

    # training loop
    print('=================================')
    print('%s Ready to train.' % (
        time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
    for i_episode in range(1, dis_episodes + 1):
        # Multi-Recoder
        # [0]: sum steps of ever worker
        # [1]: sum reward cumulate. ([1]/num_workers/sub_round) is percentage.
        # [2]: sum episodes
        mp_recoder = mp.Array('f', np.zeros(3))

        # Processes manage
        mp_flag = mp.Array('f', np.ones(num_workers))
        # print('mp_flag', mp_flag[0])
        processes = []
        # Play 'sub_round' in env(Worker)
        for worker in workers:
            process = mp.Process(target=worker.collect, args=(RUN_PATH, mp_flag, mp_recoder, mp_lock,
                                                              q_actions, q_states, q_logprobs,
                                                              q_rewards, q_is_terminals,))
            processes.append(process)
        [process.start() for process in processes]
        # [process.join() for process in processes]
        while np.sum(mp_flag):
            time.sleep(0.1)
        '''check point'''
        # print('mp_flag', mp_flag[0])
        print('%s Finished workers train.' % (
            time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))

        # Pull data(PPO)
        for _ in workers:
            # print('%s Pulling data.' % (
            #     time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
            ppo.batch_memory.actions.extend(q_actions.get())
            ppo.batch_memory.states.extend(q_states.get())
            ppo.batch_memory.logprobs.extend(q_logprobs.get())
            ppo.batch_memory.rewards.extend(q_rewards.get())
            ppo.batch_memory.is_terminals.extend(q_is_terminals.get())
        [process.terminate() for process in processes]
        print('%s Data Pulled(PPO).' % (
            time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))
        '''multiprocess data check point'''
        # print('actions', torch.Tensor(ppo.batch_memory.actions))
        # print('states', ppo.batch_memory.states)
        # print('logprobs', ppo.batch_memory.logprobs)
        # print('rewards', ppo.batch_memory.rewards)
        # print('is_terminals', ppo.batch_memory.is_terminals)
        print('mp_recoder: step', mp_recoder[0])
        print('mp_recoder: reward', mp_recoder[1])
        print('mp_recoder: episodes', mp_recoder[2])

        # Update(PPO)
        Sloss, Eloss = ppo.update()
        torch.save(ppo.g_ACnet, RUN_PATH)
        print('%s Update net(PPO). Ready to be copied.' % (
            time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))

        # Clear memory after update. Communication problem? could not clear in run.
        ppo.batch_memory.clear_memory()
        [worker.sub_memory.clear_memory() for worker in workers]
        print('%s Update net(PPO). Ready to be copied.' % (
            time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))))

        # Board
        train_episodes += mp_recoder[2]
        print(train_episodes)
        #
        board.pull_data(train_episodes, mp_recoder[1]/mp_recoder[2], Sloss, Eloss)
        board.update()

        # Save model
        if best_mean_reward is None or best_mean_reward < mp_recoder[1]:
            torch.save(ppo.g_ACnet, BEST_PATH)
            if best_mean_reward is not None:
                print("\tBest mean reward updated %.3f -> %.3f, model saved." % (best_mean_reward, mp_recoder[1]))
            best_mean_reward = mp_recoder[1]

    board.plt_sleep()


if __name__ == '__main__':
    train_agent()
