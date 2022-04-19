import numpy as np
import copy

from net import Actor, Critic

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Agent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, continuous=True, n_classes=2, log_std=-1,
                 actor_lr=1e-2, critic_lr=5e-3, gamma=0.99, lam=0.95, buffer_size=int(1e6), batch_size=10, clip_param=0.2):
        super(Agent, self).__init__()
        """
        Interacts with and learns from the environment.
        :param state_size: (int)
        :param action_size: (int)
        :param actor_lr: (float)
        :param critic_lr: (float)
        :param gamma: (float) discount factor
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.continuous = continuous
        self.n_classes = n_classes
        self.log_std = log_std
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lam = lam
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.clip_param = clip_param

        # actor
        self.actor = Actor(self.state_size, self.hidden_size, self.continuous, self.n_classes, self.log_std).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # critic
        self.critic = Critic(self.state_size, self.hidden_size).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # inter_episode_buffer
        self.tmp_buffer = None
        self.reset_tmp_buffer()

        # ppo buffer
        self.buffer = PPOBuffer(self.state_size, self.buffer_size, self.batch_size)

    def reset_tmp_buffer(self):
        self.tmp_buffer = {
            'action': [],
            'log_prob': [],
            'value': [],
            'state': [],
            'done': [],
            'reward': [],
            'turn_bool': []
        }

    def reset_buffer(self):
        self.buffer.reset()

    def cal_adv_ret(self):
        last_val = 0
        rews = np.append(self.tmp_buffer['reward'], last_val)
        vals = np.append(self.tmp_buffer['value'], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = discount_cumsum(deltas, self.gamma * self.lam)
        ret = discount_cumsum(rews, self.gamma)[:-1]
        return adv, ret

    def update_buffer(self):
        for k, v in self.tmp_buffer.items():
            self.tmp_buffer[k] = np.stack(v)
            self.tmp_buffer[k] = self.tmp_buffer[k].astype(bool) if k == 'turn_bool' else self.tmp_buffer[k].astype(np.float32)
        adv, ret = self.cal_adv_ret()
        e = {
            'state': self.tmp_buffer['state'],
            'action': self.tmp_buffer['action'],
            'log_prob': self.tmp_buffer['log_prob'],
            'return': ret[self.tmp_buffer['turn_bool']],
            'advantage': adv[self.tmp_buffer['turn_bool']]
        }
        self.buffer.add(e)
        self.reset_tmp_buffer()

    def act(self, state):
        """
        produce action from state using actor
        :param state: (np.array) [state_size]
        :return: action (float or int)
        """
        state_tensor = torch.from_numpy(state).to(DEVICE).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            dist = self.actor(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze().item()
            action = action.squeeze().item()
        if self.continuous:
            action = np.clip(action, 0, 1)
        self.actor.train()
        out = {
            'dist': dist,
            'action': action,
            'log_prob': log_prob,
            'state': state
        }
        return out

    def value(self, state):
        state_tensor = torch.from_numpy(state).to(DEVICE).unsqueeze(0)
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(state_tensor).squeeze().item()
        self.critic.train()
        return value

    def learn(self):

        e = self.buffer.sample()
        dist = self.actor(e['state'])
        value = self.critic(e['state'])
        if self.continuous:
            new_log_prob = dist.log_prob(e['action'].unsqueeze(1)).squeeze(1)
        else:
            new_log_prob = dist.log_prob(e['action'])
        ratio = (new_log_prob - e['log_prob']).exp()
        surr1 = ratio * e['advantage']
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * e['advantage']

        actor_loss = -torch.min(surr1, surr2).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic_loss = (e['return'] - value).pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


class PPOBuffer:

    def __init__(self, state_size, buffer_size, batch_size):
        """
        Fixed-size buffer to store experience tuples.
        :param state_size: (int)
        :param buffer_size: (int)
        :param batch_size: (int)
        """
        self.memory = {
            'state': np.zeros((buffer_size, state_size), dtype=np.float32),
            'action': np.zeros(buffer_size, dtype=np.float32),
            'log_prob': np.zeros(buffer_size, dtype=np.float32),
            'return': np.zeros(buffer_size, dtype=np.float32),
            'advantage': np.zeros(buffer_size, dtype=np.float32),
        }
        self.memory_keys = set(self.memory.keys())
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def add(self, experience_dict):
        """
        Add a new experience to memory.
        :param experience_dict: experience dictionary with keys {state, action, log_prob, return, advantage}
        each with shape (batch_size, *)
        """
        len_e = len(experience_dict[list(experience_dict.keys())[0]])
        assert self.memory_keys == set(experience_dict.keys())
        for k in self.memory_keys:
            len_cur_e = len(experience_dict[k])
            assert len_e == len_cur_e
            self.memory[k][self.ptr:self.ptr+len_cur_e] = experience_dict[k]
        self.ptr = (self.ptr+len_e) % self.buffer_size
        self.size = min(self.size+len_e, self.buffer_size)

    def reset(self):
        self.ptr = 0
        self.size = 0

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        idx = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=False)
        out = {k: torch.from_numpy(self.memory[k][idx]).to(device=DEVICE) for k in self.memory_keys}
        out['advantage'] = (out['advantage'] - self.memory['advantage'].mean()) / (self.memory['advantage'].std() + 1e-8)
        return out

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return self.size