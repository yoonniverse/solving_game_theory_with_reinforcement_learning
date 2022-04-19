import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical


class Actor(nn.Module):
    def __init__(self, state_size, hidden_size, continuous=True, n_classes=2, log_std=-1.):
        super(Actor, self).__init__()
        self.continuous = continuous
        if self.continuous:
            num_outputs = 1
            self.log_std = nn.Parameter(torch.ones(1, num_outputs) * log_std)
        else:
            num_outputs = n_classes
        if hidden_size == 0:
            self.actor = nn.Linear(state_size, num_outputs)
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_outputs),
            )

    def forward(self, x):
        x = self.actor(x)
        if self.continuous:
            std = self.log_std.exp().expand_as(x)
            dist = Normal(torch.sigmoid(x), std)
        else:
            dist = Categorical(torch.softmax(x, dim=-1))
        return dist


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(Critic, self).__init__()

        if hidden_size == 0:
            self.critic = nn.Linear(state_size, 1)
        else:
            self.critic = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

    def forward(self, x):
        value = self.critic(x).squeeze(-1)  # [batch_size]
        return value
