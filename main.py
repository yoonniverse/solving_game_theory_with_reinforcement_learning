import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

from envs import ENV_MAPPER
from agent import Agent


def run_env(agents, env):
    with torch.no_grad():
        state, turn = env.reset()
        done = False
        sum_rewards = [0 for _ in range(env.n_agents)]
        actions = [[] for _ in range(env.n_agents)]
        action_probs = [[] for _ in range(env.n_agents)]
        while not done:
            total_action = []
            for i, agent in enumerate(agents):
                turn_bool = 1 if (turn == -1) or (turn == i) else 0
                agent.tmp_buffer['turn_bool'].append(turn_bool)
                value = agent.value(state)
                agent.tmp_buffer['value'].append(value)
                if turn_bool:
                    act_out = agent.act(state)
                    for k in ['action', 'log_prob', 'state']:
                        agent.tmp_buffer[k].append(act_out[k])
                    total_action.append(act_out['action'])
                    actions[i].append(act_out['action'])
                    if (not agent.continuous) and (agent.n_classes == 2):
                        action_probs[i].append(act_out['dist'].probs.squeeze()[0].item())
                else:
                    total_action.append(None)
            state, reward, done, turn = env.step(total_action, turn)
            for i, agent in enumerate(agents):
                agent.tmp_buffer['reward'].append(reward[i])
                agent.tmp_buffer['done'].append(done)
            for i in range(env.n_agents):
                sum_rewards[i] += reward[i]
        for i, agent in enumerate(agents):
            agent.update_buffer()
    return sum_rewards, actions, action_probs


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--global_epochs', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n_episodes_per_update', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--ppo_epochs', type=int, default=3)
    parser.add_argument('--log_std', type=float, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_agents', type=int, default=2)
    parser.add_argument('--n_stages', type=int, default=100)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--vx', type=float, default=1)
    parser.add_argument('--vy', type=float, default=0.2)
    args = parser.parse_args()

    seed_all(args.seed)
    Path('results').mkdir(exist_ok=True)

    # define environment specific keyword arguments
    kwargs = {'n_agents': 2}
    if args.env_name == 'kitty_genovese':
        kwargs['n_agents'] = args.n_agents
        kwargs['v'] = 5
        kwargs['c'] = 3
    elif args.env_name == 'vote_buying':
        kwargs['k'] = args.k
        kwargs['vx'] = args.vx
        kwargs['vy'] = args.vy
    elif args.env_name == 'committee_decision_making':
        kwargs['n_agents'] = 3
    elif args.env_name == 'repeated_prisoners_dilemma':
        kwargs['n_stages'] = args.n_stages

    env = ENV_MAPPER[args.env_name](**kwargs)
    agents = [Agent(
        env.observation_space[i][1],
        env.action_space[i][1],
        args.hidden_size,
        continuous=env.action_space[i][0] == 'cont',
        n_classes=2,
        log_std=args.log_std,
        actor_lr=args.lr,
        critic_lr=args.lr*0.5,
        gamma=0.99,
        lam=0.95,
        buffer_size=int(1e6),
        batch_size=args.batch_size
    ) for i in range(env.n_agents)]

    action_history = [[] for _ in range(env.n_agents)]
    reward_history = [[] for _ in range(env.n_agents)]
    action_prob_history = [[] for _ in range(env.n_agents)]

    for ge in tqdm(range(args.global_epochs)):
        # collect data
        for _ in range(args.n_episodes_per_update):
            state, turn = env.reset()
            reward, action, action_prob = run_env(agents, env)
        # train
        for _ in range(args.ppo_epochs):
            for agent in agents:
                agent.learn()
        # test & log
        test_reward, test_action, test_action_prob = run_env(agents, env)
        for agent in agents:
            agent.reset_buffer()
        for i in range(env.n_agents):
            reward_history[i].append(test_reward[i])
            action_history[i].append(test_action[i])
            action_prob_history[i].append(test_action_prob[i])

    plt.figure(figsize=(20, env.n_agents * 3))
    for i in range(env.n_agents):
        plt.subplot(env.n_agents, 3, i*3+1)
        tmp = reward_history[i]
        plt.title(f'Player{i} ({tmp[-1]:.3f})')
        plt.xlabel('# Episodes')
        plt.ylabel('Sum Rewards')
        plt.plot(tmp)

        plt.subplot(env.n_agents, 3, i*3+2)
        tmp = action_history[i]
        if args.env_name in ['committee_decision_making', 'repeated_prisoners_dilemma']:  # multiple stages, cannot plot history just print last action
            plt.title(f'Player{i}')
            if len(tmp[-1]) == 1:
                plt.plot(tmp[-1], marker='o')
            else:
                plt.plot(tmp[-1])
            plt.xlabel('Last Episode # Stage')
        else:
            plt.title(f'Player{i} ({tmp[-1][0]:.3f})')
            plt.plot(tmp)
            plt.xlabel('# Episodes')
        plt.ylabel('Action')

        plt.subplot(env.n_agents, 3, i * 3 + 3)
        if (not agents[i].continuous) and (agents[i].n_classes == 2):
            tmp = action_prob_history[i]
            if args.env_name in ['committee_decision_making', 'repeated_prisoners_dilemma']:  # multiple stages, cannot plot history just print last action prob
                plt.title(f'Player{i} Playing Action 0')
                if len(tmp[-1]) == 1:
                    plt.plot(tmp[-1], marker='o')
                else:
                    plt.plot(tmp[-1])
                plt.xlabel('Last Episode # Stage')
            else:
                plt.title(f'Player{i} Playing Action 0 ({tmp[-1][0]:.3f})')
                plt.plot(tmp)
                plt.xlabel('# Episodes')
            plt.ylabel('Action Prob')
    str_kwargs = '-'.join([f'{k}_{str(v).replace(".", "_")}' for k, v in kwargs.items()])
    fname = f'{args.env_name}-seed{args.seed}-{str_kwargs}'
    plt.suptitle(fname)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'results/{fname}')

