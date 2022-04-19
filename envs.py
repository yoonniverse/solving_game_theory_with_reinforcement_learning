import gym
import numpy as np


class PrisonersDilemma:

    def __init__(self, **kwargs):
        self.n_agents = kwargs['n_agents']
        assert self.n_agents == 2
        self.action_space = [('disc', 2), ('disc', 2)]
        self.observation_space = [('disc', 1), ('disc', 1)]

    def step(self, action, turn):
        payoff_matrix_1 = np.array([[2, 0], [3, 1]])
        payoff_matrix_2 = np.array([[2, 3], [0, 1]])
        action_1, action_2 = action
        payoff_1 = payoff_matrix_1[action_1, action_2]
        payoff_2 = payoff_matrix_2[action_1, action_2]
        return np.array([0], dtype=np.float32), [payoff_1, payoff_2], True, None

    def reset(self):
        return np.array([0], dtype=np.float32), -1


class QualityChoiceGame:

    def __init__(self, **kwargs):
        self.n_agents = kwargs['n_agents']
        assert self.n_agents == 2
        self.action_space = [('disc', 2), ('disc', 2)]
        self.observation_space = [('disc', 1), ('disc', 1)]

    def step(self, action, turn):
        payoff_matrix_1 = np.array([[2, -1], [0, 0]])
        payoff_matrix_2 = np.array([[2, 3], [0, 1]])
        action_1, action_2 = action
        payoff_1 = payoff_matrix_1[action_1, action_2]
        payoff_2 = payoff_matrix_2[action_1, action_2]
        return np.array([0], dtype=np.float32), [payoff_1, payoff_2], True, None

    def reset(self):
        return np.array([0], dtype=np.float32), -1


class GoldenballsGame:

    def __init__(self, **kwargs):
        self.n_agents = kwargs['n_agents']
        assert self.n_agents == 2
        self.action_space = [('disc', 2), ('disc', 2)]
        self.observation_space = [('disc', 1), ('disc', 1)]

    def step(self, action, turn):
        payoff_matrix_1 = np.array([[1, 0], [2, 0]])
        payoff_matrix_2 = np.array([[1, 2], [0, 0]])
        action_1, action_2 = action
        payoff_1 = payoff_matrix_1[action_1, action_2]
        payoff_2 = payoff_matrix_2[action_1, action_2]
        return np.array([0], dtype=np.float32), [payoff_1, payoff_2], True, None

    def reset(self):
        return np.array([0], dtype=np.float32), -1


class StagHunt:

    def __init__(self, **kwargs):
        self.n_agents = kwargs['n_agents']
        assert self.n_agents == 2
        self.action_space = [('disc', 2), ('disc', 2)]
        self.observation_space = [('disc', 1), ('disc', 1)]

    def step(self, action, turn):
        payoff_matrix_1 = np.array([[9, 0], [7, 6]])
        payoff_matrix_2 = np.array([[9, 7], [0, 6]])
        action_1, action_2 = action
        payoff_1 = payoff_matrix_1[action_1, action_2]
        payoff_2 = payoff_matrix_2[action_1, action_2]
        return np.array([0], dtype=np.float32), [payoff_1, payoff_2], True, None

    def reset(self):
        return np.array([0], dtype=np.float32), -1


class HawkDove:

    def __init__(self, **kwargs):
        self.n_agents = kwargs['n_agents']
        assert self.n_agents == 2
        self.action_space = [('disc', 2), ('disc', 2)]
        self.observation_space = [('disc', 1), ('disc', 1)]

    def step(self, action, turn):
        payoff_matrix_1 = np.array([[-1, 4], [0, 2]])
        payoff_matrix_2 = np.array([[-1, 0], [4, 2]])
        action_1, action_2 = action
        payoff_1 = payoff_matrix_1[action_1, action_2]
        payoff_2 = payoff_matrix_2[action_1, action_2]
        return np.array([0], dtype=np.float32), [payoff_1, payoff_2], True, None

    def reset(self):
        return np.array([0], dtype=np.float32), -1


class PenaltyKickGame:

    def __init__(self, **kwargs):
        self.n_agents = kwargs['n_agents']
        assert self.n_agents == 2
        self.action_space = [('disc', 2), ('disc', 2)]
        self.observation_space = [('disc', 1), ('disc', 1)]

    def step(self, action, turn):
        payoff_matrix_1 = np.array([[2, 9], [8, 3]])
        payoff_matrix_2 = np.array([[8, 1], [2, 7]])
        action_1, action_2 = action
        payoff_1 = payoff_matrix_1[action_1, action_2]
        payoff_2 = payoff_matrix_2[action_1, action_2]
        return np.array([0], dtype=np.float32), [payoff_1, payoff_2], True, None

    def reset(self):
        return np.array([0], dtype=np.float32), -1


class KittyGenovese:

    def __init__(self, **kwargs):
        self.n_agents = kwargs['n_agents']
        self.action_space = [('disc', 2) for _ in range(self.n_agents)]
        self.observation_space = [('disc', 1) for _ in range(self.n_agents)]
        self.v = kwargs['v']
        self.c = kwargs['c']

    def step(self, action, turn):
        action = np.array(action)
        if any([a == 1 for a in action]):
            payoffs = np.array([self.v for _ in range(self.n_agents)])
            payoffs[np.where(action == 1)[0]] -= self.c
        else:
            payoffs = [0 for _ in range(self.n_agents)]
        return np.array([0], dtype=np.float32), payoffs, True, None

    def reset(self):
        return np.array([0], dtype=np.float32), -1


class EntryDeterrence:

    def __init__(self, **kwargs):
        self.n_agents = kwargs['n_agents']
        assert self.n_agents == 2
        self.action_space = [('disc', 2), ('disc', 2)]
        self.observation_space = [('disc', 1), ('disc', 1)]

    def step(self, action, turn):
        assert (turn != -1) and (turn is not None)
        action = action[turn]
        if turn == 0:
            obs = np.array([action], dtype=np.float32)
            payoffs = [0, 0]
            done = False
            next_turn = 1
            if action == 0:
                self.payoff_matrix_1 = [1, -1]
                self.payoff_matrix_2 = [1, -1]
            else:
                self.payoff_matrix_1 = [0, 0]
                self.payoff_matrix_2 = [2, 2]
        else:
            obs = np.array([0], dtype=np.float32)
            payoffs = [self.payoff_matrix_1[action], self.payoff_matrix_2[action]]
            done = True
            next_turn = None
        return obs, payoffs, done, next_turn

    def reset(self):
        return np.array([0], dtype=np.float32), 0


class UltimatumGame:

    def __init__(self, **kwargs):
        self.n_agents = kwargs['n_agents']
        assert self.n_agents == 2
        super(UltimatumGame, self).__init__()
        self.action_space = [('cont', 1), ('disc', 2)]
        self.observation_space = [('disc', 1), ('cont', 1)]

    def step(self, action, turn):
        assert (turn != -1) and (turn is not None)
        action = action[turn]
        if turn == 0:
            obs = np.array([action], dtype=np.float32)
            payoffs = [0, 0]
            done = False
            next_turn = 1
            self.payoff_matrix_1 = np.array([action, 0], dtype=np.float32)
            self.payoff_matrix_2 = np.array([1-action, 0], dtype=np.float32)
        else:
            obs = np.array([0], dtype=np.float32)
            payoffs = [self.payoff_matrix_1[action], self.payoff_matrix_2[action]]
            done = True
            next_turn = None
        return obs, payoffs, done, next_turn

    def reset(self):
        return np.array([0], dtype=np.float32), 0


class VoteBuying:

    def __init__(self, **kwargs):
        super(VoteBuying, self).__init__()
        self.n_agents = kwargs['n_agents']
        assert self.n_agents == 2
        self.action_space = [('cont', 1), ('cont', 1)]
        self.observation_space = [('cont', 1), ('cont', 1)]
        self.k = kwargs['k']
        self.vx = kwargs['vx']
        self.vy = kwargs['vy']

    def step(self, action, turn):
        assert (turn != -1) and (turn is not None)
        action = action[turn]
        if turn == 0:
            obs = np.array([action], dtype=np.float32)
            payoffs = [-action * self.k, 0]
            done = False
            next_turn = 1
            self.action_x = action
        else:
            obs = np.array([0], dtype=np.float32)
            if action > self.action_x:
                payoffs = [0, self.vy - action * np.ceil(self.k/2)]
            else:
                payoffs = [self.vx, -action * np.ceil(self.k/2)]
            done = True
            next_turn = None
        return obs, payoffs, done, next_turn

    def reset(self):
        return np.array([0], dtype=np.float32), 0


class CommitteeDecisionMaking:

    def __init__(self, **kwargs):
        self.n_agents = kwargs['n_agents']
        assert self.n_agents == 3
        self.action_space = [('disc', 2) for _ in range(self.n_agents)]
        self.observation_space = [('disc', 1) for _ in range(self.n_agents)]
        self.stage = 0

    def step(self, action, turn):
        if self.stage == 0:
            if (np.array(action) == 1).sum() >= 2:  # vote for x
                obs = np.array([0], dtype=np.float32)
                payoffs = [2, 0, 1]
                done = True
            else:  # x dropped
                self.stage = 1
                obs = np.array([1], dtype=np.float32)
                payoffs = [0, 0, 0]
                done = False
        elif self.stage == 1:
            obs = np.array([0], dtype=np.float32)
            done = True
            if (np.array(action) == 1).sum() >= 2:  # vote for y
                payoffs = [1, 2, 0]
            else:  # vote for z
                payoffs = [0, 1, 2]
        else:
            raise RuntimeError
        return obs, payoffs, done, -1

    def reset(self):
        self.stage = 0
        return np.array([0], dtype=np.float32), -1


class RepeatedPrisonersDilemma:

    def __init__(self, **kwargs):
        self.n_agents = kwargs['n_agents']
        self.n_stages = kwargs['n_stages']
        assert self.n_agents == 2
        self.action_space = [('disc', 2), ('disc', 2)]
        self.observation_space = [('disc', 2), ('disc', 2)]
        self.stage = 0

    def step(self, action, turn):
        payoff_matrix_1 = np.array([[2, 0], [3, 1]])
        payoff_matrix_2 = np.array([[2, 3], [0, 1]])
        action_1, action_2 = action
        payoff_1 = payoff_matrix_1[action_1, action_2]
        payoff_2 = payoff_matrix_2[action_1, action_2]
        self.stage += 1
        done = 1 if self.stage >= self.n_stages else 0
        return np.array(action, dtype=np.float32), [payoff_1, payoff_2], done, -1

    def reset(self):
        self.stage = 0
        return np.array([0, 0], dtype=np.float32), -1


ENV_MAPPER = {
    'prisoners_dilemma': PrisonersDilemma,
    'quality_choice_game': QualityChoiceGame,
    'goldenballs_game': GoldenballsGame,
    'stag_hunt': StagHunt,
    'hawk_dove': HawkDove,
    'penalty_kick_game': PenaltyKickGame,
    'kitty_genovese': KittyGenovese,
    'entry_deterrence': EntryDeterrence,
    'ultimatum_game': UltimatumGame,
    'vote_buying': VoteBuying,
    'committee_decision_making': CommitteeDecisionMaking,
    'repeated_prisoners_dilemma': RepeatedPrisonersDilemma
}
