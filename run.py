import subprocess


def dict2args(d):
    out = ''
    for k, v in d.items():
        if v is not None:
            out += f' --{k} {v}'
    return out


def run(d):
    print(d)
    for s in range(3):
        print(f'seed {s}')
        d['seed'] = s
        response = subprocess.call('python main.py' + dict2args(d), shell=True)


if __name__ == "__main__":

    # prisoners dilemma
    run({'env_name': 'prisoners_dilemma', 'global_epochs': 100})

    # quality choice game
    run({'env_name': 'quality_choice_game', 'global_epochs': 100})

    # goldenballs game
    run({'env_name': 'goldenballs_game', 'global_epochs': 100})

    # stag hunt
    run({'env_name': 'stag_hunt', 'global_epochs': 100})

    # hawk_dove
    run({'env_name': 'hawk_dove', 'global_epochs': 100})

    # penalty kick game
    run({'env_name': 'penalty_kick_game', 'global_epochs': 500})

    # kitty genovese
    run({'env_name': 'kitty_genovese', 'global_epochs': 300, 'n_agents': 5})
    run({'env_name': 'kitty_genovese', 'global_epochs': 300, 'n_agents': 20})

    # entry deterrence
    run({'env_name': 'entry_deterrence', 'global_epochs': 100})

    # ultimatum game
    run({'env_name': 'ultimatum_game', 'global_epochs': 300})

    # vote buying
    run({'env_name': 'vote_buying', 'global_epochs': 500, 'vx': 1, 'vy': 0.1})
    run({'env_name': 'vote_buying', 'global_epochs': 500, 'vx': 1, 'vy': 0.9})

    # committee decision making
    run({'env_name': 'committee_decision_making', 'global_epochs': 300})

    # repeated prisoners dilemma
    run({'env_name': 'repeated_prisoners_dilemma', 'global_epochs': 500, 'n_episodes_per_update': 5, 'batch_size': 5})





