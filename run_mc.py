import numpy as np
import random
import pickle
import argparse
import importlib.util

ACTIONS = ["L45","L22","FW","R22","R45"]
N_ACTIONS = 5
N_STATES = 2**18


def obs_to_state(obs):
    return int("".join(map(str, obs.astype(int))), 2)


def import_env(path):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--out", default="mc_table.pkl")

    args = parser.parse_args()

    OBELIX = import_env(args.obelix_py)

    Q = np.zeros((N_STATES, N_ACTIONS))
    returns = {}
    gamma = 0.99
    epsilon = 0.1

    for ep in range(args.episodes):

        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=500,
            wall_obstacles=False,
            difficulty=0,
            box_speed=2,
            seed=ep
        )
        obs = env.reset(seed=ep)

        episode = []

        done = False

        while not done:

            s = obs_to_state(obs)

            if random.random() < epsilon:
                a = random.randint(0,4)
            else:
                a = np.argmax(Q[s])

            obs2, r, done = env.step(ACTIONS[a], render=False)

            episode.append((s,a,r))

            obs = obs2

        G = 0
        visited = set()

        for s,a,r in reversed(episode):

            G = gamma*G + r

            if (s,a) not in visited:

                visited.add((s,a))

                if (s,a) not in returns:
                    returns[(s,a)] = []

                returns[(s,a)].append(G)

                Q[s,a] = np.mean(returns[(s,a)])

        if (ep+1)%1 == 0:
            print("Episode",ep+1)

    with open(args.out,"wb") as f:
        pickle.dump(Q,f)

    print("Saved Q table")


if __name__ == "__main__":
    main()