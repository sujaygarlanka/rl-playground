from collections import defaultdict
from game import TicTacToe
import pickle 
import argparse
import random

class Agent():
    def __init__(self, player, policy):
        self.player = player
        self.policy = policy

    def get_action(self, state):
        curr_state = state + f"-{self.player}"
        # print(self.policy.q_table[curr_state])
        return self.policy.get_action(state + f"-{self.player}")

    
class Policy():
    def __init__(self, dim):
        self.dim = dim
        self.q_table = defaultdict(lambda: [0] * self.dim)
        self.counts_table = defaultdict(lambda: [0] * self.dim)
        self.discount_factor = 0.8

    def update_policy(self, episode, rewards):
        for i in range(len(episode)):
            state, action = episode[i]
            self.counts_table[state][action] += 1

            sub_reward = rewards[i:]
            total_reward = 0
            for idx, r in enumerate(sub_reward):
                total_reward += ((self.discount_factor ** (idx + 1)) * r)
            
            self.q_table[state][action] = self.q_table[state][action] + 1/(self.counts_table[state][action]) * (total_reward - self.q_table[state][action])


    def get_action(self, state):
        return self.q_table[state].index(max(self.q_table[state]))
    
    def save(self, name):
        with open(name, "wb") as file:
            pickle.dump(dict(self.q_table), file) 

    def load(self, name):
        with open(name, "rb") as file:
            self.q_table = defaultdict(lambda: [0] * self.dim, pickle.load(file))


## Training Loop
def train():
    game = TicTacToe()
    policy = Policy(9)
    agents = [Agent("X", policy), Agent("O", policy)]
    for i in range(100000):
        count = 0
        state, done, win = game.reset()
        episodes = [[], []]
        rewards = [[], []]
        epsilon = 1/((1 + i)**0.1)
        while not done:
            player_index = count % 2
            agent = agents[player_index]
            random_number = random.random()
            if random_number < epsilon:
                action = random.randint(0, policy.dim - 1)
            else:
                action = agent.get_action(state)
            next_state, reward, done, win = game.step(action, agent.player)
            episodes[player_index].append((state + f"-{agent.player}", action))
            rewards[player_index].append(reward)
            state = next_state
            count += 1
        # Negative reward for losing
        if win:
            player_index = count % 2
            agent = agents[player_index]
            episodes[player_index].append((state + f"-{agent.player}", 0))
            rewards[player_index].append(reward * -1)
        policy.update_policy(episodes[0], rewards[0])
        policy.update_policy(episodes[1], rewards[1])
        # print("---- X ----")
        # print(episodes[0])
        # print(rewards[0])
        # print("---- O ----")
        # print(episodes[1])
        # print(rewards[1])
        # print(" ")

    policy.save("policy.pkl")

## Game loop
def game():
    game = TicTacToe()
    policy = Policy(9)
    policy.load("policy.pkl")
    agents = [Agent("X", policy), Agent("O", policy)]
    count = 0
    state, done, _ = game.reset()
    game.render()

    while not done:
        player_index = count % 2
        if player_index == 0:
            action = int(input("Action: "))
        else:
            action = agents[player_index].get_action(state)
        # print(action)
        state, _, done, _ = game.step(action, agents[player_index].player)
        game.render()
        count += 1

def test():
    game = TicTacToe()
    game.board = ["X", "O", " ", "X", "O", "O", "O", "X", "X"]
    game.render()
    policy = Policy(9)
    policy.load("policy.pkl")
    state = game._get_state() + "-O"
    action = policy.get_action(state)
    print(action)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play Tic-Tac-Toe with a simple policy.")
    
    parser.add_argument("mode", type=str)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "game":
        game()
    elif args.mode == "test":
        test()
