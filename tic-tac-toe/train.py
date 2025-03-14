from collections import defaultdict
from game import TicTacToe
import pickle 

class Agent():
    def __init__(self, player, policy):
        self.player = player
        self.policy = policy

    def get_action(self, state):
        return self.policy.get_action(state + f"-{self.player}")

    
class Policy():
    def __init__(self):
        self.q_table = defaultdict(lambda: [0] * 9)
        self.counts_table = defaultdict(lambda: [0] * 9)
        self.discount_factor = 0.8

    def update_policy(self, episode, rewards):
        for i in range(len(episode)):
            state, action = episode[i]
            self.counts_table[state][action] += 1

            sub_reward = rewards[i:]
            total_reward = 0
            for idx, r in enumerate(sub_reward):
                total_reward += (self.discount_factor ** (idx + 1) + r)
            
            self.q_table[state][action] = self.q_table[state][action] + 1/(self.counts_table[state][action]) * total_reward

    def get_action(self, state):
        return self.q_table[state].index(max(self.q_table[state]))
    
    def save(self, name):
        with open(name, "wb") as file:
            pickle.dump(dict(self.q_table), file) 

    def load(self, name):
        with open(name, "rb") as file:
            self.q_table = defaultdict(lambda: [0] * 9, pickle.load(file))


## Training Loop
def train():
    game = TicTacToe()
    policy = Policy()
    agents = [Agent("X", policy), Agent("O", policy)]
    for i in range(1000000):
        print(i)
        count = 0
        state, done = game.reset()
        episodes = [[], []]
        rewards = [[], []]
        while not done:
            player_index = count % 2
            agent = agents[player_index]
            action = agent.get_action(state)
            next_state, reward, done = game.step(action, agent.player)
            episodes[player_index].append((state + f"-{agent.player}", action))
            rewards[player_index].append(reward)
            state = next_state
            count += 1
        policy.update_policy(episodes[0], rewards[0])
        policy.update_policy(episodes[1], rewards[1])
        
        # player_index = count % 2
        # agent = agents[player_index]
        # episodes[player_index].append(next_state + f"-{agent.player}")
        # rewards[player_index].append(reward)
    policy.save("policy.pkl")

## Test loop
def test():
    game = TicTacToe()
    policy = Policy()
    policy.load("policy.pkl")
    # print(policy.q_table)
    agents = [Agent("X", policy), Agent("O", policy)]
    count = 0
    state, done = game.reset()
    game.render()

    while not done:
        player_index = count % 2
        if player_index == 0:
            action = int(input("Action: "))
        else:
            action = agents[player_index].get_action(state)
        # print(action)
        state, _, done = game.step(action, agents[player_index].player)
        game.render()
        count += 1
    
# train()
test()