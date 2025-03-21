from game import SimpleTicTacToe
from train import Policy, Agent
import argparse
import random

## Training Loop
def train():
    game = SimpleTicTacToe()
    policy = Policy(3)
    agents = [Agent("X", policy), Agent("O", policy)]
    for i in range(10000):
        count = 0
        state, done, win = game.reset()
        episodes = [[], []]
        rewards = [[], []]
        epsilon = 1/((1 + i)**0.5)
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

    # from IPython import embed; embed()
    print(policy.counts_table)
    policy.save("policy_simple.pkl")

def fake_train():
    game = SimpleTicTacToe()
    policy = Policy(3)
    # agents = [Agent("X", policy), Agent("O", policy)]
    # count = 0
    state, done, win = game.reset()
    episodes = [[('   -X', 0), ('X O-X', 1)], [('  X-O', 0)]]
    rewards = [[0, 10], [0]]
    # policy.update_policy(episodes[0], rewards[0])
    print(policy.q_table)
    policy.update_policy(episodes[1], rewards[1])
    print(policy.q_table)

## Game loop
def game():
    game = SimpleTicTacToe()
    policy = Policy(3)
    policy.load("policy_simple.pkl")
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

        state, _, done, _ = game.step(action, agents[player_index].player)
        game.render()
        count += 1

def test():
    game = SimpleTicTacToe()
    game.board = ["X"," "," "]
    game.render()
    policy = Policy(3)
    policy.load("policy_simple.pkl")
    state = game._get_state() + "-O"
    action = policy.get_action(state)
    # from IPython import embed; embed()

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
    else:
        fake_train()
