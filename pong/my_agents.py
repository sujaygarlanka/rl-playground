import math
import random
import numpy as np
import gym
from tensorflow import keras
from tensorflow.keras import layers

from paddle import Paddle

class Net():

     def __init__(self, learning_rate):
          inputs = keras.Input(shape=(5))
          x = layers.Dense(24, activation='relu')(inputs)
          x = layers.Dense(32, activation='relu')(x)
          outputs = layers.Dense(3, 'linear')(x)
          model = keras.Model(inputs=inputs, outputs=outputs)
          model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
          self.model = model

     def get_model(self):
          return self.model

     def get_summary(self):
          return self.model.summary()

     def feed_forward(self, inputs):
          return self.model(inputs)

class ReplayMemory():
     def __init__(self, capacity):
          self.capacity = capacity
          self.memory = []
          self.push_count = 0

     # experience is [state, action, next_state, reward, done]
     def push(self, experience):
          if len(self.memory) < self.capacity:
               self.memory.append(experience)
          else:
               self.memory[self.push_count % self.capacity] = experience
               self.push_count += 1

     def sample(self, batch_size):
          memories = random.sample(self.memory, batch_size)
          states = np.asarray([mem[0] for mem in memories])
          actions = np.asarray([mem[1] for mem in memories]) 
          next_states = np.asarray([mem[2] for mem in memories])
          rewards = np.asarray([mem[3] for mem in memories])
          dones = np.asarray([mem[4] for mem in memories])
          return states, actions, rewards, next_states, dones
          

     def can_provide_sample(self, batch_size):
          return len(self.memory) >= batch_size


class DQNAgent():
     def __init__(self):
          self.current_step = 0
          self.num_actions = 3

          # Greedy strategy
          self.start = 1
          self.end = 0.01
          self.decay = 0.001

     def select_action(self, state, policy_net):
          rate = self.get_exploration_rate()
          self.current_step += 1

          if rate > random.random():
               return random.randrange(self.num_actions) # explore      
          else:
               q_values = policy_net.feed_forward(state) # exploit 
               return np.argmax(q_values[0]) 

     def get_exploration_rate(self):
          return self.end + (self.start - self.end) * \
               math.exp(-1. * self.current_step * self.decay)

     def train(self):
          batch_size = 100
          gamma = 0.999

          target_update = 10
          memory_size = 1000
          lr = 0.001
          num_episodes = 500
          max_steps_per_episode = 1000

          env = Paddle()
          memory = ReplayMemory(memory_size)
          policy_net = Net(lr)
          target_net = Net(lr)
          copy_step = 25
          steps = 0
          print(policy_net.get_summary())

          for episode in range(num_episodes):
               state = env.reset()
               score = 0
               for step in range(max_steps_per_episode):
                    action = self.select_action(np.asarray([state]), policy_net)
                    reward, next_state, done = env.step(action)
                    score += reward
                    memory.push([state, action, next_state, reward, done])
                    if memory.can_provide_sample(batch_size):
                         states, actions, rewards, next_states, dones = memory.sample(batch_size)
                         next_q_values = np.amax(target_net.feed_forward(next_states), axis=1) * (1-dones)
                         target_q_values = (next_q_values * gamma) + rewards
                         policy_net.get_model().fit(states, target_q_values, epochs=1, verbose=0)
                    state = next_state
                    if steps % copy_step == 0:
                         target_net.get_model().set_weights(policy_net.get_model().get_weights()) 
                    if done:
                         break
                    steps += 1
               print("episode: {}/{}, score: {}".format(episode + 1, num_episodes, score))


class QAgentPong():
     def __init__(self):
          self.current_step = 0
          self.num_actions = 3

          self.q_table = np.zeros((600, 600, 600, 2, 2, num_actions))
          print(self.q_table.size)

          # Greedy strategy
          self.start = 1
          self.end = 0.01
          self.decay = 0.001

     def select_action(self, state):
          rate = self.get_exploration_rate()
          self.current_step += 1

          if rate > random.random():
               return random.randrange(self.num_actions) # explore      
          else:
               x_paddle, x_ball, y_ball, x_vel, y_vel = self._get_q_values_coor(state)
               q_values = self.q_table[x_paddle][x_ball][y_ball][x_vel][y_vel]
               return np.argmax(q_values)
     
     def get_q_value(self, state, action=None):
          x_paddle, x_ball, y_ball, x_vel, y_vel = self._get_q_values_coor(state)
          q_values = self.q_table[x_paddle][x_ball][y_ball][x_vel][y_vel]
          if action == None:
               return np.max(q_values)
          else:
               return q_values[action]

     def set_q_value(self, state, action, new_value):
          x_paddle, x_ball, y_ball, x_vel, y_vel = self._get_q_values_coor(state)
          self.q_table[x_paddle][x_ball][y_ball][x_vel][y_vel][action] = new_value

     def _get_q_values_coor(self, state):
          x_paddle = int(state[0] * 100 + 300)
          x_ball = int(state[1] * 100 + 300)
          y_ball = int(state[2] * 100 + 300)
          if (state[3] < 0):   
               x_vel = 0
          elif (state[3] > 0):   
               x_vel = 1
          if (state[4] < 0):   
               y_vel = 0
          elif (state[4] > 0):   
               y_vel = 1
          return x_paddle, x_ball, y_ball, x_vel, y_vel


     def get_exploration_rate(self):
          return self.end + (self.start - self.end) * \
               math.exp(-1. * self.current_step * self.decay)

     def train(self):
          lr = 0.2
          num_episodes = 1000
          max_steps_per_episode = 5000
          gamma = 0.99

          env = Paddle()
          for episode in range(num_episodes):
               state = env.reset()
               score = 0
               for step in range(max_steps_per_episode):
                    action = self.select_action(state)
                    reward, next_state, done = env.step(action)
                    new_q_value = lr * ((self.get_q_value(next_state) * gamma) + reward) + (1 - lr) * agent.get_q_value(state, action)
                    self.set_q_value(state, action, new_q_value)

                    score += reward
                    state = next_state
                    if done:
                         break
               print("episode: {}/{}, score: {}".format(episode + 1, num_episodes, score))

class QAgentFrozenLake():
     def __init__(self):
          self.current_step = 0
          self.env = gym.make('FrozenLake-v0')
          self.num_states = self.env.observation_space.n
          self.num_actions = self.env.action_space.n

          self.q_table = np.zeros((self.num_states, self.num_actions))

          # Greedy strategy
          self.start = 1
          self.end = 0.01
          self.decay = 0.001

     def select_action(self, state):
          rate = self.get_exploration_rate()
          self.current_step += 1

          if rate > random.random():
               return random.randrange(self.num_actions) # explore      
          else:
               return np.argmax(self.q_table[state])

     def get_exploration_rate(self):
          return self.end + (self.start - self.end) * \
               math.exp(-1. * self.current_step * self.decay)

     def train(self):
          lr = 0.2
          num_episodes = 15000
          max_steps_per_episode = 1000
          gamma = 0.99
          
          average_reward = 0
          for episode in range(num_episodes):
               state = self.env.reset()
               score = 0
               steps = 0
               for step in range(max_steps_per_episode):
                    # self.env.render()
                    action = self.select_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    new_q_value = lr * ((np.max(self.q_table[next_state]) * gamma) + reward) + (1 - lr) * self.q_table[state][action]
                    self.q_table[state][action] = new_q_value
                    score += reward
                    steps += 1
                    state = next_state
                    if done:
                         break
               # print("episode: {}/{}, score: {}, steps: {}".format(episode + 1, num_episodes, score, steps))
               if episode % 1000 == 0:
                    print(average_reward/1000)
                    print("average reward per thousand episodes ending with episode {}: {:.2f}".format(episode, average_reward/1000))
                    average_reward = 0
               else:
                    average_reward += score

          self.env.close()
          
# agent = QAgentPong()
# agent.train()

agent = QAgentFrozenLake()
agent.train()

# env = Paddle()
# while True:
#      env.run_frame()
#      env.step(0)
#      reward, next_state, done = env.step(2)
#      print(next_state)

