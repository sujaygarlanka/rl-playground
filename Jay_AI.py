import numpy as numpy
from tensorflow import keras
from tensorflow.keras import layers

from paddle import Paddle

inputs = keras.Input(shape=(5))
x = layers.Dense(24)(inputs)
x = layers.Dense(32)(x)
outputs = layers.Dense(2)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()


Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory():
     def __init__(self, capacity):
          self.capacity = capacity
          self.memory = []
          self.push_count = 0

     def push(self, experience):
          if len(self.memory) < self.capacity:
               self.memory.append(experience)
          else:
               self.memory[self.push_count % self.capacity] = experience
               self.push_count += 1

     def sample(self, batch_size):
          return random.sample(self.memory, batch_size)

     def can_provide_sample(self, batch_size):
          return len(self.memory) >= batch_size


class Agent():
    def __init__(self, num_actions):
          self.current_step = 0
          self.num_actions = num_actions

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
               return policy_net(state).argmax(dim=1).item() # exploit  

     def get_exploration_rate(self):
          return self.end + (self.start - self.end) * \
               math.exp(-1. * self.current_step * self.decay)


def train_agent():

     batch_size = 256
     gamma = 0.999

     target_update = 10
     memory_size = 100000
     lr = 0.001
     num_episodes = 1000

     env = Paddle()
     agent = Agent()
     memory = ReplayMemory(memory_size)

     state = env.reset()
     for episode in range(num_episodes):
          action = agent.select_action(state, policy_net)
          reward, next_state, done = env.step(action)
          memory.push(Experience(state, action, next_state, reward))
          state = next_state
# env = Paddle()
# while True:
#      env.step(0)
#      print(env.step(2))

