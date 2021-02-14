from paddle import Paddle

env = Paddle()
while True:
     env.step(0)
     print(env.step(2))

