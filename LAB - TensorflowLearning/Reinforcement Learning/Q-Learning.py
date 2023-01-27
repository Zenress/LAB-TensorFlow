#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

import gym
import numpy as np
import time
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES,ACTIONS))

EPISODES = 1500 #How many times to run the enviroment from the beginning
MAX_STEPS = 100 #Max number of steps allowed for each run of enviroment

LEARNING_RATE = 0.81 #Learning rate
GAMMA = 0.96

RENDER = False #If you want to see training set to true

epsilon = 0.9

rewards = []
for episode in range(EPISODES):
  state = env.reset()

  for _ in range(MAX_STEPS):
    if RENDER:
      env.render()
    if np.random.uniform(0,1) < epsilon:
      action = env.action_space.sample()
    else:
      action = np.argmax(Q[state, :])

    next_state, reward, done, _ = env.step(action)
    Q[state,action] = Q[state,action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state,action])

    state = next_state

    if done:
      rewards.append(reward)
      epsilon -= 0.001
      break #reached goal
 
def get_average(values):
  return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
  avg_rewards.append(get_average(rewards[i:i+100]))

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()