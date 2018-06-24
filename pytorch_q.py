import math, random

from sabretooth_command import CartCommand
from cart_controller import CartController
from encoder_analyzer import EncoderAnalyzer
import serial.tools.list_ports
import time


#import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from collections import deque

import matplotlib.pyplot as plt


Variable = autograd.Variable

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data.numpy()[0]
            print(q_value.max(1)[1].data.numpy()[0])
        else:
            action = random.randrange(self.num_actions)
        return action
# theta, thetadot, x, xdot = state
# left right = actios
model = DQN(5, 2)

    
optimizer = optim.Adam(model.parameters(), lr=0.0001)

replay_buffer = ReplayBuffer(1000)


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

ports = list(serial.tools.list_ports.comports())
print(dir(ports))
for p in ports:
    print(dir(p))
    print(p.device)
    if "Sabertooth" in p.description:
       sabreport = p.device
    else:
       ardPort = p.device

print("Initilizing Commander")
comm = CartCommand(port= sabreport) #"/dev/ttyACM1")
print("Initilizing Analyzer")
analyzer = EncoderAnalyzer(port=ardPort) #"/dev/ttyACM0")
print("Initializing Controller.")
cart = CartController(comm, analyzer)
time.sleep(0.5)
print("Starting Zero Routine")
cart.zeroAnalyzer()

cart.goTo(500)

batch_size = 32
gamma      = 0.99
speed = 1000

losses = []
all_rewards = []
episode_reward = 0

def get_reward(state):
    x,x_dot,cos_theta, sin_theta ,theta_dot = state
    return -cos_theta + 1.0

def expand_state(state):
    return [state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]

def plot(frame_idx, rewards, losses):
    #clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

i = 0
all_rewards = []
while True:
    i += 1
    state = cart.analyzer.getState()
    state = expand_state(state)
    epsilon = epsilon_by_frame(i)
    action = model.act(state, epsilon)
    
    cart.setSpeedMmPerS(1000 * ((2 * action) -1))
    if not 100 < state[0] < 800:
        cart.setSpeedMmPerS(0)
        continue

    next_state = cart.analyzer.getState()    
    next_state = expand_state(next_state)
    reward = get_reward(next_state)
    print(reward, epsilon, action)
    done = False

    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    all_rewards.append(reward)
        
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.data[0])
        
    if i % 2000 == 0:
        plot(i, all_rewards, losses)
