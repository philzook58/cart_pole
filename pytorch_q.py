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

#plt.ion()

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
            nn.Linear(num_inputs, 4),
            nn.ReLU(),
            #nn.Linear(2, 2),
            #nn.ReLU(),
            nn.Linear(4, num_actions)
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
model = DQN(2, 2)

    
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

replay_buffer = ReplayBuffer(5000)


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

    for param in model.parameters():
        print(param.data)
    
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
gamma      = 0.95
speed = 1000

losses = []
all_rewards = []
episode_reward = 0

def get_reward(state):
    x = state[0]#,x_dot,cos_theta, sin_theta ,theta_dot = state
    '''
    if state[0] < .1:
        return -1
    if state[0] > .8:
        return -1
    '''
    return -1 * (x)**2 # -cos_theta + 1.0 

def expand_state(state):
    return [state[0]/1000 - 0.5 , state[1]/1000.]#, 0, 0, state[3]] # np.cos(state[2])np.sin(state[2])

def plot(frame_idx, rewards, losses, model):
    #clear_output(True)
    plt.clf()
    plt.figure(1, figsize=(20,5))
    plt.subplot(141)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(142)
    plt.title('loss')
    plt.plot(losses)

    plt.subplot(143)
    plt.title('q')
    qls = []
    qrs = []
    diffs = []
    xs = np.arange(-0.3, 0.4, 0.1)
    for x in xs:
        out = model.forward(torch.tensor([x, 0]).unsqueeze(0))
        print("out", out.detach().numpy()[0])
        ql, qr = out.detach().numpy()[0]
        qls.append(ql)
        qrs.append(qr)
        diffs.append(ql-qr)
    plt.plot(xs,qls)
    plt.plot(xs,qrs)    

    plt.subplot(144)
    plt.title('q diff')
    plt.plot(xs,diffs)  

    plt.pause(0.01)
    #plt.show()

epsilon_start = 1.2
epsilon_final = 0.01
epsilon_decay = 3000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

cart_speed = 0

i = 0
all_rewards = []
while True:
    i += 1
    state = cart.analyzer.getState()
    state = expand_state(state)
    print("state", state)
    epsilon = epsilon_by_frame(i)
    action = model.act(state, epsilon)
    
    cart_speed = 1000 * ((2 * action) -1)
    cart.setSpeedMmPerS(cart_speed)

    if state[0] < -.3:
        cart.setSpeedMmPerS(800)
        cart_speed = 800
        continue
    if state[0] > .2:
        cart.setSpeedMmPerS(-800)
        cart_speed = -800
        continue

    next_state = cart.analyzer.getState()    
    next_state = expand_state(next_state)
    reward = get_reward(next_state)
    print(reward, epsilon, action)
    done = 0

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
        
    if i % 20 == 0:
        plot(i, all_rewards, losses, model)
