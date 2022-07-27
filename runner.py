import gym
import random
import Maze 
import numpy as np
import scipy.stats


class LinearSoftmaxAgent(object):
    """Act with softmax policy. Features are encoded as
    phi(s, a) is a 1-hot vector of states."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.theta = np.random.random(state_size * action_size)
        self.alpha = .01
        self.gamma = .99

    def store(self, state, action, prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)

    def _phi(self, s, a):
        encoded = np.zeros([self.action_size, self.state_size])
        encoded[a] = s
        return encoded.flatten()

    def _softmax(self, s, a):
        return np.exp(self.theta.dot(self._phi(s, a)) / 100)

    def pi(self, s):
        """pi(a | s)"""
        weights = np.empty(self.action_size)
        for a in range(self.action_size):
            weights[a] = self._softmax(s, a)
        return weights / np.sum(weights)

    def act(self, state):
        probs = self.pi(state)
        a = random.choices(range(0, self.action_size), weights=probs)
        a = a[0]
        pi = probs[a]
        return (a, pi)

    def _gradient(self, s, a):
        expected = 0
        probs = self.pi(s)
        for b in range(0, self.action_size):
            expected += probs[b] * self._phi(s, b)
        return self._phi(s, a) - expected

    def _R(self, t):
        """Reward function."""
        total = 0
        for tau in range(t, len(self.rewards)):
            total += self.gamma**(tau - t) * self.rewards[tau]
        return total

    def train(self):
        self.rewards -= np.mean(self.rewards)
        self.rewards /= np.std(self.rewards)
        for t in range(len(self.states)):
            s = self.states[t]
            a = self.actions[t]
            r = self._R(t)
            grad = self._gradient(s, a)
            self.theta = self.theta + self.alpha * r * grad
        # print(self.theta)
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []

    def getName(self):
        return 'LinearSoftmaxAgent'

    def save(self):
        pass

def encode_state(env):
    goal = env.goal_position
    pos = env.position
    encoding = goal[0]*maze_size+goal[1]
    encoding*=maze_size*maze_size
    encoding+=pos[0]*maze_size+pos[1]
    return encoding

SAVE_FREQUENCY = 100
maze_size = 4
env = Maze.Maze(dimensions = [maze_size, maze_size])
env.reset()
state = encode_state(env)
score = 0
episode = 0
state_size = (maze_size*maze_size)*(maze_size*maze_size)
action_size = 4
g = LinearSoftmaxAgent(state_size, action_size)
actions = ['UP','DOWN','RIGHT','LEFT']

MAX_EPISODES = 10000
while episode < MAX_EPISODES:  # episode loop
    # env.render()
    action, prob = g.act(state)

    reward = env.move(actions[action])
    state, done = encode_state(env), env.over # take a random action
    score += reward
    g.store(state, action, prob, reward)

    if done:
        episode += 1
        g.train()
        print('Episode: {} Score: {}'.format(episode, score))
        score = 0
        state = env.reset()
        if episode % SAVE_FREQUENCY == 0:
            g.save()