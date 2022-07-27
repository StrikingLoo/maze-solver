""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle 
import Maze

# hyperparameters
H = 256 # number of hidden layer neurons
batch_size = 256 # every how many episodes to do a param update?
learning_rate = 1e-6
gamma = 0.75 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False



actions = ['UP','DOWN','RIGHT','LEFT']
maze_size = 8
# model initialization
D = maze_size * maze_size * 3 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(4, H) / np.sqrt(H)
  print('w1 shape:')
  print(model['W1'].shape)
  print('w2 shape:')
  print(model['W2'].shape)
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def softmax(x): 
  max_x = np.max(x)
  return np.exp(x - max_x) / np.sum(np.exp(x - max_x)) # softmax!

def prepro(I):
  """ map 32x32x3 input into a 3072 numbers 1d vector """
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] == 100 or r[t] == -50: 
      running_add = r[t] # reset the sum, since this was a game boundary (maze specific!)
    else:
      running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = softmax(logp)

  return p, h # return probability of taking action 2, and hidden state

## Missing rewrite
def policy_backward(eph, epdlogp):
  print(eph.shape) # (1290, 256)
  print(epdlogp.shape) # (1290, 4)

  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp)
  print(dW2.shape) # (256, 4)

  dh = epdlogp @ model['W2'] # np.outer()
  print(dh.shape) # (1290, 256)

  dh[eph <= 0] = 0 # backprop relu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2.T}

env = Maze.Maze(dimensions = [maze_size, maze_size])
env.reset()

observation = env.underlying_scene()
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

print([env.position, env.goal_position])
printed_aprob = None

while True:
  # preprocess the observation, set input to network to be difference image
  x = prepro(observation)
  #print('maze: ')
  #print([env.position, env.goal_position])
  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)

  # fix 
  if env.position[0] == 0:
    aprob[0] = 0
  if env.position[0] == maze_size -1:
    aprob[1] = 0
  if env.position[1] == maze_size -1:
    aprob[2] = 0
  if env.position[1] == 0:
    aprob[3] = 0



  action = np.random.choice(actions, p = softmax(aprob))



  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  # fallible!
  y = np.zeros_like(aprob)
  y[action.argmax()] = 1

  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  reward = env.move(action)

  observation, reward, done = env.underlying_scene(), reward, env.over
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
  if len(drs) % 3000 == 0:
    print(len(drs))

  if done: # an episode finished
    episode_number += 1
    print(f'done! episodes: {episode_number}')
    print(y)
    print(aprob)

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: 
      print(k)
      grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    if episode_number % 100 == 0: pickle.dump(model, open('save'+str(episode_number)+'.p', 'wb'))
    reward_sum = 0
    env.reset()
    observation = env.underlying_scene()
    print([env.position, env.goal_position])

  if reward == 100: # Pong has either +1 or -1 reward exactly when game ends.
    print(f'ep {episode_number}: game finished, reward: {reward}')

