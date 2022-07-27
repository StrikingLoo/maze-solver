import Maze
import pickle 
import random

def q(state, action, value_dict):
	return value_dict.get(state, {}).get(action, 100)

def update(state, action, value, value_dict):
	state_values = value_dict.get(state, None)
	if state_values:
		state_values[action] = value
	else:
		value_dict[state] = {}
		value_dict[state][action] = value

GAMMA = 0.9
alpha = 0.1
EPISODES = 1000
MAZE_SIZE = 8
ACTIONS = ['UP','DOWN','RIGHT','LEFT']

def allowed_actions(state):
	allowed = ['UP','DOWN','RIGHT','LEFT']
	last_line = state.split('\n')[-1]
	for c in last_line:
		if c == 'P':
			allowed.remove('DOWN')
	first_line = state.split('\n')[0]
	for c in first_line:
		if c == 'P':
			allowed.remove('UP')
	line_length = MAZE_SIZE+1
	for i in range(MAZE_SIZE):
		if state[line_length*i] == 'P':
			allowed.remove('LEFT')
		if state[line_length*i + line_length - 1] == 'P':
			allowed.remove('RIGHT')

	return allowed

def policy(values, state, epsilon = 0.1):
	best_action = None
	best_value = float('-inf')
	allowed = allowed_actions(state)
	random.shuffle(allowed)
	for action in allowed:
		if q(state, action, values) > best_value:
			best_value = q(state, action, values)
			best_action = action

	return action, best_value

def train(env, save_to = '/models/model1.dump'):
	env.reset()
	values = {}
	rewards_list = []
	for episoden in range(EPISODES):
		env.reset()
		current_state = env.current_state_string()
		action, action_v = policy(values, current_state)
		total_reward = 0
		step = 0
		while (not env.over):
			step +=1
			reward = env.move(action)
			total_reward += reward
			next_state = env.current_state_string()
			next_action, next_action_v = policy(values, next_state)

			new_value = action_v*(1-alpha) + (next_action_v*GAMMA + reward)*alpha
			update(current_state, action, new_value, values)
			current_state = next_state
			action = next_action
			action_v = next_action_v
		rewards_list.append(total_reward)
		if episoden > 10:
			print(f'episode: {episoden} total_reward: {total_reward} agg: {sum(rewards_list[-10:])} ')
		

	print(rewards_list)
	with open(save_to, 'wb') as handle:
		pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)


env = Maze.Maze(dimensions = [MAZE_SIZE, MAZE_SIZE])
train(env)

