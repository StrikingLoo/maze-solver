import Maze
import pickle 
import random
import zlib
import numpy as np

def q(state, action, value_dict):
	return value_dict.get(state, {}).get(action, 0)

def update(state, action, value, value_dict):
	state_values = value_dict.get(state, None)
	if state_values:
		state_values[action] = value
	else:
		value_dict[state] = {}
		value_dict[state][action] = value

GAMMA = 0.9
alpha = 0.25

MAZE_SIZE = 16
EPISODES = 70000
#assert EPISODES > MAZE_SIZE**4
number_before_metrics = 30
ACTIONS = ['UP','DOWN','RIGHT','LEFT']

def allowed_actions(input_state):
	state = zlib.decompress(input_state).decode()
	#state = input_state
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

def unique_starts(i):
	position = [(i // (MAZE_SIZE*MAZE_SIZE))%MAZE_SIZE, (i // (MAZE_SIZE*MAZE_SIZE*MAZE_SIZE))%MAZE_SIZE]
	goal_position = [i % MAZE_SIZE, (i // MAZE_SIZE)%MAZE_SIZE ] #this way goal shifts in the first maze_size**2 positions.
	if i%2 == 0:
		position[0] = abs(MAZE_SIZE - position[0]) -1
		position[1] = abs(MAZE_SIZE - position[1]) -1
		goal_position[0] = abs(MAZE_SIZE - goal_position[0]) -1
		goal_position[1] = abs(MAZE_SIZE - goal_position[1]) -1
	return position, goal_position

def train(env, save_to = '/models/model2.dump'):
	env.reset()
	values = {}
	rewards_list = []
	for episoden in range(EPISODES):
		env.reset()
		position, goal_position = unique_starts(episoden)

		env.position = np.asarray(position)
		env.goal_position = np.asarray(goal_position)
		current_state = env.compressed_state_rep()
		action, action_v = policy(values, current_state)
		total_reward = 0
		step = 0
		while (not env.over):
			step +=1
			reward = env.move(action)
			total_reward += reward
			next_state = env.compressed_state_rep()
			next_action, next_action_v = policy(values, next_state, epsilon = 0.05 if episoden < EPISODES/2 else 0.001)

			new_value = action_v + (next_action_v*GAMMA + reward - action_v)*alpha
			update(current_state, action, new_value, values)
			current_state = next_state
			action = next_action
			action_v = next_action_v
		rewards_list.append(total_reward)
		if episoden > number_before_metrics:
			print(f'episode: {episoden} total_reward: {total_reward} agg: {sum(rewards_list[-number_before_metrics:])/number_before_metrics} ')
			print(f'won %: {100.0*len([r for r in rewards_list[-number_before_metrics:] if r > 0])/number_before_metrics} ')
			print(f'original reward: {sum(rewards_list[:number_before_metrics])/number_before_metrics} ')
			print(f'explored states: {len(values.keys())} ')
			print(f'{env.position} to {env.goal_position}')

	print(rewards_list)
	with open(save_to, 'wb') as handle:
		pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)


env = Maze.Maze(dimensions = [MAZE_SIZE, MAZE_SIZE])
train(env)

