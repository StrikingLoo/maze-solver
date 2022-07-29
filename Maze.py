import numpy as np
from PIL import Image
import random 
import zlib

GOAL_COLOR = [255,0,0]
PLAYER_COLOR = [0,0,255]
UNKNOWN_COLOR = [0,0,0]
EMPTY_COLOR = [255,255,255]
direction_to_vector = {
	'RIGHT':[0,1],
	'LEFT':[0,-1],
	'UP':[-1,0],
	'DOWN':[1,0]
}

def color_to_char(color):

	color = list(color)
	if color == GOAL_COLOR:
		return 'G'
	if color == PLAYER_COLOR:
		return 'P'
	if color == UNKNOWN_COLOR:
		return 'U'
	if color == EMPTY_COLOR:
		return 'W'
	else:
		print('dead color found')
		return 'W'


def make_gif(frames, path):

    frame_one = frames[0]
    frame_one.save(path, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

class Maze():
	def __init__(self, goal_position = np.asarray([0,0]), initial_position = np.asarray([3,3]), dimensions = [32,32]):
		self.goal_position = goal_position
		self.initial_position = initial_position
		self.width = dimensions[0]
		self.height = dimensions[1]
		self.position = initial_position
		self.rewards = []
		self.total_reward = 0
		self.over = False
		self.visible_mask = np.zeros(shape = [self.width,self.height, 3], dtype=np.uint8)
		i, j = self.position
		for h in range(i-1, i+2):
			for w in range(j-1, j+2):
				if self.bound_check([h,w]):
					self.visible_mask[h][w] = [1,1,1]

	def reset(self):
		self.goal_position = np.random.randint(0, self.width, [2], dtype = np.uint8)
		self.position = np.random.randint(0, self.width, [2], dtype = np.uint8)
		self.rewards = []
		self.total_reward = 0
		self.over = False
		self.visible_mask = np.zeros(shape = [self.width,self.height, 3], dtype=np.uint8)
		i, j = self.position
		for h in range(i-1, i+2):
			for w in range(j-1, j+2):
				if self.bound_check([h,w]):
					self.visible_mask[h][w] = [1,1,1]

	def underlying_scene(self):
		underlying_scene = np.ones(shape = [self.width,self.height,3], dtype=np.uint8)*255 #full white canvas
		underlying_scene[self.position[0]][self.position[1]] = PLAYER_COLOR
		underlying_scene[self.goal_position[0]][self.goal_position[1]] = GOAL_COLOR
		return underlying_scene

	def visible_scene(self):
		underlying_scene = self.underlying_scene() #full picture
		return underlying_scene * self.visible_mask

	def picture(self):
		IMG_SCALE_FACTOR = 16
		scene = self.visible_scene()
		scene = scene.repeat(IMG_SCALE_FACTOR, axis=0).repeat(IMG_SCALE_FACTOR, axis=1)

		return scene

	def move(self, direction):
		assert direction in ['UP','DOWN','RIGHT','LEFT']
		new_position = self.position + direction_to_vector[direction]
		reward = 0

		if self.bound_check(new_position):
			self.position = new_position
		else:
			reward = -1
			if self.total_reward < -(10000):
				reward = -100
				self.over = True

		# calculate and store reward
		if (self.position[0] == self.goal_position[0]) and (self.position[1] == self.goal_position[1]):
			reward = 1000
			self.over = True
		elif reward == 0:
			reward = -1

		self.rewards.append(reward)
		self.total_reward += reward

		# update visible mask
		i, j = self.position
		for h in range(i-1, i+2):
			for w in range(j-1, j+2):
				if self.bound_check([h,w]):
					self.visible_mask[h][w] = [1,1,1]

		return reward*1.0

	def venture(self, direction):
		assert direction in ['UP','DOWN','RIGHT','LEFT']
		new_position = self.position + direction_to_vector[direction]

		next_state_string = ''
		print(self.position)
		print('moves to')
		print(new_position)
		if self.bound_check(new_position):
			new_scene = self.visible_scene()
			new_scene[self.position[0]][self.position[1]] = EMPTY_COLOR
			print(new_scene[new_position[0]][new_position[1]])
			print(new_scene[self.position[0]][self.position[1]])
			i, j = new_position
			for h in range(i-1, i+2):
				for w in range(j-1, j+2):
					if self.bound_check([h,w]):
						new_scene[h][w] = EMPTY_COLOR
						if self.goal_position[0] == h and self.goal_position[1] == w:
							new_scene[h][w] = GOAL_COLOR
			new_scene[new_position[0]][new_position[1]] = PLAYER_COLOR

			print('scene:')
			#print(new_scene)
			next_state_string = self.state_string(new_scene)
		else:
			next_state_string = self.state_string(self.visible_scene())

		return next_state_string

	def state_string(self, visible_scene):

		string_representation = ''
		for i in range(self.height):
			for j in range(self.width):
				string_representation += color_to_char(visible_scene[i][j])
			string_representation+='\n'

		return string_representation

	def current_state_string(self):
		visible_scene = self.underlying_scene()

		string_representation = ''
		for i in range(self.height):
			for j in range(self.width):
				string_representation += color_to_char(visible_scene[i][j])
			string_representation+='\n'

		return string_representation

	def compressed_state_rep(self):
		string_rep = self.current_state_string()
		return zlib.compress(string_rep.encode('ASCII'))


	def path(self, steps):
		pictures = [self.picture()]

		for step in steps:
			self.move(step)
			pictures.append(self.picture())
		
		return pictures

	def bound_check(self, coordinates):
		x, y = coordinates
		return (x >= 0 and x < self.height) and (y >= 0 and y < self.width)


first_maze = Maze(dimensions = [5,5])

## test 1
#img = Image.fromarray(first_maze.picture(), 'RGB')
#img.save('media/first_test.png')

## test 2

#boards = first_maze.path(['UP']*25 + ['LEFT']*25)
#frames = [Image.fromarray(frame, 'RGB') for frame in boards]
#print(first_maze.rewards)
#path = 'media/first_path_visible.gif'
#make_gif(frames, path)


## test 3
#print(first_maze.over)
#returns = []
#for _ in range(100):
#	first_maze = Maze(np.asarray([0,0]), np.asarray([10,10]))
#	while True:
#		first_maze.move(random.choice(['UP','DOWN','RIGHT', 'LEFT']))
#		if first_maze.over:
#			break
#
#	returns.append(sum(first_maze.rewards))
#
#print(sum(returns)/100)

## test 4 venturing
'''
exploration = [first_maze.venture('UP'),
			   first_maze.venture('DOWN'),
			   first_maze.venture('LEFT'),]

for e in exploration:
	print(e)
'''

## test 5 maze string rep compressed
starting = zlib.decompress(first_maze.compressed_state_rep()).decode()
print(starting)
first_maze.move('RIGHT')
forward = zlib.decompress(first_maze.compressed_state_rep()).decode()
print(forward)
first_maze.move('LEFT')
forward = zlib.decompress(first_maze.compressed_state_rep()).decode()
print(forward)
assert(forward == starting)


