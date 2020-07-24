import gym
import argparse
from ai4u.utils import environment_definitions
from ai4u.ml.a3c.run_checkpoint import run as run_test
from ai4u.ml.a3c.train import run as run_train
import AI4UGym
from AI4UGym import BasicAgent
import numpy as np
from ai4u.utils import image_decode
from stable_baselines.common.callbacks import CheckpointCallback
from math import log, e
from collections import deque

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--run",
						choices=['train', 'test'],
						default='train')
	parser.add_argument("--id", default='0')
	parser.add_argument('--path', default='.')
	parser.add_argument('--preprocessing', choices=['generic', 'user_defined'])
	return parser.parse_args()

def get_frame_from_fields(fields):
	imgdata = image_decode(fields['frame'], 20, 20)
	return imgdata

IMAGE_SHAPE = (20, 20, 4)
ACTION_SIZE = 5


def make_inference_network(obs_shape, n_actions, debug=False, extra_inputs_shape=None):
	import tensorflow as tf
	from ai4u.ml.a3c.multi_scope_train_op import make_train_op 
	from ai4u.ml.a3c.utils_tensorflow import make_grad_histograms, make_histograms, make_rmsprop_histograms, \
		logit_entropy, make_copy_ops

	observations = tf.placeholder(tf.float32, [None] + list(obs_shape))
	
	normalized_obs = tf.keras.layers.Lambda(lambda x : x/255.0)(observations)

	# Numerical arguments are filters, kernel_size, strides
	conv1 = tf.keras.layers.Conv2D(16, (1,1), (1,1), activation='relu', name='conv1')(normalized_obs)
	if debug:
		# Dump observations as fed into the network to stderr for viewing with show_observations.py.
		conv1 = tf.Print(conv1, [observations], message='\ndebug observations:',
						 summarize=2147483647)  # max no. of values to display; max int32
	conv2 = tf.keras.layers.Conv2D(16, (3,3), (1,1), activation='relu', name='conv2')(conv1)
	
	flattened = tf.keras.layers.Flatten()(conv2)
	hidden = tf.keras.layers.Dense(512, activation='relu', name='hidden')(flattened)
	action_logits = tf.keras.layers.Dense(n_actions, activation=None, name='action_logits')(hidden)
	
	action_probs = tf.nn.softmax(action_logits)
	
	values = tf.layers.Dense(1, activation=None, name='value')(hidden)


	# Shape is currently (?, 1)
	# Convert to just (?)
	values = values[:, 0]

	layers = [conv1, conv2, hidden]

	return observations, action_logits, action_probs, values, layers

def make_env_def():
	environment_definitions['state_shape'] = IMAGE_SHAPE
	environment_definitions['action_shape'] = (ACTION_SIZE,)
	environment_definitions['actions'] = [('walk', 10), ('run', 20), ('walk_in_circle',  1), ('walk_in_circle', -1), ('NoOp', -1) ]
	environment_definitions['input_port'] = 8080
	environment_definitions['make_inference_network'] = make_inference_network
	environment_definitions['output_port'] = 7070
	environment_definitions['agent'] = Agent
	environment_definitions['host'] = '127.0.0.1'
	BasicAgent.environment_definitions = environment_definitions

class Agent(BasicAgent):
	def __init__(self):
		BasicAgent.__init__(self)
		self.seq = deque(maxlen=4)

	def __make_state__(imageseq):
		frameseq = np.array(imageseq, dtype=np.float32)
		frameseq = np.moveaxis(frameseq, 0, -1)
		return frameseq

	def reset(self, env):
		env.remoteenv.step("SetNMoves", 1)
		env_info = env.remoteenv.step("restart")
		for i in range(np.random.choice(15)):
			env_info = env.one_step(np.random.choice([0, 1, 2]))
		img = image_decode(env_info['frame'], 20, 20)
		self.seq.append(img)
		self.seq.append(img)
		self.seq.append(img)
		self.seq.append(img)
		return Agent.__make_state__(self.seq)

	def __step(self, env, action):
		reward_sum = 0
		done = False
		for i in range(4):
			state = env.one_step(action)
			reward_sum += state['reward']
			done = state['done']
			if (done):
				break
		if not done:
			state = env.remoteenv.step("get_result")
			reward_sum += state['reward']
			done = state['done']
		return reward_sum, done, state

	def act(self, env, action, info=None):
		reward_sum, done, env_info = self.__step(env, action)
		img = image_decode(env_info['frame'], 20, 20)
		self.seq.append(img)
		frameseq = Agent.__make_state__(self.seq)
		return frameseq, reward_sum, done, env_info


def train():
		args = ['--n_workers=8', '--steps_per_update=30', 'AI4U-v0']
		make_env_def()
		run_train(environment_definitions, args)

def test(path, id=0):
		args = ['AI4U-v0', path]
		make_env_def(id)
		run_test(environment_definitions, args)


if __name__ == '__main__':
	make_env_def()
	args = parse_args()
	if args.run == "train":
		train()
	elif args.run == "test":
		test(args.path, int(args.id))


