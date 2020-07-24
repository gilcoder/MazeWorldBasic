import gym

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from ai4u.utils import environment_definitions
import AI4UGym
from AI4UGym import BasicAgent
import numpy as np
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.deepq.policies import FeedForwardPolicy
from ai4u.utils import image_decode
from stable_baselines.common.callbacks import CheckpointCallback
from math import log, e
from collections import deque

def modified_cnn(unscaled_images, **kwargs):
	import tensorflow as tf
	scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
	activ = tf.nn.relu
	layer_1 = activ(conv(scaled_images, 'c1', n_filters=16, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
	layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=2, stride=2, init_scale=np.sqrt(2), **kwargs))
	layer_2 = conv_to_fc(layer_2)
	return activ(linear(layer_2, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class CustomPolicy(CnnPolicy):
	def __init__(self, *args, **kwargs):
		super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn)

IMAGE_SHAPE = (20, 20, 4)
ACTION_SIZE = 5

model = None

def make_env_def():
	environment_definitions['state_shape'] = IMAGE_SHAPE
	environment_definitions['action_shape'] = (ACTION_SIZE,)
	environment_definitions['actions'] = [('walk', 10), ('run', 20), ('walk_in_circle',  5), ('walk_in_circle', -5), ('NoOp', -1) ]
	environment_definitions['input_port'] = 8080
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
		for i in range(7):
			state = env.one_step(action)
			reward_sum += state['reward']
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

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func

def train():
	checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./models/',
										 name_prefix='rl_model')
	make_env_def()
	# multiprocess environment
	env = make_vec_env('AI4U-v0', n_envs=8)
	model = PPO2(CustomPolicy, env, verbose=1, n_steps=128, noptepochs=4, nminibatches=4, learning_rate=linear_schedule(2.5e-5), cliprange=linear_schedule(0.1), vf_coef=0.5, ent_coef=0.01, cliprange_vf=-1, tensorboard_log="./logs/")
	model.learn(total_timesteps=int(1e7), callback=checkpoint_callback)
	model.save("ppo2_model")
	del model # remove to demonstrate saving and loading

def test():
	make_env_def()
	# multiprocess environment
	env = make_vec_env('AI4U-v0', n_envs=8)
	
	model = PPO2.load("ppo2_model_baked", policy=CustomPolicy, tensorboard_log="./logs/")
	model.set_env(env)
	
	# Enjoy trained agent
	obs = env.reset()
	while True:
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		#env.render()

if __name__ == '__main__':
	train()

