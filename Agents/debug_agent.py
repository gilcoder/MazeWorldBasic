from ai4u.core import RemoteEnv
import numpy as np
from ai4u.utils import image_decode
import getch

def transform(frame):
	return image_decode(frame, 20, 20)

def agent():
	env = RemoteEnv(IN_PORT=8080, OUT_PORT=7070, host="127.0.0.1")
	env.open(0)
	actions = {'w': ('walk', 10), 's': ('walk', -10), 'd':('walk_in_circle',10), 'a':('walk_in_circle', -10), 'r': ('run', 20)}
	for i in range(10000000):
		#state = env.step("restart")
		state = env.step("restart")
		prev_energy = state['energy']
		done = state['done']
		print("OK")
		while not done:
			print('Action: ', end='')
			action = getch.getche()
			#action = np.random.choice([0, 1])
			reward_sum = 0
			for i in range(7):
				state = env.step(actions[action][0], actions[action][1])
				reward_sum += state['reward']
				if (done):
					break
			if not done:
				state = env.step("get_result")
				reward_sum += state['reward']
				done = state['done']

			energy = state['energy']
			prev_energy = energy
			frame = state['frame']
			print(transform(frame))
			print('Reward: ', reward_sum)
			print('Done: ', state['done'])
			print('Delta energy: ', prev_energy-energy)
			prev_energy = energy
		if i >= 2:
			break
	env.close()

agent()