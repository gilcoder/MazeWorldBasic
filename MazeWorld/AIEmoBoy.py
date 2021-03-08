from ai4u.ml.a3c.train import run as run_train
from ai4u.ml.a3c.run_checkpoint import run as run_test
from ai4u.utils import environment_definitions
import AI4UGym
from AI4UGym import BasicAgent
import numpy as np
import argparse
from collections import deque
from ai4u.utils import image_decode

def make_inference_network(obs_shape, n_actions, debug=False, extra_inputs_shape=None, network=None):
    import tensorflow as tf
    from ai4u.ml.a3c.multi_scope_train_op import make_train_op 
    from ai4u.ml.a3c.utils_tensorflow import make_grad_histograms, make_histograms, make_rmsprop_histograms, logit_entropy, make_copy_ops
    observations = tf.placeholder(tf.float32, [None] + list(obs_shape) )
    linearInput = tf.placeholder(tf.float32, (None, extra_inputs_shape[0]) )
    normalized_LinearInput = tf.keras.layers.Lambda(lambda x : x/100)(linearInput)
    normalized_obs = tf.keras.layers.Lambda(lambda x : x/4)(observations)
    conv1 = tf.keras.layers.Conv2D(128, (1,1), (1,1), activation='relu', name='conv1')(normalized_obs)
    if debug:
        conv1 = tf.Print(conv1, [observations], message='\ndebug observations:', summarize=2147483647)
    liDense1 = tf.keras.layers.Dense(30, activation='tanh', name='phidden')(normalized_LinearInput[:, 0:extra_inputs_shape[0]])
    liDense2 = tf.keras.layers.Dense(30, activation='tanh', name='phidden')(liDense1)
    flattened = tf.keras.layers.Flatten()(conv1)
    exp_features = tf.keras.layers.Concatenate()([flattened, liDense2])
    hidden1 = tf.keras.layers.Dense(256, activation='tanh', name='hidden1')(exp_features)
    hidden2 = tf.keras.layers.Dense(64, activation='tanh', name='hidden2')(hidden1)
    action_logits = tf.keras.layers.Dense(n_actions, activation=None, name='action_logits')(hidden2)
    action_probs = tf.nn.softmax(action_logits)
    values = tf.keras.layers.Dense(1, activation=None, name='value')(hidden2)
    values = values[:, 0]
    layers = [conv1, liDense1, liDense2, exp_features, hidden1, hidden2]
    return (observations, linearInput), action_logits, action_probs, values, layers

def to_image(img):
    imgdata = image_decode(img, 20, 20)
    return imgdata

'''
This method extract environemnt state from a remote environment response.
'''
def get_state_from_fields(fields):
    return [np.concatenate( (fields['AgentForward'], fields['AgentPosition'])), to_image(fields['raycasting'])]

'''
It's necessary overloading the BasicAgent because server response (remote environment) don't have default field 'frame' as state.
'''
class Agent(BasicAgent):
    def __init__(self):
        BasicAgent.__init__(self)
        self.history = deque(maxlen=4)
        for _ in range(4):
            self.history.append( np.zeros( (20, 20) ) )

    def __get_state__(self, env_info):
        p = get_state_from_fields(env_info)
        state = None
        if p[1] is not None:
            self.history.append(p[1])
            frameseq = np.array(self.history, dtype=np.float32)
            frameseq = np.moveaxis(frameseq, 0, -1)
            if p[0] is not None:
                state = (frameseq, np.array(p[0], dtype=np.float32))
            else:
                state = frameseq
        elif p[0] is not None:
            state = np.array(p[0], dtype=np.float32)
        return state

    def reset(self, env):
        env_info = env.remoteenv.step("restart")
        return self.__get_state__(env_info)

    def act(self, env, action, info=None):
        reward = 0
        envinfo = {}
        for _ in range(8):
            envinfo = env.one_stepfv(action)
            reward += envinfo['reward']
            if envinfo['done']:
                break
        return self.__get_state__(envinfo), reward, envinfo['done'], envinfo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",
                        choices=['train', 'test'],
                        default='train')
    parser.add_argument('--path', default='.')
    parser.add_argument('--preprocessing', choices=['generic', 'user_defined'])
    return parser.parse_args()

def make_env_def():
        environment_definitions['state_shape'] = (20, 20, 4)
        environment_definitions['extra_inputs_shape'] = (6,)
        environment_definitions['make_inference_network'] = make_inference_network
        environment_definitions['action_shape'] = (10, )
        environment_definitions['actions'] = [('Character',[10, 0, 0, 0, 0, 0, 0, 0, 0, 0]),('Character',[-10, 0, 0, 0, 0, 0, 0, 0, 0, 0]),('Character',[0, 10, 0, 0, 0, 0, 0, 0, 0, 0]),('Character',[0, -10, 0, 0, 0, 0, 0, 0, 0, 0]),('Character',[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),('Character',[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),('Character',[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),('Character',[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),('Character',[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),('Character',[0, 0, 0, 0, 1, 0, 0, 0, 0, 0])]
        environment_definitions['agent'] = Agent

def train():
        args = ['--n_workers=8', 'AI4U-v0']
        make_env_def()
        run_train(environment_definitions, args)

def test(path):
        args = ['AI4U-v0', path,]
        make_env_def()
        run_test(environment_definitions, args)


if __name__ == '__main__':
   args = parse_args()
   if args.run == "train":
        train()
   elif args.run == "test":
        test(args.path)

