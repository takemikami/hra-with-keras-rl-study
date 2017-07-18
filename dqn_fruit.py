import numpy as np
import gym
import curses
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from rl.core import Processor

# choice or sumup reward list
class RewardProcessor(Processor):
    def __init__(self, target):
        self.target = target
        return
    def process_reward(self, reward_list):
        rtn = 0
        if self.target == -1:
            rtn = np.sum(reward_list)
        else:
            rtn = reward_list[self.target]
        return rtn

# define dqn agent
def agent(env, target=-1):
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(env.action_space.n, activation='softmax'))
    print(model.summary())

    # create dqn agent
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy, processor=RewardProcessor(target))
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    return dqn

if __name__ == '__main__':
    cnt_step = 50000
    subcmd = sys.argv[1]

    # create fruit collect env
    import fruit_env
    env = fruit_env.FruitCollectEnv()
    np.random.seed(123)
    env.seed(123)

    if subcmd == 'train':
        # create agent
        dqn = agent(env=env)

        # fitting
        dqn.fit(env, nb_steps=cnt_step, visualize=False, verbose=2)

        # save weights
        dqn.save_weights('dqn_weights_sum.h5f', overwrite=True)

    elif subcmd == 'test':
        # create agent
        dqn = agent(env=env)

        # load weights
        dqn.load_weights('dqn_weights_sum.h5f')

        # test
        dqn.test(env, nb_episodes=5, visualize=False)

    elif subcmd == 'train_hr':
        agent_list = []

        # repeat fruit points
        for idx in range(0, 10):
            # create agent
            dqn = agent(env=env, target=idx)
            agent_list.append(dqn)

            # fitting
            dqn.fit(env, nb_steps=cnt_step, visualize=False, verbose=2)

            # save weights
            dqn.save_weights('dqn_weights_{}.h5f'.format(idx), overwrite=True)

            # test
            dqn.test(env, nb_episodes=5, visualize=False)

    elif subcmd == 'test_hr':
        agent_list = []

        # repeat fruit points
        for idx in range(0, 10):
            # create agent
            dqn = agent(env=env, target=idx)
            agent_list.append(dqn)

            # save weights
            dqn.load_weights('dqn_weights_{}.h5f'.format(idx))

        # test merged model
        for episode in range(0, 5):
            obs = env.reset()
            done = False
            total_step = 0
            total_reward = 0.0
            while done == False:
                merged_q_values = np.zeros(4)
                for idx in range(0, 10):
                    q_values = agent_list[idx].compute_q_values([obs])
                    merged_q_values += 1/10 * q_values
                act = np.argmax(merged_q_values)
                obs, reward, done, info = env.step(act)
                total_step += 1
                total_reward += np.sum(reward)
                env.render()
            print("Episode {0}: reward: {1}, steps: {2}".format(episode, total_reward, total_step))
