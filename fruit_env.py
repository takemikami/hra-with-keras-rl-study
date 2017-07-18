import gym
import curses
import numpy as np
from gym import spaces

class FruitCollectEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'array']
    }

    def __init__(self, visualize=False):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2, shape=(10,10))
        self.map = np.zeros((10,10))
        self.fruit_points = [
            (2,2),
            (2,5),
            (2,8),
            (5,2),
            (5,4),
            (5,6),
            (5,8),
            (8,2),
            (8,5),
            (8,8),
        ]
        self.fruit_exists = np.zeros(10, dtype=int)
        self.fruit_done = np.zeros(10, dtype=int)
        self.my_pos = (0,0)
        self.win = None
        if visualize:
            self.win = curses.newwin(10,10, 0,0)

    def _step(self, action):
        self.total_step += 1
        done = False

        old_pos = self.my_pos
        if action == 0:
            self.my_pos = (self.my_pos[0]+1, self.my_pos[1]  )
        elif action == 1:
            self.my_pos = (self.my_pos[0]  , self.my_pos[1]-1)
        elif action == 2:
            self.my_pos = (self.my_pos[0]  , self.my_pos[1]+1)
        elif action == 3:
            self.my_pos = (self.my_pos[0]-1, self.my_pos[1]  )

        # fix position when out of map
        if self.my_pos[0] < 0 or self.my_pos[1] < 0 or self.my_pos[0] > 9 or self.my_pos[1] > 9:
            self.my_pos = old_pos

        # check fruit collect
        reward_list = np.zeros(10)
        fruit_idx = self.check_on_fruit()
        if fruit_idx != -1:
            self.fruit_done[fruit_idx] = 1
            reward_list[fruit_idx] = 1
            self.fruit_cnt += 1

        # check done: over 300step or all fruit collected
        if self.total_step > 300 or self.fruit_cnt >= 5:
            done = True

        return self.render(mode='array'), reward_list, done, {}

    def _reset(self):
        # setup fruit
        pointlist = np.arange(0,9,1)
        np.random.shuffle(pointlist)
        exist_pointlist = pointlist[0:5]

        self.fruit_exists = np.zeros(10, dtype=int)
        self.fruit_done = np.zeros(10, dtype=int)
        for idx in exist_pointlist:
            self.fruit_exists[idx] = 1

        # setup my position
        self.my_pos = (int(np.random.rand()*10), int(np.random.rand()*10))
        while self.check_on_fruit() != -1:
            self.my_pos = (int(np.random.rand()*10), int(np.random.rand()*10))
        self.total_step = 0
        self.fruit_cnt = 0

        return self.render(mode='array')

    def _render(self, mode='human', close=False):
        obs = np.zeros((10,10))
        obs[self.my_pos[0], self.my_pos[1]] = 1
        for idx,pos in enumerate(self.fruit_points):
            if self.fruit_exists[idx] == 1 and self.fruit_done[idx] == 0:
                obs[pos[0],pos[1]] = 2

        if mode=='human' and self.win is not None:
            self.win.clear()
            for yidx in range(0,10):
                for xidx in range(0,10):
                    if obs[yidx, xidx] == 1:
                        self.win.addch(yidx, xidx, "@")
                    elif obs[yidx, xidx] == 2:
                        self.win.addch(yidx, xidx, "*")
            self.win.refresh()

        return obs

    def _seed(self, seed=None):
        return []

    def check_on_fruit(self):
        on_fruit_idx = -1
        for idx,pos in enumerate(self.fruit_points):
            if self.fruit_exists[idx] == 1 and self.fruit_done[idx] == 0:
                if self.my_pos[0] == pos[0] and self.my_pos[1] == pos[1]:
                    on_fruit_idx = idx
        return on_fruit_idx
