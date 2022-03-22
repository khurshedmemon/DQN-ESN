"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import random
import numpy as np

# from .utils import save_npy, load_npy


class ReplayMemory:

    def __init__(self, config, model_dir, shape, k):
        self.model_dir = model_dir
        self.memory_size = int(config.memory_size)
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.integer)
        self.state = np.empty((self.memory_size, ) + shape, dtype=np.float16)
        #self.mask = np.empty((self.memory_size, k), dtype=np.bool_)
        self.poststate = np.empty((self.memory_size, ) + shape, dtype=np.float16)
        #self.postmask = np.empty((self.memory_size, k), dtype=np.bool_)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0

        # # pre-allocate prestates and poststates for minibatch
        # self.prestates = np.empty((self.batch_size, ) + shape, dtype=np.float16)
        # self.premasks = np.empty((self.batch_size, k), dtype=np.bool_)
        # self.poststates = np.empty((self.batch_size, ) + shape, dtype=np.float16)
        # self.postmasks = np.empty((self.batch_size, k), dtype=np.bool_)

    def add(self, state, poststate, reward, action, terminal):
    #def add(self, state, mask, poststate, postmask, reward, action, terminal):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.state[self.current, ...] = state
        #self.mask[self.current, ...] = mask
        self.poststate[self.current, ...] = poststate
        #self.postmask[self.current, ...] = postmask
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self):
        # memory must include poststate, prestate and history
        assert self.count > 3
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            while True:
                index = random.randint(0, self.count - 1)
                break

            indexes.append(index)

        prestates = self.state[indexes, ...]
        #premasks = self.mask[indexes, ...]
        poststates = self.poststate[indexes, ...]
        #postmasks = self.postmask[indexes, ...]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        #return prestates, premasks, actions, rewards, poststates, postmasks, terminals
        return prestates, actions, rewards, poststates, terminals


    # def save(self):
    #     for idx, (name, array) in enumerate(
    #         zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates', 'premasks', 'postmasks'],
    #             [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates, self.premasks, self.postmasks])):
    #         save_npy(array, os.path.join(self.model_dir, name))

    # def load(self):
    #     for idx, (name, array) in enumerate(
    #         zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates', 'premasks', 'postmasks'],
    #             [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates, self.premasks, self.postmasks])):
    #         array = load_npy(os.path.join(self.model_dir, name))
