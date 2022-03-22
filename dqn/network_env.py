import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from six import StringIO
import sys
from graph.graphBoard import graphBoard
import pdb


class NetEnv(gym.Env):
    def __init__(self, g, config, opponent, terminal_round=7):
        """
        Args:
            opponent: An opponent policy
        """
        np.set_printoptions(threshold=sys.maxsize)
        self._seed()
        self.g = g
        self.config = config
        self.gb = graphBoard(self.g, config, terminal_round)

        #self.init_mask = np.ones(self.gb.k * self.gb.action_space_size)
        #revised by Ali
        self.init_mask = np.ones(self.gb.action_space_size)
        self.opponent = opponent
        self.opp_probe = True

        # netstate shape
        self.shape = self.gb.state.shape
        #self.observation_space = spaces.Box(np.zeros(self.shape), np.ones(self.shape))
        self.action_space = spaces.Discrete(self.gb.action_size)
        #added by Ali
        self.num_round = 0

    def _reset(self):
        self.gb.reset()
        self.opp_probe = True
        self.num_round = 0
        return self.gb.state, self.init_mask, 0, 0, 0, False

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.gb) + '\n')
        return outfile

    def _step(self, action):
        #target_com = action / self.gb.action_space_size
        #modified by Ali        
        target_com = int(action / self.gb.action_space_size)
        #assert self.gb.valid_com[target_com]
        #mask = np.zeros((self.gb.k, self.gb.action_space_size))
        #revised by Ali
        mask = np.zeros(self.gb.action_space_size)
        #addeb by Ali
        self.num_round += 1
        self.gb.cur_eps_round = self.num_round
        # Play
        if self.gb._play(action) == -1:
            raise
        self.gb.switch_player()
        # Opponent play
        if not self.gb.is_terminal():            
            if self.gb._play(opponent=self.opponent) == -1:
                raise            
            # After opponent play, we should be back to the original color
            self.gb.switch_player()

        # Diffusing after two parties play
        if not self.gb.is_terminal():
            self.gb.propagate()
            #get the next state
            self.gb.update_state()

        ''' 
        for k in range(self.gb.k):
            if self.gb.valid_com[k]:
                mask[k] = 1.0
            else:
                mask[k] = 0.0
        mask = mask.flatten()        
        '''        
        if action != -1:
            mask[action] = 1.0        
        mask = mask.flatten()        
        #compute reward if terminal round
        if self.num_round == self.gb.terminal_round:
            #print('Terminal round')            
            reward, p1_reward, p2_reward = self.gb.calc_reward()
            print('Terminal round: ', self.gb.terminal_round, 'Reward: ', reward)            
            self.num_round = 0
            return self.gb._terminal_state(), mask, reward, p1_reward, p2_reward, True

        # Reward: if nonterminal, then the reward is 0
        if not self.gb.is_terminal():
            return self.gb.state, mask, 0, 0, 0, False
        # We're in a terminal state.
        reward, p1_reward, p2_reward = self.gb.calc_reward()
        return self.gb._terminal_state(), mask, reward, p1_reward, p2_reward, True

    @property
    def _state(self):
        return self.gb.state
