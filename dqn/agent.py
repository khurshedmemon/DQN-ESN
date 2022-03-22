from __future__ import print_function
import os
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from functools import reduce
from .base import BaseModel
from .replay_memory import ReplayMemory
from .ops import linear, clipped_error
from .utils import save_pkl, load_pkl
import cProfile
import matplotlib.pyplot as plt
import time
import pdb

class Agent(BaseModel):
    
    def __init__(self, config, environment, sess):
        super(Agent, self).__init__(config)
        self.sess = sess
        self.weight_dir = 'weights'
        self.env = environment
        self.shape = self.env.gb.state.shape
        self.num_neuron = 512
        self.memory = ReplayMemory(self.config, self.model_dir, self.shape, self.env.gb.action_size)

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.build_dqn()    

    def train(self):        
        start_step = self.step_op.eval()        
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = float('-inf')
        ep_rewards, actions = [], []
        action = 0
        state, mask, reward, p1_reward, p2_reward, terminal = self.env._reset()
        episode_rewards = []
        eps = self.ep_start
        start = time.time()
        
        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
            print('#steps:', self.step)
            if self.step == self.learn_start:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []
            # 1. predict
            eps = max(self.ep_end, self.ep_decay*eps)
            action = self.predict(state, mask)
            # 2. act
            poststate, postmask, reward, p1_reward, p2_reward, terminal = self.env._step(action)
            #Revised by Ali
            #poststate, postmask, reward, terminal = self.env._step(action)
            # 3. observe
            #self.observe(state, mask, poststate, postmask, reward, action, terminal)
            self.observe(state, poststate, reward, action, terminal)

            state = poststate
            mask = postmask
            actions.append(action)
            total_reward += reward

            if terminal:
                ep_reward += reward
                state, mask, reward, p1_reward, p2_reward, terminal = self.env._reset()
                num_game += 1
                ep_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step                    
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count
                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d'
                          % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game))
                    '''
                    if max_avg_ep_reward * 0.8 <= avg_ep_reward:
                        self.step_assign_op.eval(
                            {self.step_input: self.step + 1})
                        self.save_model(self.step + 1)

                        max_avg_ep_reward = max(
                            max_avg_ep_reward, avg_ep_reward)
                    '''
                    #Revised by Ali for saving the model after completion of training episodes
                    if (self.step + 1) == self.max_step:
                        self.step_assign_op.eval(
                            {self.step_input: self.step + 1})
                        self.save_model(self.step + 1)
                    
                    if self.step > 180:
                        self.inject_summary({
                            'average.reward': avg_reward,
                            'average.loss': avg_loss,
                            'average.q': avg_q,
                            'episode.max reward': max_ep_reward,
                            'episode.min reward': min_ep_reward,
                            'episode.avg reward': avg_ep_reward,
                            'episode.num of game': num_game,
                            'episode.rewards': ep_rewards,
                            'episode.actions': actions,
                            'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
                        }, self.step)

                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []
            #save the rewards after 1000 training iterations
            #if (self.step + 1) % 10000 == 0:
                #self.save_tr_fig(episode_rewards, len(episode_rewards))
        #Time consumption for training
        print('Total training time consumed after all training episodes: ', time.time() - start, ' (seconds)')

    def predict(self, s_t, mask, test_ep=None):
        ep = test_ep or (self.ep_end +
                         max(0., (self.ep_start - self.ep_end)
                             * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
        if random.random() < ep:
            #target_com = random.sample(np.where(self.env.gb.valid_com)[0], 1)[0]
            #added by Ali
            #target_com = random.sample(list(np.where(self.env.gb.valid_com)[0]), 1)[0]
            #action = random.randrange(target_com * self.env.gb.action_space_size, (target_com + 1) * self.env.gb.action_space_size - 1)
            action = random.choice(np.arange(self.env.gb.action_space_size))
            #print('random action selection', action)
        else:            
            #action = self.q_action.eval({self.s_t: [s_t], self.mask: [mask]})[0]
            action = self.q_action.eval({self.s_t: [s_t]})[0]
            #print('Action selection using Q-value', action)
        return action
    ''' Revised
    def observe(self, state, mask, poststate, postmask, reward, action, terminal):
        reward = reward / (self.env.g.num_vertices() * 0.5)
        self.memory.add(state, mask, poststate, postmask, reward, action, terminal)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()
    '''
    #Revised observe function without masking
    def observe(self, state, poststate, reward, action, terminal):
        reward = reward / (self.env.g.num_vertices() * 0.5)
        self.memory.add(state, poststate, reward, action, terminal)
        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()

    def q_learning_mini_batch(self):
        if self.memory.count < 5:
            return
        else:
            #s_t, mask_t, action, reward, s_t_plus_1, mask_t_plus_1, terminal = self.memory.sample()
            s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

        if self.double_q:
            # Double Q-learning
            #pred_action = self.q_action.eval({self.s_t: s_t_plus_1, self.mask: mask_t_plus_1})
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})
            '''
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                self.target_s_t: s_t_plus_1,
                self.target_mask: mask_t_plus_1,
                self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
            })
            '''
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                self.target_s_t: s_t_plus_1,                
                self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
            })

            target_q_t = (1. - terminal) * self.discount * \
                q_t_plus_1_with_pred_action + reward
        else:
            #q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1, self.target_mask: mask_t_plus_1})
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
            terminal = np.array(terminal) + 0.
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = (1. - terminal) * self.discount * \
                max_q_t_plus_1 + reward

        _, q_acted, loss = self.sess.run([self.optim, self.q_acted, self.loss], {
            self.target_q_t: target_q_t,
            #self.mask: mask_t,
            self.action: action,
            self.s_t: s_t,
            self.learning_rate_step: self.step
        })
        self.total_loss += loss
        self.total_q += q_acted.mean()
        self.update_count += 1

        #print('Total loss at q-learning: ', self.total_loss)
        #print('Total Q: ', self.total_q)
        #print('Total Update count: ', self.update_count)

    def build_dqn(self):
        self.w = {}
        self.t_w = {}
        self.l = {}
        self.target_l = {}

        activation_fn = tf.nn.relu

        # training network
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32', [None] + list(self.shape), name='s_t')
            #self.mask = tf.placeholder('float32', [None, self.env.gb.action_size], name='mask')
            shape = self.s_t.get_shape().as_list()
            self.s_t_flat = tf.reshape(
                self.s_t, [-1, reduce(lambda x, y: x * y, shape[1:])])

            if self.dueling:
                self.value_hid, self.w['l_val_w'], self.w['l_val_b'] = \
                    linear(self.s_t_flat, self.num_neuron,
                           activation_fn=activation_fn, name='value_hid')

                self.adv_hid, self.w['l_adv_w'], self.w['l_adv_b'] = \
                    linear(self.s_t_flat, self.num_neuron,
                           activation_fn=activation_fn, name='adv_hid')

                self.value, self.w['val_w_out'], self.w['val_w_b'] = \
                    linear(self.value_hid, 1, name='value_out')

                self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
                    linear(self.adv_hid, self.env.gb.action_size, name='adv_out')
                # Average Dueling
                self.q = self.value + (self.advantage -
                                       tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            else:
                self.l['l1_out'], self.w['l1_w'], self.w['l1_b'] = linear(
                    self.s_t_flat, self.num_neuron, activation_fn=activation_fn, name='l1')
                k = 1
                for k in range(2, self.num_hidden_layer + 1):
                    self.l['l' + str(k) + '_out'], self.w['l' + str(k) + '_w'], self.w['l' + str(k) + '_b'] = linear(
                        self.l['l' + str(k - 1) + '_out'], self.num_neuron, activation_fn=activation_fn, name='l' + str(k))
                '''
                self.q, self.w['q_w'], self.w['q_b'] = linear(
                    self.l['l' + str(k) + '_out'], self.env.gb.action_size, name='q', mask=self.mask)
                '''
                self.q, self.w['q_w'], self.w['q_b'] = linear(
                    self.l['l' + str(k) + '_out'], self.env.gb.action_size, name='q')

            self.q_action = tf.argmax(self.q, axis=1)

        # target network
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', [None] + list(self.shape), name='target_s_t')
            #self.target_mask = tf.placeholder('float32', [None, self.env.gb.action_size], name='target_mask')
            shape = self.target_s_t.get_shape().as_list()
            self.target_s_t_flat = tf.reshape(
                self.target_s_t, [-1, reduce(lambda x, y: x * y, shape[1:])])

            if self.dueling:
                self.t_value_hid, self.t_w['l_val_w'], self.t_w['l_val_b'] = \
                    linear(self.target_l3_flat, self.num_neuron,
                           activation_fn=activation_fn, name='target_value_hid')

                self.t_adv_hid, self.t_w['l_adv_w'], self.t_w['l_adv_b'] = \
                    linear(self.target_l3_flat, self.num_neuron,
                           activation_fn=activation_fn, name='target_adv_hid')

                self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
                    linear(self.t_value_hid, 1, name='target_value_out')

                self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
                    linear(self.t_adv_hid, self.env.gb.action_size,
                           name='target_adv_out')

                # Average Dueling
                self.target_q = self.t_value + (self.t_advantage -
                                                tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
            else:
                self.target_l['l1_out'], self.t_w['l1_w'], self.t_w['l1_b'] = \
                    linear(self.target_s_t_flat, self.num_neuron,
                           activation_fn=activation_fn, name='target_l1')
                k = 1
                for k in range(2, self.num_hidden_layer + 1):
                    self.target_l['l' + str(k) + '_out'], self.t_w['l' + str(k) + '_w'], self.t_w['l' + str(k) + '_b'] = \
                        linear(self.target_l['l' + str(k - 1) + '_out'], self.num_neuron,
                               activation_fn=activation_fn, name='target_l' + str(k))

                #self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                #    linear(self.target_l['l' + str(k) + '_out'], self.env.gb.action_size, name='target_q', mask=self.target_mask)
                self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                    linear(self.target_l['l' + str(k) + '_out'], self.env.gb.action_size, name='target_q')

            self.target_q_idx = tf.placeholder(
                'int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(
                self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder(
                    'float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[
                    name].assign(self.t_w_input[name])        
        # optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder(
                'float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            action_one_hot = tf.one_hot(
                self.action, self.env.gb.action_size, 1.0, 0.0, name='action_one_hot')
            self.q_acted = tf.reduce_sum(
                self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - self.q_acted

            self.global_step = tf.Variable(0, trainable=False)

            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
            self.learning_rate_step = tf.placeholder(
                'int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.learning_rate,
                                                   self.learning_rate_step,
                                                   self.learning_rate_decay_step,
                                                   self.learning_rate_decay,
                                                   staircase=True))
            self.optim = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss, name="optim")
            #self.optim = tf.train.AdamOptimizer(
                #self.learning_rate_op).minimize(self.loss, name="optim")

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q',
                                   'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder(
                    'float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.scalar(
                    "%s/%s" % (self.env_name, tag), self.summary_placeholders[tag])

            histogram_summary_tags = ['episode.rewards', 'episode.actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder(
                    'float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.histogram(
                    tag, self.summary_placeholders[tag])

            self.writer = tf.summary.FileWriter(
                './logs/%s' % self.model_dir, self.sess.graph)

        tf.initialize_all_variables().run()
        '''
        self._saver = tf.train.Saver(
            self.w.values() + [self.step_op], max_to_keep=30)
        '''
        #changed by Ali
        self._saver = tf.train.Saver(
            list(self.w.values()) + [self.step_op], max_to_keep=30)

        #check if pre-trained model is used for testing against other strategies
        if(self.config.use_bm & self.config.is_train):
            self.load_model(bm_dir=self.config.bm_dir)            
        elif(self.config.use_bm and (~self.config.is_train)):
            self.load_model(bm_dir=self.config.bm_dir)
            self.update_target_q_network()
        else:
            self.load_model()
            self.update_target_q_network()

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval(
                {self.t_w_input[name]: self.w[name].eval()})

    def save_weight_to_pkl(self):
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(
                self.weight_dir, "%s.pkl" % name))

    def load_weight_from_pkl(self, cpu_mode=False):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}

            for name in self.w.keys():
                self.w_input[name] = tf.placeholder(
                    'float32', self.w[name].get_shape().as_list(), name=name)
                self.w_assign_op[name] = self.w[
                    name].assign(self.w_input[name])

        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]: load_pkl(
                os.path.join(self.weight_dir, "%s.pkl" % name))})

        self.update_target_q_network()

    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, self.step)

    def play(self, n_step=10000, n_episode=1000, test_ep=None, render=False):
        if test_ep is None:
            test_ep = self.ep_end        
        sum_reward = 0
        p1_noes_act = 0
        p2_nodes_act = 0
        best_reward, best_idx = 0, 0
        worst_reward, worst_idx = 200, 0
        num_wins = 0
        n_episode = self.config.testing_episode
        episode_rewards = []
        #time consumption
        start = time.time()
        #for idx in range(self.config.testing_episode):
        for idx in range(n_episode):
            state, mask, reward, p1_reward, p2_reward, terminal = self.env._reset()
            #state, mask, reward, terminal = self.env.reset()
            #for t in tqdm(range(n_step), ncols=70):
            for t in tqdm(range(self.config.terminal_round), ncols=70):
                # 1. predict
                action = self.predict(state, mask, test_ep)
                # print(action)
                # 2. act
                state, mask, reward, p1_reward, p2_reward, terminal = self.env._step(action)
                if terminal:
                    break

            sum_reward += reward
            p1_noes_act += p1_reward
            p2_nodes_act += p2_reward
            episode_rewards.append(reward)

            if reward > best_reward:
                best_reward = reward
                best_idx = idx
            if reward < worst_reward:
                worst_reward = reward
                worst_idx = idx
            if reward > 0:
                num_wins += 1

            print("=" * 30)
            print(" [%d] Current reward : %d" % (idx, reward))
            print(" [%d] Best reward : %d" % (best_idx, best_reward))
            print(" [%d] Worst reward : %d" % (worst_idx, worst_reward))
            print("=" * 30)
        #Time consumption for training
        print('Total time consumed after all testing episodes: ', time.time() - start, ' (seconds)')
        print('Average.reward : %d' % (sum_reward / n_episode))
        print('Best reward : %d' % (best_reward))
        print('Worst reward : %d' % (worst_reward))
        print('Average nodes activated by party 1 : %d' % (p1_noes_act / n_episode))
        print('Average nodes activated by party 2 : %d' % (p2_nodes_act / n_episode))
        print('Variance : %d' % (np.var(episode_rewards)))
        print('num of game : %d' % (n_episode))
        print('num of wins : %d' % (num_wins))

    #Fix Testing - Baseline
    def fix_play(self, n_step=10000, n_episode=1000, test_ep=None, render=False, fix_test=True, fix_alt_test=False, fix_random=False, fix_deg=False):
        if test_ep is None:
            test_ep = self.ep_end        
        sum_reward = 0
        p1_noes_act = 0
        p2_nodes_act = 0
        best_reward, best_idx = 0, 0
        worst_reward, worst_idx = 200, 0
        num_wins = 0
        n_episode = self.config.testing_episode
        episode_rewards = []
        #for idx in range(self.config.testing_episode):
        for idx in range(n_episode):
            state, mask, reward, p1_reward, p2_reward, terminal = self.env._reset()
            #state, mask, reward, terminal = self.env.reset()
            #for t in tqdm(range(n_step), ncols=70):
            p1_prob = True
            #default action, random node selection
            p1_action_random = 5
            p1_action_deg = 1
            for t in tqdm(range(self.config.terminal_round), ncols=70):
                #select random nodes without probing
                if fix_alt_test and fix_random:
                    action = 0 if p1_prob else p1_action_random
                    p1_prob = False if p1_prob == True else True
                elif fix_alt_test and fix_deg:
                    action = 0 if p1_prob else p1_action_deg
                    p1_prob = False if p1_prob == True else True
                elif fix_test and fix_random:
                    action = 5
                elif fix_test and fix_deg:
                    action = 1
                else:
                    #probing for initial 5 rounds and then use random strategy
                    if t <= 4:
                        action = 0
                    else:
                        action = 5                
                # 1. predict
                #action = self.predict(state, mask, test_ep)
                # print(action)
                # 2. act
                state, mask, reward, p1_reward, p2_reward, terminal = self.env._step(action)
                if terminal:
                    break

            sum_reward += reward
            p1_noes_act += p1_reward
            p2_nodes_act += p2_reward
            episode_rewards.append(reward)

            if reward > best_reward:
                best_reward = reward
                best_idx = idx
            if reward < worst_reward:
                worst_reward = reward
                worst_idx = idx
            if reward > 0:
                num_wins += 1

            print("=" * 30)
            print(" [%d] Current reward : %d" % (idx, reward))
            print(" [%d] Best reward : %d" % (best_idx, best_reward))
            print(" [%d] Worst reward : %d" % (worst_idx, worst_reward))
            print("=" * 30)

        print('Average.reward : %d' % (sum_reward / n_episode))
        print('Besr reward : %d' % (best_reward))
        print('Worst reward : %d' % (worst_reward))
        print('Average nodes activated by party 1 : %d' % (p1_noes_act / n_episode))
        print('Average nodes activated by party 2 : %d' % (p2_nodes_act / n_episode))
        print('Variance : %d' % (np.var(episode_rewards)))
        print('num of game : %d' % (n_episode))
        print('num of wins : %d' % (num_wins))