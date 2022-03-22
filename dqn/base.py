import os
import shutil
import pprint
import inspect
import tensorflow as tf
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
pp = pprint.PrettyPrinter().pprint
import numpy as np


def class_vars(obj):
    return {k: v for k, v in inspect.getmembers(obj)
            if not k.startswith('__') and not callable(k)}


class BaseModel(object):
    """Abstract object representing an Reader model."""

    def __init__(self, config):
        self._saver = None
        self.config = config

        try:
            self._attrs = config.__dict__['__flags']
        except:
            self._attrs = class_vars(config)
            pp(self._attrs)

        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))

    def save_model(self, step=None):
        print(" [*] Saving checkpoints...")
        #remove the previous saved models
        if os.path.isdir(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

    def load_model(self, bm_dir=""):
        print(" [*] Loading checkpoints...")

        if(bm_dir !=""):
            print('Pre-trained base model directory: ', bm_dir)                        
            ckpt = tf.train.get_checkpoint_state(bm_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                fname = os.path.join(bm_dir, ckpt_name)
                self.saver.restore(self.sess, fname)
                print(" [*] Pre-Trained Model Load SUCCESS: %s" % fname)
                return True
            else:
                print(" [!] Pre-Trained Model Load FAILED: %s" % bm_dir)
                return False
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                fname = os.path.join(self.checkpoint_dir, ckpt_name)
                self.saver.restore(self.sess, fname)
                print(" [*] Load SUCCESS: %s" % fname)
                return True
            else:
                print(" [!] Load FAILED: %s" % self.checkpoint_dir)
                return False

    ##added by Ali
    def save_tr_fig(self, reward, tr_episode):
        print(" [*] Saving training rewards...")
        graph_dir = self.config.env_name + "/" + self.config.opponent
        plt_dir = os.path.join('checkpoints', graph_dir)
        graph_file = plt_dir + '_' + str(tr_episode)
        plt.plot(reward)
        #plt.plot(np.arange(len(reward)), reward)
        plt.xlabel('Episode #')
        plt.ylabel('Reward')
        plt.title(graph_dir)
        #plt.grid(True)
        plt.savefig(graph_file + '.png')

    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints', self.model_dir)

    @property
    def model_dir(self):
        '''
        model_dir = self.config.env_name + "/" + self.config.opponent
        for k, v in self._attrs.items():
            if v == self.config.env_name or k == self.config.opponent:
                continue
            if not k.startswith('_'):
                model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v])
                                         if type(v) == list else v)
        return model_dir + '/'
        '''
        model_dir = self.config.env_name + "/trained_models/op_stra-" + self.config.opponent + "/mv-" + str(self.config.mv) + "/train_scale-" + str(self.config.scale)
        return model_dir + '/'

    @property
    def saver(self):
        if self._saver is not None:
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver
