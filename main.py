from __future__ import print_function
import argparse
import os, sys
import numpy as np
import random
import tensorflow as tf
import graph_tool.all as gt
from dqn.agent import Agent
from dqn.network_env import NetEnv
from config import get_config
flags = tf.app.flags
import pdb

# Model
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', True, 'Whether to use double q-learning')
flags.DEFINE_string('mv', 'DRL_DSN_Eps_50k_st_hcf', 'model-version based on training episodes')
# Environment
flags.DEFINE_string('env_name', 'bitcoinalpha','The name of graph to use')
flags.DEFINE_integer('scale', 50000,'Number of training episodes')
flags.DEFINE_string('st_type', 'hcf','State: Hand-Crafted feature-based or embedded state; [emb, hcf]')
flags.DEFINE_string('diff_model', 'transient','Transient or Persistent diffusion model')
# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '2/3', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_string('gpu_id', '2', 'GPU Id to use')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_string('opponent', 'degree', 'oppoent strategy')
flags.DEFINE_integer('terminal_round', 7, 'Default 7; Number of rounds to invest budget and propagate diffusion- max_t + 1')
flags.DEFINE_integer('AI_budget', 10, 'Budget by first party')
flags.DEFINE_integer('OP_budget', 10, 'Budget by second party')
flags.DEFINE_integer('testing_episode', 2000, 'Number of test episodes')
flags.DEFINE_boolean('use_bm', False, 'Whether to use pre-trained model or not')
flags.DEFINE_string('bm_dir', './checkpoints/bitcoinotc/trained_models/op_stra-degree/mv-DRL_DSN_Eps_50k_st_emb/train_scale-50000/', 'Base model directory')

FLAGS = flags.FLAGS
# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")


def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = idx / num
    print(" [*] GPU : %.4f" % fraction)
    return fraction


def set_graph(g, config):
    print("Total num of vertices: %d" % g.num_vertices())
    print("Total num of edges: %d" % g.num_edges())
    g.set_directed(True)
    g.vp.visited = g.new_vp("int", 0)
    g.ep.weight = g.new_ep("double", 0.0)

    #assign fixed edge-weights [0.4, 0.5, 0.7, 1]
    for e in g.edges():
        g.ep.weight[e] = 0.4
    '''
    #Revised edge-weights    
    for v in g.vertices():        
        for u in v.out_neighbours():
            #in_deg_u = u.in_degree()
            in_deg_u = 0
            for z in u.in_neighbours():
                in_deg_u += 1
            if(in_deg_u != 0):                
                g.ep.weight[g.edge(g.vertex_index[v], g.vertex_index[u])] = round((1/in_deg_u), 2)
            else:                
                g.ep.weight[g.edge(g.vertex_index[v], g.vertex_index[u])] = 1    
    '''
    g.vp.thres = g.new_vp("double", 0.0)
    g.vp.thres_p1 = g.new_vp("double", 0.0)
    g.vp.thres_p2 = g.new_vp("double", 0.0)

    #Random Activation Thresholds
    for v in g.vertices():
        sample = round(np.random.normal(0.5, 0.125), 2)
        if not (sample <= 0 or sample >= 1):
            g.vp.thres[v] = sample    
    
    g.save("../data/" + config.env_name + "_weighted.txt.xml.gz")


def main(_):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
    #which gpu to use
    #which gpu to use
    if FLAGS.use_gpu:
      device_name = "/gpu:"+ str(FLAGS.gpu_id)
    else:
      device_name = "/cpu:0"
    with tf.device(device_name):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True)) as sess:
            config = get_config(FLAGS) or FLAGS
            #set parameters            
            try:
                g = gt.load_graph("../data/" + config.env_name + "_weighted.txt.xml.gz")
            except:
                g = gt.load_graph("../data/" + config.env_name + ".txt.xml.gz")
                set_graph(g, config)
            
            print('Graph Stats: , Total Number of Nodes; ', g.num_vertices(), 'Total Number of edges: ', g.num_edges())            
            #get the maximum time-stamps based on edge-creations times
            #revised edge-creation times based on numerical digits
            g.ep.creat_time = g.edge_properties["edge_cr_time"]
            edge_cr_time_arr = g.ep.creat_time.a
            max_t = np.amax(edge_cr_time_arr)            
            FLAGS.terminal_round = max_t
            opponent = FLAGS.opponent
            terminal_round = FLAGS.terminal_round
            config.terminal_round = terminal_round
            
            '''
            Evaluate number of subgraphs
            for k in range(20):
                sub_grphs = {}
                v_rnd = random.choice(list(g.vertices()))            
                for i in range(1, max_t+1):
                    print('iteration :', i)
                    fl_G = gt.GraphView(g, efilt=lambda e: g.ep.creat_time[e] == i)
                    sub_grphs[i] = gt.Graph(fl_G, prune=True)
                    #check the number of nodes and edges at each time-stamped graph
                    print("number of vertices: ", sub_grphs[i].num_vertices())
                    print("number of edges: ", sub_grphs[i].num_edges())
                    v_src = sub_grphs[i].vertex(v_rnd)
                    print ('source node: ', v_src)            
            pdb.set_trace()
            ''' 
            #Create Network Environment object           
            env = NetEnv(g, config, opponent, terminal_round)

            if not tf.test.is_gpu_available() and FLAGS.use_gpu:
                raise Exception("use_gpu flag is true when no GPUs are available")

            #Create Agent object
            agent = Agent(config, env, sess)
            if FLAGS.is_train:
                agent.train()            
            else:
                agent.play(test_ep=0.000001)

if __name__ == '__main__':
    tf.app.run()