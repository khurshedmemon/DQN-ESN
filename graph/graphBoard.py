import numpy as np
from collections import defaultdict
import random
import graph_tool.all as gt
import pandas as pd
import sys
if (sys.version_info[0] == 2):
    import cPickle
elif (sys.version_info[0] == 3):
    import _pickle as cPickle
import pdb

class graphBoard:

    def __init__(self, g, config, terminal_round=10):
        self.g = g        
        self.k = 1
        self.step = config.step
        self.env_name = config.env_name
        self.diff_model = config.diff_model
        self.terminal_round = terminal_round
        self.cur_eps_round = 1
        self.num_players = 2
        self.terminal_st = False
        self.init_subgraphs = {}
        self.emb_dims = config.dimensions
        self.st_type = config.st_type   #state type: embedded or hand-crafted feature based
        #self.expectation = np.zeros((self.g.num_vertices(), 4 +  (self.step + 1) * 2))  #Status, Degree, Weight, Blocking, SubGreedy
        #revise the state based on number of users and dimensions for network embedding
        if(self.st_type == 'emb'):
            self.expectation = np.zeros((self.g.num_vertices(), (self.emb_dims+1))) # node activation status + embedding dimensions
        else:
            self.expectation = np.zeros((self.g.num_vertices(), 4 ))  #Status, Degree, Weight, Blocking, SubGreedy
        self.initialize()
        self.init_state = self.expectation
        self.action_space = {"degree": 0, "weight": 1, "blocking": 2, "subGreedy": 3, "voting": 4}
        self.action_space_opponent = {"degree": 0, "weight": 1, "blocking": 2, "subGreedy": 3, "voting": 4, "random": 5}
        self.action_space_size = len(self.action_space)
        self.action_size = self.action_space_size * self.k

        self.reset()

    def initialize(self):
        edgeMap = {}
        free_nodes = []
        #for e in self.g.edges():
            #self.g.ep.weight[e] = 1.0
        
        for v in self.g.vertices():
            for w in v.out_neighbours():
                edgeMap.setdefault(int(v), []).append(self.g.edge(v, w))

        for k in range(self.k):            
            free_nodes.append(list(self.g.vertices()))
            self.g.clear_filters()

        self.edgeMap = edgeMap
        self.init_free_nodes = free_nodes
        self.init_visited = self.g.vp.visited.copy()
        self.init_thres = self.g.vp.thres.copy()        
        # initilize time-stamped sub-graphs        
        for i in range(1, self.terminal_round + 1):
            fl_G = gt.GraphView(self.g, efilt=lambda e: self.g.ep.creat_time[e] == i)
            self.init_subgraphs[i] = gt.Graph(fl_G, prune=True)

        #load the first time-stamped Network embeded file
        if(self.st_type == 'emb'):            
            init_emb_file = "../data/" + self.env_name + "_emb_lists/" + self.env_name + ".emb_1"
            self.init_emb_st = self.read_embeddings(init_emb_file)
            #print('initial embedded state: ', self.init_emb_st)
            #pdb.set_trace()
            #initialize state
            self.expectation[:, 0] = np.around(self.expectation[:, 0], 0)
            self.expectation[:, 1:self.emb_dims+1] = self.init_emb_st
        else:
            #load initial sub-graph
            sub_graph = self.init_subgraphs[1]
            max_deg = -1
            max_weight = 0
            for v in sub_graph.vertices():
                out_deg = 0
                out_weight = 0
                in_ind = int(v)
                out_deg = v.out_degree()
                if out_deg > max_deg:
                    max_deg = out_deg

                self.expectation[in_ind][1] = out_deg
                for e in v.out_edges():
                    out_weight += self.g.ep.weight[e]            
                self.expectation[in_ind][2] = out_weight
                if out_weight > max_weight:
                    max_weight = out_weight
            self.ini_max_deg = max_deg
            self.ini_max_weight = max_weight
            self.ini_max_blc_weight = max_weight
            self.expectation[:, 1] = np.around(self.expectation[:, 1] / max_deg, 2)
            self.expectation[:, 2] = np.around(self.expectation[:, 2] / max_weight, 2)

    def reset(self):
        self.g.vp.visited = self.init_visited.copy()
        self.g.vp.thres = self.init_thres.copy()
        self.g.vp.thres_p1 = self.g.new_vp("double", 0.0)
        self.g.vp.thres_p2 = self.g.new_vp("double", 0.0)

        self.free_nodes = [list(nodes_in_com) for nodes_in_com in self.init_free_nodes]
        self.terminal_st = False
        self.spreaders = []
        self.player = 1
        self.valid_com = [True for _ in range(self.k)]
        self.state = self.init_state
        self.sub_graphs = self.init_subgraphs        
    '''
    #read embeddings
    def read_embeddings(self, emb_file):
        words = []
        vectors = []        
        with open(emb_file, 'rb') as f:
            data = file.readlines()[1:]            
            next(emb_file)
            for line in emb_file:
                fields = line.split()
                word = fields[0].decode('utf-8')
                #vector = np.fromiter((float(x) for x in fields[1:]), dtype=np.float)
                vector = [float(x) for x in fields[0:]]
                #words.append(word)
                vectors.append(vector)
        #st = vectors
        st = np.array(vectors)
        return st
    '''

    #read embeddings saved in a pickle format sorted by vetex ids
    def read_embeddings(self, emb_file):
        words = []
        vectors = []        
        with open(emb_file, 'rb') as f:
            #data = file.readlines()[1:]
            emb_file = cPickle.load(f)
        return emb_file

    def _play(self, action=0, opponent=None, round=0):
        self.target_com = int(action / self.action_space_size if not opponent else random.sample(list(np.where(self.valid_com)[0]), 1)[0])        
        #self.target_com = action / self.action_space_size if not opponent else random.sample(np.where(self.valid_com)[0], 1)[0]
        strategy = action % self.action_space_size if not opponent else self.action_space_opponent[opponent]
        assert self.valid_com[self.target_com]
        #print('Player: ', self.player, 'Strategy', strategy)
        if strategy == 0:
            ind = self.degree()
        elif strategy == 1:
            ind = self.weight()
        elif strategy == 2:
            ind = self.blocking()
        elif strategy == 3:
            ind = self.subGreedy()
        #elif strategy == 4:
            #ind = self.centrality()
        #elif strategy == 4:
            #ind = self.randomy()
        elif strategy == 4:
            ind = self.voting()
        elif strategy == 5:
            ind = self.ranodm_node()
        else:
            print("Invalid strategy number")
            return -1

        if ind == -1:
            return -1
        #print ("player: ", self.player, ", ind: ", ind)        
        p1_reward = 0
        p2_reward = 0
        for v in self.g.vertices():
            if self.g.vp.visited[v] == 1:
                p1_reward += 1
            elif self.g.vp.visited[v] == 2:
                p2_reward += 1
        # print "p1: ", p1_reward, ", p2: ", p2_reward
        self.g.vp.visited[ind] = self.player
        #update node's status in state df
        #self.state[:, 0]
        #ver_index_int = int(ind)
        #self.state[ver_index_int, 0] = self.player
        self.free_nodes[self.target_com].remove(ind)
        self.valid_com[self.target_com] = True if self.free_nodes[self.target_com] else False        
        self.spreaders.append(ind)

    def propagate(self, spreaders=None, visited=None, thres_list=None, virtual=False):
        if not virtual:
            spreaders = self.spreaders
            visited = self.g.vp.visited.a
            thres_list = [self.g.vp.thres.a, self.g.vp.thres_p1.a, self.g.vp.thres_p2.a]

        weight = self.g.ep.weight
        edge_crt_time = self.g.ep.creat_time
        candidates = set()
        #check the diffusion model
        if (self.diff_model == 'transient'):
            #u_g = gt.GraphView(self.g, efilt=lambda e: edge_crt_time[e] == self.cur_eps_round)
            sub_graph = self.sub_graphs[self.cur_eps_round]
            for v in spreaders:
                player = visited[int(v)]
                src_node = sub_graph.vertex(v)
                for e in src_node.out_edges():
                    if not visited[int(e.target())]:
                        thres_list[player][int(e.target())] += weight[e]
                        candidates.add(e.target())
        else:
            for v in spreaders:
                player = visited[int(v)]
                for e in v.out_edges():
                    if not visited[int(e.target())]:
                        thres_list[player][int(e.target())] += weight[e]
                        candidates.add(e.target())

        spreaders = []
        for v in candidates:
            max_thres = -1
            winner_list = []
            winner = -1
            for ind, thres_ in enumerate(thres_list):
                if thres_[int(v)] > max_thres:
                    max_thres = thres_[int(v)]
                    winner_list = [ind]
                    winner = ind
                elif thres_[int(v)] == max_thres:
                    if winner != 0:
                        winner_list.append(ind)
            visited[int(v)] = random.sample(winner_list, 1)[0]
            if visited[int(v)]:
                spreaders.append(v)
                if not virtual:
                    self.free_nodes[0].remove(v)        
        if virtual:
            return spreaders
        else:
            self.spreaders = spreaders


    #greedy propagate
    def greedy_propagate(self, spreaders=None, visited=None, thres_list=None, virtual=True):
        if virtual:
            temp_spreaders = self.spreaders
            temp_visited = self.g.vp.visited.a
            temp_thres_list = [self.g.vp.thres.a, self.g.vp.thres_p1.a, self.g.vp.thres_p2.a]        
        temp_weight = self.g.ep.weight
        temp_edge_crt_time = self.g.ep.creat_time
        
        candidates = set()
        #check the diffusion model
        if (self.diff_model == 'transient'):
            #u_g = gt.GraphView(self.g, efilt=lambda e: temp_edge_crt_time[e] == self.cur_eps_round)
            sub_graph = self.sub_graphs[self.cur_eps_roun]
            for v in temp_spreaders:
                player = temp_visited[int(v)]
                src_node = sub_graph.vertex(v)
                for e in src_node.out_edges():
                    if not temp_visited[int(e.target())]:
                        temp_thres_list[player][int(e.target())] += temp_weight[e]
                        candidates.add(e.target())

        else:            
            for v in temp_spreaders:
                player = temp_visited[int(v)]
                #for e in self.edgeMap[int(v)]:
                for e in v.out_edges():
                    if not temp_visited[int(e.target())]:
                        temp_thres_list[player][int(e.target())] += temp_weight[e]
                        candidates.add(e.target())
        spreaders = []
        for v in candidates:
            max_thres = -1
            winner_list = []
            winner = -1
            for ind, thres_ in enumerate(temp_thres_list):                
                if thres_[int(v)] > max_thres:
                    max_thres = thres_[int(v)]                    
                    winner_list = [ind]
                    winner = ind
                elif thres_[int(v)] == max_thres:
                    if winner != 0:
                        winner_list.append(ind)
            temp_visited[int(v)] = random.sample(winner_list, 1)[0]
            if temp_visited[int(v)]:
                spreaders.append(v)

        if virtual:
            return spreaders

    def update_state(self, num_round=0):
        for com in range(self.k):
            self.valid_com[com] = True if self.free_nodes[com] else False

        if sum(self.free_nodes, []) == 0:
            self.is_terminal = True
            self.terminal_st = True
        else:
            if (self.cur_eps_round + 1) <= self.terminal_round:
                self.state[:, 0] = self.g.vp.visited.a                
                #check the state type
                if(self.st_type == 'emb'):
                    init_emb_file = "../data/" + self.env_name + "_emb_lists/" + self.env_name + ".emb_" + str((self.cur_eps_round + 1))
                    t_emb_st = self.read_embeddings(init_emb_file)
                    self.state[:, 1:self.emb_dims+1] = t_emb_st
                else:
                    deg_stat, max_deg, max_weight = self.upd_deg_weight_st()                    
                    for ind in deg_stat:
                        self.state[ind][1] = deg_stat[ind][0]
                        self.state[ind][2] = deg_stat[ind][1]
                    if max_deg != 0:
                        self.state[:, 1] = np.around(self.state[:, 1] / max_deg, 2)
                    if max_weight != 0:
                        self.state[:, 2] = np.around(self.state[:, 2] / max_weight, 2)
                    #Block-status column
                    block_stat, block_dat, max_blk_weight = self.upd_block_st()
                    
                    if (block_stat):
                        for ind, val in block_dat.items():
                            self.state[ind][3] = val
                    if max_blk_weight != 0:
                        self.state[:, 3] = np.around(self.state[:, 3] / max_blk_weight, 2)

                #self.state = t_emb_st

    ##Update vertices degree and weight
    def upd_deg_weight_st(self):
        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)
        deg_st = defaultdict(list)
        #self.target_com = 1
        max_deg = 0
        max_weight = 0
        u_g = self.sub_graphs[self.cur_eps_round]
        for v in self.free_nodes[self.target_com]:
            key = int(v)
            out_deg = 0
            node = u_g.vertex(v)
            out_deg = node.out_degree()
            out_weight = node.out_degree(weight=self.g.ep.weight)
            deg_st[key].append(out_deg)
            deg_st[key].append(out_weight)
            if out_deg > max_deg:
                max_deg = out_deg
            if out_weight > max_weight:
                max_weight = out_weight

        #return updated degree status        
        return deg_st, max_deg, max_weight

    #update vertices blocking weight status
    def upd_block_st(self):
        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)
        block_st = {}
        fringeNodes = []
        blk_st = False
        max_weight = 0
        u_g = self.sub_graphs[self.cur_eps_round]
        for v in self.spreaders:
            if self.g.vp.visited[v] == self._other_player:
                for w in v.out_neighbours():
                    if not self.g.vp.visited[w]:
                        fringeNodes.append(v)

        if not fringeNodes:
            return blk_st, block_st, max_weight

        max_weight = 0
        ind = None
        blk_st = True
        for v in fringeNodes:
            node = u_g.vertex(v)
            out_weight = node.out_degree(weight=self.g.ep.weight)
            ind = int(v)
            block_st[key].append(out_weight)
            if out_weight > max_weight:
                max_weight = out_weight

        return blk_st, block_st, max_weight
        

    def _other_player(self):
        return 1 if self.player == 2 else 2

    def switch_player(self):
        self.player = 1 if self.player == 2 else 2

    def degree(self):
        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)
        max_degree = -1        
        ind = None
        edge_crt_time = self.g.ep.creat_time
        #u_g = gt.GraphView(self.g, efilt=lambda e: edge_crt_time[e] == self.cur_eps_round)
        u_g = self.sub_graphs[self.cur_eps_round]
        #u_g = gt.Graph(u_g, prune = True)
        #check the vertices and edges in sub-graph
        #print("number of vertices: ", u_g.num_vertices())
        #print("number of edges: ", u_g.num_edges())
        
        for v in self.free_nodes[self.target_com]:
            node = u_g.vertex(v)
            degree = node.out_degree()
            if degree > max_degree:
                max_degree = degree
                ind = v        
        
        self.g.clear_filters()
        return ind

    def weight(self):
        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)
        max_weight = -1
        ind = None
        edge_crt_time = self.g.ep.creat_time
        #u_g = gt.GraphView(self.g, efilt=lambda e: edge_crt_time[e] == self.cur_eps_round)
        u_g = self.sub_graphs[self.cur_eps_round]
        #print("number of vertices in edge fundtion: ", u_g.num_vertices())
        #print("number of edges in weight function: ", u_g.num_edges())
        
        for v in self.free_nodes[self.target_com]:            
            node = u_g.vertex(v)
            weight = node.out_degree(weight=self.g.ep.weight)
            if weight > max_weight:
                max_weight = weight
                ind = v
        '''
        stat_df = pd.DataFrame(self.state, columns=self.df_column_names)        
        inact_nod = stat_df[stat_df['Status']==0.0]            
        max_wei_ind = inact_nod[inact_nod['Weig']==inact_nod['Weig'].max()].index.values
        if len(max_wei_ind > 0):
            vertex_ind = max_wei_ind[0]
            ind = self.g.vertex(vertex_ind)
        #print('returned vertex index type : ', type(ind))
        '''
        self.g.clear_filters()
        return ind

    def blocking(self):
        fringeNodes = []
        edge_crt_time = self.g.ep.creat_time
        #u_g = gt.GraphView(self.g, efilt=lambda e: edge_crt_time[e] == self.cur_eps_round)
        u_g = self.sub_graphs[self.cur_eps_round]
        #print("number of vertices in blocking function: ", u_g.num_vertices())
        #print("number of edges in blocking function: ", u_g.num_edges())
        for v in self.spreaders:
            if self.g.vp.visited[v] == self._other_player:
                node = u_g.vertex(v)
                for w in node.out_neighbours():
                    if not self.g.vp.visited[w]:
                        fringeNodes.append(v)

        if not fringeNodes:
            return self.weight()

        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)
        max_weight = -1
        ind = None
        for v in fringeNodes:
            node = sub_graph.vertex(v)
            weight = node.out_degree(weight=self.g.ep.weight)
            if weight > max_weight:
                max_weight = weight
                ind = v
        '''
        ind = None
        stat_df = pd.DataFrame(self.state, columns=self.df_column_names)        
        inact_nod = stat_df[stat_df['Status']==0.0]            
        max_wei_ind = inact_nod[inact_nod['Blk_Wei']==inact_nod['Blk_Wei'].max()].index.values
        if len(max_wei_ind > 0):
            vertex_ind = max_wei_ind[0]
            ind = self.g.vertex(vertex_ind)
        '''
        self.g.clear_filters()
        return ind

    def subGreedy(self):
        free_nodes = self.free_nodes[self.target_com]
        weight = self.g.ep.weight

        max_exp = -1
        ind = None
        edge_crt_time = self.g.ep.creat_time
        #u_g_f = gt.GraphView(self.g, efilt=lambda e: edge_crt_time[e] == self.cur_eps_round)
        u_g_f = self.sub_graphs[self.cur_eps_round]
        #u_g_f = gt.Graph(u_g_f, prune=True)
        second_round = False
        #load the next time-stamped graph if not reached the terminal round already
        if (self.cur_eps_round + 1 ) <= self.terminal_round:
            #u_g_s = gt.GraphView(self.g, efilt=lambda e: edge_crt_time[e] == (self.cur_eps_round + 1 ))
            u_g_s = self.sub_graphs[(self.cur_eps_round + 1)]
            #u_g_s = gt.Graph(u_g_s, prune=True)
            second_round = True

        for spreader in free_nodes:
            visited = np.copy(self.g.vp.visited.a)
            thres_list = [np.copy(self.g.vp.thres.a), np.copy(self.g.vp.thres_p1.a), np.copy(self.g.vp.thres_p2.a)]
            candidates = [spreader]
            exp = 0
            count = 0
            for step in range(2):
                temp_candidates = []
                for v in candidates:
                    if count == 0:
                        node = u_g_f.vertex(v)
                    elif count == 1 and second_round:
                        node = u_g_s.vertex(v)
                    else:
                        node = u_g_f.vertex(v)
                    for e in node.out_edges():
                    #for e in self.edgeMap[int(v)]:
                        w = int(e.target())
                        if not visited[w]:
                            thres_list[self.player][w] += weight[e]
                            if thres_list[0][w] < thres_list[self.player][w]:
                                visited[w] = self.player
                                temp_candidates.append(e.target())
                exp += len(temp_candidates)
                candidates = temp_candidates
                count +=1

            if exp > max_exp:
                max_exp = exp
                ind = spreader

        return ind

    def centrality(self):
        local_cen = self.NB_model.local_cen_com[self.target_com]
        max_cen = -1
        ind = None
        for v in self.free_nodes[self.target_com]:
            cen = local_cen[v]
            if cen > max_cen:
                max_cen = cen
                ind = v

        return ind

    def randomy(self):
        rn = random.randint(0, 3)
        if rn == 0:
            return self.degree()
        if rn == 1:
            return self.weight()
        if rn == 2:
            return self.blocking()
        if rn == 3:
            return self.subGreedy()
        if rn == 4:
            return self.centrality()

    def voting(self):
        candidates = [self.degree(),
                      self.weight(),
                      self.blocking(),
                      self.subGreedy()]
        unique, counts = np.unique(candidates, return_counts=True)

        ind = np.argmax(counts)
        return unique[ind]

    def ranodm_node(self):
        bool_visited = self.g.new_vp("bool", self.g.vp.visited.a.astype(bool))
        self.g.set_vertex_filter(bool_visited, inverted=True)
        ind = None
        edge_crt_time = self.g.ep.creat_time
        #u_g = gt.GraphView(self.g, efilt=lambda e: edge_crt_time[e] == self.cur_eps_round)
        u_g = self.sub_graphs[self.cur_eps_round]
        ind = random.choice(self.free_nodes[self.target_com])
        
        self.g.clear_filters()
        return ind

    def calc_expectation(self, step):  # player = 1 or 2
        spreaders = list(self.spreaders)
        visited_a = np.copy(self.g.vp.visited.a)
        thres_list = [np.copy(self.g.vp.thres.a), np.copy(self.g.vp.thres_p1.a), np.copy(self.g.vp.thres_p2.a)]
        exp_k_steps = [([np.equal(1, visited_a) * 1, np.equal(2, visited_a) * 1])]

        for k in range(0, step):
            spreaders = self.greedy_propagate(spreaders, visited_a, thres_list, True)
            exp_k_steps.append([np.equal(1, visited_a) * 1, np.equal(2, visited_a) * 1])

        exps = np.vstack((exp_k_steps[k] for k in range(step + 1)))
        return exps.T

    def calc_reward(self):
        p1_reward = 0
        p2_reward = 0

        for v in self.g.vertices():
            if self.g.vp.visited[v] == 1:
                p1_reward += 1
            elif self.g.vp.visited[v] == 2:
                p2_reward += 1        

        #reward = p1_reward - p2_reward
        #print ('P1 reward: ', p1_reward, '; P2 Reward: ', p2_reward)
        reward = p1_reward
        return reward, p1_reward, p2_reward

    def is_terminal(self):
        if sum(self.free_nodes, []):
            return False
        else:
            return True

    def _terminal_state(self):
        return np.ones(self.state.shape)

    def __repr__(self):
        return 'test'