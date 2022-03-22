import argparse
import graph_tool.all as gt
import numpy as np
import node2vec
import random
import os
from sklearn.cluster import DBSCAN
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='./colgmsg.txt.xml.gz',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='./colmsg_emb_lists/',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=48,
                        help='Number of dimensions. Default is 48.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--eps', type=float, default=0.3,
                        help='The maximum distance between two samples for \
                        them to be considered as in the same neighborhood. \
                        Default is 0.3.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--unweighted', dest='unweighted', action='store_false')

    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')

    parser.add_argument('--undirected', dest='undirected', action='store_false')

    parser.set_defaults(directed=True)

    return parser.parse_args()


def read_graph():
    df = pd.read_csv(args.input, delim_whitespace=True)
    G = gt.Graph()

    if args.weighted:
        G.ep.weight = G.new_ep("double")
        G.add_edge_list(df.values, eprops=[G.ep.weight])
    else:
        G.add_edge_list(df.values)
        G.ep.weight = G.new_ep("double", val=1.0)

    G.set_directed(args.directed)

    return G


def learn_embeddings(walks, out_file):
    walks = [list(map(str, walk)) for walk in walks]
    #walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size,
                     min_count=0, sg=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(out_file)

    return model


def sort_embeddings(model, G):
    X = []
    for node in G.vertices():
        X.append(model.wv[str(node)])

    X = StandardScaler().fit_transform(X)

    return X


def main(args):
    if os.path.splitext(args.input)[1] == ".gz":
        gt_G = gt.load_graph(args.input)
        gt_G.ep.weight = gt_G.new_ep("double", val=1.0)        
    else:
        gt_G = read_graph()
    #check the time-stamped
    gt_G.ep.creat_time = gt_G.edge_properties["edge_cr_time"]
    edge_cr_time_arr = gt_G.ep.creat_time.a
    max_t = np.amax(edge_cr_time_arr)
    print('Edge creation time: ', edge_cr_time_arr, 'maximum time: ', max_t)
    #loop through each time-stamped sub-graph
    for i in range(1, max_t+1):
        print('iteration :', i)
        fl_G = gt.GraphView(gt_G, efilt=lambda e: gt_G.ep.creat_time[e] == i)
        sub_i = gt.Graph(fl_G, prune=True)
        #check the number of nodes and edges at each time-stamped graph
        print("number of vertices: ", sub_i.num_vertices())
        print("number of edges: ", sub_i.num_edges())
        '''
        v_rnd = random.choice(list(sub_i.vertices()))
        #v_rnd = sub_i.vertex_index[1]
        print('vertex: ', v_rnd, 'out degree: ', v_rnd.out_degree())        
        #check the out-edges of random vertex
        v_src = sub_i.vertex(v_rnd)
        v_out_edges = sub_i.get_out_edges(v_rnd)
        print('vertex: ', v_rnd, 'out-edges :', v_out_edges)
        for e in v_src.out_edges():
            print ('source node: ', v_src, 'edge: ', e)
        pdb.set_trace()
        '''
        out_file = args.output + 'colmsg.emb_' + str(i)
        #generate embeddings of each sub-graph
        G = node2vec.Graph(sub_i, args.directed, args.p, args.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(args.num_walks, args.walk_length)
        model = learn_embeddings(walks, out_file)
	    #X = sort_embeddings(model, sub_i)
	    #db = DBSCAN(eps=args.eps, min_samples=10).fit(X)
	    #oh = output.output_handler(sub_i, db, X)
	    #oh.draw_graph(str(args.p), str(args.q), str(args.walk_length), str(args.num_walks),
	    #              str(args.eps))
	    #oh.print_db_estimations()


if __name__ == "__main__":
    args = parse_args()
    main(args)
