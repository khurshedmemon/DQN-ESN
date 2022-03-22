from __future__ import division, absolute_import, print_function
import graph_tool.all as gt
import sys
if sys.version_info < (3,):
    range = xrange
import os
import fileinput
import pdb

g = gt.Graph()
nv = dict()

count = 0
eprop_time_stamp = g.new_edge_property("int32_t")
g.ep.edge_cr_time = eprop_time_stamp

with open(sys.argv[1]) as f:
    next(f)
    for line in f:
        line_split = line.split()
        v1 = line_split[0]
        v2 = line_split[1]
        edge_time = line_split[2]
        e_time = int(edge_time)
        nv1 = 0
        nv2 = 0
        if v1 in nv:
            nv1 = nv[v1]
        else:
            nv[v1] = count
            count += 1
            nv1 = nv[v1]
        
        if v2 in nv:
            nv2 = nv[v2]
        else:
            nv[v2] = count
            count += 1
            nv2 = nv[v2]

        g.add_edge(nv1, nv2, True)
        g.ep.edge_cr_time[g.edge(g.vertex_index[nv1], g.vertex_index[nv2])] = e_time
        #pdb.set_trace()

#fileinput.close()

print("number of vertices: {0}".format(g.num_vertices(True)))
print("number of edges: {0}".format(g.num_edges(True)))

outputfile = sys.argv[1] + ".xml.gz"
print("transform to graph file to: {0}".format(outputfile))
g.save(outputfile)