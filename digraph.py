import networkx as nx
import matplotlib.pyplot as plt
import pylab
import pickle
import numpy as np


def draw_transition_table(transition_table,cluster_centers):
    plt.figure()
    G = nx.DiGraph()
    # print transition_table.sum(axis=1)

    transition_table = (transition_table.transpose()/transition_table.sum(axis=1)).transpose()
    transition_table[np.isnan(transition_table)]=0

    # transition_table = (transition_table.transpose()/transition_table.sum(axis=1)).transpose()
    # print transition_table
    # print transition_table.sum(axis=0)
    # assert(np.all(transition_table.sum(axis=0)!=0))
    transition_table[transition_table<0.1]=0

    pos = cluster_centers[:,0:2]
    m,n = transition_table.shape
    for i in range(m):
        for j in range(n):
            if transition_table[i,j]!=0:
                G.add_edges_from([(i, j)], weight=np.round(transition_table[i,j]*100)/100)

    values = cluster_centers[:,2]
    edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
    edge_colors = ['black' for edge in G.edges()]
    node_labels = {node:node for node in G.nodes()};
    counter=0
    for key in node_labels.keys():
        # node_labels[key] =  np.round(100*cluster_centers[counter,3])/100
        node_labels[key] =  counter
        counter+=1
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,label_pos=0.7,font_size=8)
    nx.draw_networkx_labels(G, pos, labels=node_labels,font_color='w',font_size=8)
    nx.draw(G,pos, node_color = values, node_size=np.round(3000*cluster_centers[:,3]),edge_color=edge_colors,edge_cmap=plt.cm.Reds)
    plt.show()



# transition_table = pickle.load(file('/home/tom/git/graying_the_box/data/breakout/120k' + '/knn/' + 'transition_table.bin'))
# cluster_centers = pickle.load(file('/home/tom/git/graying_the_box/data/breakout/120k' + '/knn/' + 'cluster_centers.bin'))
# draw_transition_table(transition_table,cluster_centers)


    # transition_table = pickle.load(file('/home/tom/git/graying_the_box/data/seaquest/120k' + '/knn/' + 'transition_table.bin'))
    # transition_table[transition_table<0.1]=0
    # cluster_centers = pickle.load(file('/home/tom/git/graying_the_box/data/seaquest/120k' + '/knn/' + 'cluster_centers.bin'))
