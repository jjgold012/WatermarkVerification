import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def visualize(epsilons, title="figure 1"):
        nn = nx.Graph()
        # adding nodes
        layers = [(0,epsilons.shape[0]), (1,epsilons.shape[1])]

        for i in range(epsilons.shape[0]):
            name = '{}_{}'.format(0, i)
            nn.add_node(name,
                        pos=(0, -i*1),
                        size=10
                        )
        for i in range(epsilons.shape[1]):
            name = '{}_{}'.format(1, i)
            nn.add_node(name,   
                        lable=i,
                        pos=(1, -i*15 - 7.5),
                        size=200
                        )
        # adding out_edges (no need to iterate over output layer)
        edges = list()
        weights = list()
        for i in range(epsilons.shape[0]):
            for j in range(epsilons.shape[1]):
                src = '{}_{}'.format(0, i)
                dest = '{}_{}'.format(1, j)

                visual_weight = epsilons[i][j]
                nn.add_edge(src, dest, weight=visual_weight)
                edges.append((src, dest))
                weights.append(visual_weight)

        pos = nx.get_node_attributes(nn,'pos')
        lables = nx.get_node_attributes(nn,'lable')
        nodes, sizes = zip(*nx.get_node_attributes(nn,'size').items())
        colors = nx.get_node_attributes(nn,'color')
        edges,weights = zip(*nx.get_edge_attributes(nn,'weight').items())
        
        print(np.min(weights))
        print(np.max(weights))
        maxWeight = np.max(np.abs(weights))
        plt.figure(figsize=(3,9))
        gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[29, 1]) 
        plt.subplot(gs[0])
        colormap = mpl.cm.coolwarm
        nx.draw(nn, pos,nodelist=nodes,
                        node_size=sizes,
                        node_color='black',
                        node_shape='o',
                        edgelist=edges,
                        edge_vmin=-maxWeight,
                        edge_vmax=maxWeight,
                        edge_color=weights,
                        edge_cmap=colormap,
                        with_labels=True,
                        font_color='white',
                        labels=lables,
                        linewidths=2,
                        width=0.5
                        )
        ax = plt.subplot(gs[1])
        norm = mpl.colors.Normalize(vmin=-maxWeight, vmax=maxWeight)
        mpl.colorbar.ColorbarBase(ax, cmap=colormap,
                                norm=norm,
                                orientation='horizontal',
                                ticks=[round(i,3) for i in np.linspace(-maxWeight, maxWeight, 5)],
                                )
        # plt.savefig('./data/results/problem2/last_layer_example.pdf', format='pdf')
        plt.show()

epsilons = np.load('./data/results/problem4/mnist.w.wm.2.wm_0-1.vals.npy')

visualize(epsilons[1])