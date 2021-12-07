import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx


def separate_in_two_subsets(G, subset1, subset2):
    G2 = nx.Graph()

    edge_list1 = []
    edge_list2 = []
    for (i, j) in G.edges:
        if ((i in subset1) and (j in subset2)) or ((j in subset1) and (i in subset2)):
            edge_list2.append((i, j))
        else:
            edge_list1.append((i, j))

    print('number of cutting edges = ', len(edge_list2))

    G2.add_nodes_from(G.nodes)
    nodes_color = []
    for i in G.nodes:
        G2.add_node(i)
        if i in subset1:
            nodes_color.append('red')
        else:
            nodes_color.append('blue')

    G2.add_edges_from(edge_list1)
    G2.add_edges_from(edge_list2)

    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    nx.draw_networkx_nodes(G2, pos, node_size=700, node_color=nodes_color)

    # edges
    nx.draw_networkx_edges(G2, pos, edgelist=edge_list1, width=5)
    nx.draw_networkx_edges(G2, pos, edgelist=edge_list2, width=2, style='--')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    G = nx.Graph()

    edge_list = [(1, 2), (2, 4), (3, 5), (4, 3), (1, 3), (4, 5)]

    G.add_edges_from(edge_list)

    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='blue')

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=5)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()