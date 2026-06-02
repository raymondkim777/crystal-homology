import pickle
import networkx as nx
import matplotlib.pyplot as plt


with open('data/graphs/triclinic.pkl', 'rb') as f:
    graph_list = pickle.load(f)

# print(sorted(list(graph_list.keys())))

nx_graph = graph_list['mp-2856']

# Draw the graph with labels
pos = nx.spring_layout(nx_graph)

nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=800)
edge_labels = nx.get_edge_attributes(nx_graph, "weight")
formatted_labels = {edge: f"{weight:.3f}" for edge, weight in edge_labels.items()}
nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=formatted_labels)

# Display the plot
plt.show()