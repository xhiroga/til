import networkx as nx
import matplotlib.pyplot as plt


def main():
    G = nx.Graph()
    print(G.nodes())  # returns a list
    print(G.edges())  # returns a list

    G.add_node("A")
    G.add_nodes_from(["B", "C", "D", "E"])

    G.add_edge(*("A", "B"))
    G.add_edges_from([("A", "C"), ("B", "D"), ("B", "E"), ("C", "E")])

    print("Vertex set: ", G.nodes())
    print("Edge set: ", G.edges())

    nx.draw(G)

    plt.show()


if __name__ == "__main__":
    main()