from cgcnn.data import GraphData


def test():
    dataset = GraphData('data/graph_data')
    print(dataset[0])

    pass


if __name__ == "__main__":
    test()