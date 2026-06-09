from cgcnn.data import GraphData


def test():
    dataset = GraphData('data/graph_data')
    structure, target, mp_id = dataset[0]
    print(structure[1].shape)
    print(structure[2].shape)

    pass


if __name__ == "__main__":
    test()