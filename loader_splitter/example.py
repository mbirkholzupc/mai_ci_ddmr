if __name__ == "__main__":
    from loader_splitter.Loader import Loader

    loader = Loader("../data", ["benign", "malignant"])

    data = loader.load()
    print(data.X)
    print(data.y)

    import matplotlib.pyplot as plt

    plt.suptitle(data.y[0])
    plt.imshow(data.X[0])
    plt.show()

    from loader_splitter.Splitter import Splitter

    splitter = Splitter()
    splits = splitter.split(data)
    print("Nº of splits: ", splits)
    plt.suptitle("First training set and image (%s)" % splits[0].train_set.y[0])
    plt.imshow(splits[0].train_set.X[0])
    plt.show()
    plt.suptitle("First test set and image (%s)" % splits[0].test_set.y[0])
    plt.imshow(splits[0].test_set.X[0])
    plt.show()
