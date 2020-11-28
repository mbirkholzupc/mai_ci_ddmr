if __name__ == "__main__":
    from preprocess import preprocess_main
    from train import train_main
    from test import test_main

    preprocess_main()
    train_main()
    test_main()
