from utils.config_file import config_file

def main_ci():
    from preprocess import preprocess_main
    from train import train_main
    from test import test_main

    print('Reading config file: \'config.txt\'')
    cfg = config_file('config.txt')
    print(cfg)

    # Right now, config_file object passed into each step. We may want to consider different
    # configs for each step, especially if we want to run preprocessing once and then
    # train/test multiple times
    pp_imgs = preprocess_main(cfg)
    print('Preprocessed images are in: ' + pp_imgs)
    train_main()
    test_main()

# Call it this way so we can call from external modules like jupyter notebook also
if __name__ == "__main__":
    main_ci()
