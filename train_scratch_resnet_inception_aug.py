import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from models.MultipleAugmentedModelRunner import main


if __name__ == "__main__":
    main()
