from src.train import Train
from fundus_dataset.dataset import FundusDataset
import sys

def main():
    train = Train()
    # train.visualize_sample()
    train.train()
    train.logger.display_loss()
    # train.datautils.clear_cache()


if __name__ == '__main__':
    main()