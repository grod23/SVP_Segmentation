from src.train import Train
from fundus_dataset.dataset import FundusDataset
import sys

def main():
    train = Train()
    for batch in train.training_loader:
        print(f'Batch: {batch}')
        print(f'Batch Type: {type(batch)}')
        print(f'Batch Shape: {batch.shape}')
        sys.exit()

    # train.visualize_sample()
    # train.run_epoch()
    # train.datautils.clear_cache()


if __name__ == '__main__':
    main()