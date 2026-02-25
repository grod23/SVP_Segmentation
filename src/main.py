from src.train import Train
import sys

def main():
    train = Train()
    # train.visualize_sample()
    train.train()
    # train.logger.display_loss()
    # train.save_model()
    train.test_model()

    # train.datautils.clear_cache()


if __name__ == '__main__':
    main()