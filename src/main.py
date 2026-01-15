from src.train import Train

def main():
    train = Train()
    train.visualize_sample()
    # train.datautils.clear_cache()
    train.run_epoch()

if __name__ == '__main__':
    main()