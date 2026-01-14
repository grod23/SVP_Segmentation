from src import Train

def main():
    train = Train()
    train.visualize_sample()
    train.datautils.clear_cache()

if __name__ == '__main__':
    main()