from src.train import Train


def main():
    train = Train()
    # train.diagnose_masks()
    train.train()
    train.tester.visuals.display_training_loss()
    # train.save_model()
    train.test_model(load_model=False)

    # train.datautils.clear_cache()


if __name__ == '__main__':
    main()