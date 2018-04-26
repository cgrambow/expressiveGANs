import argparse

from mcnn import MCNN


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--train_dataset_name', default='celebA', help='Name of training set folder')
    parser.add_argument('--test_dataset_name', default=None, help='Name of testing set folder')
    parser.add_argument('--data_dir', default='', help='Directory containing dataset folders')
    parser.add_argument('--epochs', default=22, type=int, help='Number of epochs')
    parser.add_argument('--aux_epochs', default=10, type=int, help='Number of epochs for aux network')
    parser.add_argument('--train_size', default=-1, type=int, help='Number of training points to use (-1: all)')
    parser.add_argument('--test_size', default=-1, type=int, help='Number of testing points to use (-1: all)')
    parser.add_argument('--aux', action='store_true', help='Train auxiliary net')
    args = parser.parse_args()

    mcnn = MCNN(train_dataset_name=args.train_dataset_name,
                test_dataset_name=args.test_dataset_name,
                data_dir=args.data_dir,
                train_size=args.train_size,
                test_size=args.test_size)
    mcnn.train(epochs=args.epochs)
    if args.aux:
        mcnn.train(epochs=args.aux_epochs,
                   aux=True)


if __name__ == '__main__':
    main()
