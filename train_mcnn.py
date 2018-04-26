import argparse

import numpy as np

from mcnn import MCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_name', default='celebA', help='Name of training set folder')
    parser.add_argument('--test_dataset_name', default=None, help='Name of testing set folder')
    parser.add_argument('--data_dir', default='', help='Directory containing dataset folders')
    parser.add_argument('--epochs', default=22, help='Number of epochs')
    parser.add_argument('--aux_epochs', default=10, help='Number of epochs for aux network')
    parser.add_argument('--train_size', default=np.inf, help='Number of training points to use')
    parser.add_argument('--test_size', default=np.inf, help='Number of testing points to use')
    parser.add_argument('--aux', action='store_true', help='Train auxiliary net')
    args = parser.parse_args()

    mcnn = MCNN(train_dataset_name=args.train_dataset_name,
                test_dataset_name=args.test_dataset_name,
                data_dir=args.data_dir)
    mcnn.train(epochs=args.epochs,
               train_size=args.train_size,
               test_size=args.test_size)
    if args.aux:
        mcnn.train(epochs=args.aux_epochs,
                   train_size=args.train_size,
                   test_size=args.test_size,
                   aux=True)


if __name__ == '__main__':
    main()
