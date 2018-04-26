import argparse

from mcnn import MCNN


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_height', default=108, type=int, help='Input image height')
    parser.add_argument('--input_width', default=108, type=int, help='Input image width')
    parser.add_argument('--output_height', default=64, type=int, help='Image height for training')
    parser.add_argument('--output_width', default=64, type=int, help='Image width for training')
    parser.add_argument('--no_crop', action='store_false', dest='crop', help='Do not crop images')
    parser.add_argument('--train_dataset_name', default='celebA', help='Name of training set folder')
    parser.add_argument('--test_dataset_name', default=None, help='Name of testing set folder')
    parser.add_argument('--data_dir', default='', help='Directory containing dataset folders')
    parser.add_argument('--input_fname_pattern', default='*.jpg', help='Glob pattern describing file names')
    parser.add_argument('--attribute_file_name', default='list_attr_celeba.txt', help='Name of attribute file')
    parser.add_argument('--model_path', default='weights.h5', help='Path to model weights')
    parser.add_argument('--aux_model_path', default='weights_aux.h5', help='Path to auxiliary model weights')
    parser.add_argument('--epochs', default=22, type=int, help='Number of epochs')
    parser.add_argument('--aux_epochs', default=10, type=int, help='Number of epochs for aux network')
    parser.add_argument('--train_size', default=-1, type=int, help='Number of training points to use (-1: all)')
    parser.add_argument('--test_size', default=-1, type=int, help='Number of testing points to use (-1: all)')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
    parser.add_argument('--aux', action='store_true', help='Train auxiliary net')
    parser.add_argument('--no_base_train', action='store_false', dest='train', help='Do not train base model')
    args = parser.parse_args()

    mcnn = MCNN(input_height=args.input_height,
                input_width=args.input_width,
                output_height=args.output_height,
                output_width=args.output_width,
                crop=args.crop,
                train_dataset_name=args.train_dataset_name,
                test_dataset_name=args.test_dataset_name,
                data_dir=args.data_dir,
                input_fname_pattern=args.input_fname_pattern,
                attribute_file_name=args.attribute_file_name,
                model_path=args.model_path,
                aux_model_path=args.aux_model_path,
                train_size=args.train_size,
                test_size=args.test_size)
    if args.train:
        mcnn.train(epochs=args.epochs,
                   batch_size=args.batch_size)
    if args.aux:
        mcnn.train(epochs=args.aux_epochs,
                   batch_size=args.batch_size,
                   aux=True)


if __name__ == '__main__':
    main()
