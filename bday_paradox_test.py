import argparse
import glob
import os

import numpy as np

from egan.mcnn import MCNN
from egan.util import imread, imsave


def main():
    args = parse_args()
    imdir = args.imdir
    nsamples = args.nsamples
    sample_sizes = args.sample_size
    nsim = args.nsim
    model_path = args.model_path
    aux_model_path = args.aux_model_path
    out_dir = args.out_dir

    if sample_sizes is None:
        raise Exception('Specify sample size(s)')

    mcnn = aux = None
    could_load = False
    if model_path:
        mcnn = MCNN(model_path=model_path, aux_model_path=aux_model_path)
        could_load = mcnn.load()
        if could_load == 2:
            aux = True
            print('Loaded base and aux model\n')
        elif could_load:
            aux = False
            print('Loaded base model (no aux)\n')

    # Don't consider attributes 'smiling', 'wearing_lipstick', 'mouth_slightly_open', 'blurry', 'heavy_makeup'
    # because a human would also ignore them
    attribute_inds = np.array([0,1,2,3,7,8,9,10,11,12,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34])

    impaths = glob.glob(os.path.join(imdir, '*'))[:nsamples]
    imlist = [imread(path)/127.5 - 1.0 for path in impaths]  # Scale between -1 and 1
    assert all(len(imlist) >= s for s in sample_sizes)
    all_imgs = np.array(imlist)
    del imlist

    # Scale distances by image size
    img_size = all_imgs[0].size

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for s in sample_sizes:
        print('Sample size {}'.format(s))
        np.random.shuffle(all_imgs)

        # Find nsim most similar images for each sample
        duplicate_flags = []
        for sidx in range(len(all_imgs)//s):
            imgs = all_imgs[sidx*s:(sidx+1)*s]

            # Find Euclidian distances between pairs of images
            dists = np.zeros((len(imgs), len(imgs)))
            for i in range(len(imgs)):
                for j in range(i+1, len(imgs)):
                    dists[i,j] = np.linalg.norm(imgs[i]-imgs[j]) / img_size

            # Find nsim smallest distances
            inds = np.triu_indices_from(dists, k=1)
            inds_stacked = np.vstack(inds).T
            inds_sort = dists[inds].argsort()
            inds_most_sim = inds_stacked[inds_sort][:nsim]

            # Save nsim most similar image pairs into one composite image
            sim_imgs = (imgs[inds_most_sim.flatten()]+1.0)/2.0
            manifold_size = (nsim, 2)
            out_path = os.path.join(out_dir, 'ssize{}_sample{}.png'.format(s, sidx))
            imsave(sim_imgs, manifold_size, out_path)

            # Automatically detect duplicates
            if could_load:
                imgs0 = imgs[inds_most_sim[:,0]]
                imgs1 = imgs[inds_most_sim[:,1]]
                attributes0 = mcnn.predict(imgs0, aux=aux)
                attributes1 = mcnn.predict(imgs1, aux=aux)
                for idx, (a0, a1) in enumerate(zip(attributes0, attributes1)):
                    # Only retain the attributes we're interested in comparing
                    if np.array_equal(a0[attribute_inds], a1[attribute_inds]):
                        duplicate_flags.append(1)
                        dup_path = os.path.join(out_dir, 'ssize{}_sample{}_duplicate.png'.format(s, sidx))
                        imsave(np.array((imgs0[idx], imgs1[idx])), (1, 2), dup_path)
                        break
                else:
                    duplicate_flags.append(0)

        if duplicate_flags:
            print('Evaluated {} samples'.format(len(duplicate_flags)))
            print('Probability of at least 1 pair of duplicates: {}\n'.format(
                sum(duplicate_flags)/len(duplicate_flags)))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('imdir', metavar='DIR',
                        help='Directory containing sampled images')
    parser.add_argument('--sample_size', type=int, nargs='+', metavar='S', help='Sample size(s)')
    parser.add_argument('--nsim', type=int, default=20, metavar='N',
                        help='Number of most similar pairs to display per sample batch')
    parser.add_argument('--nsamples', type=int, default=100000, metavar='N',
                        help='Only use up to nsamples samples in imdir')
    parser.add_argument('--model_path', default='', metavar='model',
                        help='Trained MCNN model weights for celebA dataset to automatically determine duplicates')
    parser.add_argument('--aux_model_path', default='', metavar='model',
                        help='Trained AUX-MCNN model weights')
    parser.add_argument('-o', '--out_dir', default=os.getcwd(), metavar='D',
                        help='Output directory')
    return parser.parse_args()


if __name__ == '__main__':
    main()
