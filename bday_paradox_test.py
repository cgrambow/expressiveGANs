import argparse
import glob
import os
import random

import numpy as np

from mcnn import MCNN
from util import imread, imsave


def main():
    args = parse_args()
    imdir = args.imdir
    nsamples = args.nsamples
    s = args.sample
    nsim = args.nsim
    model_path = args.model_path
    aux_model_path = args.aux_model_path
    out_dir = args.out_dir

    mcnn = aux = None
    if model_path:
        mcnn = MCNN(model_path=model_path, aux_model_path=aux_model_path)
        could_load = mcnn.load()
        if could_load == 2:
            aux = True
            print('Loaded base and aux model')
        elif could_load:
            aux = False
            print('Loaded base model (no aux)')

    impaths = glob.glob(os.path.join(imdir, '*'))[:nsamples]
    imlist = [imread(path)/127.5 - 1.0 for path in impaths]  # Scale between -1 and 1
    assert len(imlist) >= s
    random.shuffle(imlist)  # Do this just in case
    all_imgs = np.array(imlist)
    del imlist

    # Scale distances by image size
    size = all_imgs[0].size

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Find nsim most similar images for each sample
    duplicate_flags = []
    for sidx in range(len(all_imgs)//s):
        imgs = all_imgs[sidx*s:(sidx+1)*s]

        # Find Euclidian distances between pairs of images
        dists = np.zeros((len(imgs), len(imgs)))
        for i in range(len(imgs)):
            for j in range(i+1, len(imgs)):
                dists[i,j] = np.linalg.norm(imgs[i]-imgs[j]) / size

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
        imgs0 = imgs[inds_most_sim[:,0]]
        imgs1 = imgs[inds_most_sim[:,1]]
        attributes0 = mcnn.predict(imgs0, aux=aux)
        attributes1 = mcnn.predict(imgs1, aux=aux)
        for idx, (a0, a1) in enumerate(zip(attributes0, attributes1)):
            if np.array_equal(a0, a1):
                duplicate_flags.append(1)
                break
        else:
            duplicate_flags.append(0)

    print('Evaluated {} samples of size {}'.format(len(duplicate_flags), s))
    print('Probability of at least 1 pair of duplicates: {}'.format(sum(duplicate_flags)/len(duplicate_flags)))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('imdir', metavar='DIR',
                        help='Directory containing sampled images')
    parser.add_argument('-s', '--sample', type=int, metavar='S', help='Sample size')
    parser.add_argument('-n', '--nsim', type=int, default=20, metavar='N',
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
