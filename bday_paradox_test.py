import argparse
import glob
import os
import random

import numpy as np

from util import imread, imsave


def main():
    args = parse_args()
    imdir = args.imdir
    s = args.sample
    nsim = args.nsim
    out_dir = args.out_dir

    impaths = glob.iglob(os.path.join(imdir, '*'))
    imlist = [imread(path)/127.5 - 1.0 for path in impaths]  # Scale between -1 and 1
    assert len(imlist) >= s
    random.shuffle(imlist)  # Do this just in case
    all_imgs = np.array(imlist)
    del imlist

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Find nsim most similar images for each sample
    for sidx in range(len(all_imgs)//s):
        imgs = all_imgs[sidx*s:(sidx+1)*s]

        # Find Euclidian distances between pairs of images
        dists = np.zeros((len(imgs), len(imgs)))
        for i in range(len(imgs)):
            for j in range(i+1, len(imgs)):
                dists[i,j] = np.linalg.norm(imgs[i]-imgs[j])

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


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('imdir', metavar='DIR',
                        help='Directory containing sampled images')
    parser.add_argument('-s', '--sample', type=int, metavar='S', help='Sample size')
    parser.add_argument('-n', '--nsim', type=int, default=20, metavar='N',
                        help='Number of most similar pairs to display per sample batch')
    parser.add_argument('-o', '--out_dir', default=os.getcwd(), metavar='D',
                        help='Output directory')
    return parser.parse_args()


if __name__ == '__main__':
    main()
