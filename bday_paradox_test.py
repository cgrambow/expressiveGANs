import argparse
import glob
import os

import numpy as np

from util import imread, imsave


def main():
    args = parse_args()
    imdir = args.imdir
    nsim = args.nsim
    out_dir = args.out_dir

    impaths = glob.iglob(os.path.join(imdir, '*'))
    imgs = np.array([imread(path)/127.5 - 1.0 for path in impaths])  # Scale between -1 and 1

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

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_path = os.path.join(out_dir, 'similar_images.png')
    if os.path.exists(out_path):
        i = 2
        while os.path.exists(os.path.join(out_dir, 'similar_images_{}.png'.format(i))):
            i += 1
        out_path = os.path.join(out_dir, 'similar_images_{}.png'.format(i))

    # Save nsim most similar image pairs into one composite image
    sim_imgs = (imgs[inds_most_sim.flatten()]+1.0)/2.0
    manifold_size = (nsim, 2)
    imsave(sim_imgs, manifold_size, out_path)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('imdir', metavar='DIR',
                        help='Directory containing images')
    parser.add_argument('-n', '--nsim', type=int, default=20, metavar='N',
                        help='Number of most similar pairs to evaluate')
    parser.add_argument('-o', '--out_dir', default=os.getcwd(), metavar='D',
                        help='Output directory')
    return parser.parse_args()


if __name__ == '__main__':
    main()
