import numpy as np
from pathlib import Path
import argparse
from argparse import ArgumentParser, Namespace

from seis_model import SeisModel
from utils import fix_seed


def main(args: Namespace):
    fix_seed(args.seed)
    model_generator = SeisModel(interp_signal=args.interp_signal,
                                min_thickness=args.min_thickness,
                                max_thickness=args.max_thickness,
                                smoothness=args.smoothness,
                                height=args.height,
                                width=args.width,
                                filename_signal=args.filename_signal)

    for k in range(args.num_data):
        model_and_picking = model_generator.get_random_model()
        np.save(args.data_dir / f'model_{k}.npy', model_and_picking)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', type=Path,
                        help='Directory to store models')
    parser.add_argument('--num_data', dest='num_data', type=int,
                        help='Number of generated model')

    parser.add_argument('--smoothness', dest='smoothness', type=int, default=5,
                        help='Smoothness of the variability of the travel time curves of the waves')
    parser.add_argument('--height', dest='height', type=int, default=1000,
                        help='Height of model in samples')
    parser.add_argument('--width', dest='width', type=int, default=24,
                        help='Width of model in samples')
    parser.add_argument('--min_thickness', dest='min_thickness', type=int, default=30,
                        help='Minimum distance between layers in samples')
    parser.add_argument('--max_thickness', dest='max_thickness', type=int, default=80,
                        help='Maximum distance between layers in samples')
    parser.add_argument('--interp_signal', dest='interp_signal', nargs=2, type=int, default=(4, 8),
                        help='Two numbers with minimum and maximum factor of signal pulse interpolation')
    parser.add_argument('--file_signal', dest='filename_signal', type=str, default='',
                        help='Select a file with signal with specific name (without extension).'
                             'If nothing is specified, then a random file is selected with each generation')
    parser.add_argument('--random_seed', dest='seed', type=int, default=1,
                        help='Fix seeds for reproducibility')
    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    main(args=arg_parser.parse_args())

