import argparse
from argparse import ArgumentParser, Namespace
from datetime import datetime
import ast

from pathlib import Path
import torch
import pandas as pd

from utils import fix_seed, split_dataset, Stopper
from network import UNetFB
from picker import Picker
from trainer import Trainer
from seis_dataset import SeisDataset


def main(args: Namespace):
    results_path = args.log_dir / str(datetime.now())

    results_path.mkdir(exist_ok=True, parents=True)
    write_args(results_path, vars(args))

    fix_seed(args.seed)

    height_model = 1000
    width_model = 24

    filenames_train, filenames_valid, filenames_test = split_dataset(args.data_root, args.fracs_dataset)

    train_set = SeisDataset(filenames_train,
                            height_model=height_model,
                            width_model=width_model,
                            prob_aug=args.prob_aug)

    valid_set = SeisDataset(filenames_valid,
                            height_model=height_model,
                            width_model=width_model,
                            prob_aug=args.prob_aug)

    test_set = SeisDataset(filenames_test,
                           height_model=height_model,
                           width_model=width_model,
                           prob_aug=args.prob_aug)

    net = UNetFB()
    picker = Picker(net)

    stopper = Stopper(args.n_wrongs, args.delta_wrongs)

    trainer = Trainer(picker=picker, results_path=results_path,
                      train_set=train_set, valid_set=valid_set,
                      test_set=test_set, device=args.device,
                      batch_size=args.batch_size, lr=args.lr,
                      freq_valid=args.freq_valid, num_workers=args.num_workers,
                      dt_ms=args.dt_ms, height_model=height_model,
                      width_model=width_model, visual=args.visual,
                      stopper=stopper, weights=torch.tensor(args.weights))

    trainer.train(num_epoch=args.num_epoch)


def write_args(path: Path, args_dict: dict) -> None:
    frame_args = pd.DataFrame.from_dict(args_dict, orient="index")
    frame_args.to_csv(str(path / 'args.txt'), header=False, sep='=', mode='w+')


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', dest='data_root', type=Path)
    parser.add_argument('--log_dir', dest='log_dir', type=Path)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--num_epoch', dest='num_epoch', type=int, default=3)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=60)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=4)
    parser.add_argument('--device', dest='device', type=torch.device, default='cuda')
    parser.add_argument('--random_seed', dest='seed', type=int, default=1)
    parser.add_argument('--fracs_dataset', dest='fracs_dataset', type=ast.literal_eval, default="(0.8, 0.1, 0.1)")
    parser.add_argument('--prob_aug', dest='prob_aug', type=float, default=0.8)
    parser.add_argument('--dt_ms', dest='dt_ms', type=float, default=1)
    parser.add_argument('--freq_valid', dest='freq_valid', type=float, default=100)
    parser.add_argument('--visual', dest='visual', type=ast.literal_eval,
                        default="{'train': [50, 4], 'valid': [10, 4], 'test': [15, 5]}",
                        help="Set the frequency (the first number - every 'K' batches) "
                             "and the number of visualizations (second number) for each training stage.")
    parser.add_argument('--weights', dest='weights', type=ast.literal_eval, default="(0.005, 0.015, 0.98)")
    parser.add_argument('--n_wrongs', dest='n_wrongs', type=int, default=10)
    parser.add_argument('--delta_wrongs', dest='delta_wrongs', type=float, default=0.01)

    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    main(args=arg_parser.parse_args())
