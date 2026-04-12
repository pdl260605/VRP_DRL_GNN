import pickle
import os
import argparse
from datetime import datetime


def arg_parser_light():
    """Argument parser optimized for lightweight GNN model"""
    parser = argparse.ArgumentParser(description='Lightweight GNN VRP Solver')

    parser.add_argument('-m', '--mode', metavar='M', type=str, default='train',
                        choices=['train', 'test'], help='train or test')
    parser.add_argument('--seed', metavar='SE', type=int, default=123,
                        help='random seed for reproducibility')
    parser.add_argument('-n', '--n_customer', metavar='N', type=int, default=20,
                        help='number of customer nodes')

    # Training hyperparameters - OPTIMIZED FOR LIGHTWEIGHT MODEL
    parser.add_argument('-b', '--batch', metavar='B', type=int, default=256,
                        help='batch size (default: 256, reduced from 512)')
    parser.add_argument('-bs', '--batch_steps', metavar='BS', type=int, default=2500,
                        help='number of batch iterations per epoch')
    parser.add_argument('-bv', '--batch_verbose', metavar='BV', type=int, default=10,
                        help='print progress every N batches')
    parser.add_argument('-nr', '--n_rollout_samples', metavar='R', type=int, default=5000,
                        help='baseline samples (default: 5000, reduced from 10000)')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='number of training epochs')

    # Model architecture - LIGHTWEIGHT
    parser.add_argument('-em', '--embed_dim', metavar='EM', type=int, default=64,
                        help='embedding dimension (default: 64, reduced from 128)')
    parser.add_argument('-ne', '--n_encode_layers', metavar='NE', type=int, default=2,
                        help='number of GNN layers (default: 2, reduced from 3)')

    # Optimizer settings
    parser.add_argument('-c', '--tanh_clipping', metavar='C', type=float, default=10.,
                        help='tanh clipping value for exploration')
    parser.add_argument('--lr', metavar='LR', type=float, default=5e-4,
                        help='learning rate (default: 5e-4, slightly increased)')
    parser.add_argument('-wb', '--warmup_beta', metavar='WB', type=float, default=0.8,
                        help='exponential moving average for baseline warmup')
    parser.add_argument('-we', '--wp_epochs', metavar='WE', type=int, default=1,
                        help='number of warmup epochs')

    # Logging and saving
    parser.add_argument('--islogger', action='store_false', help='disable CSV logging')
    parser.add_argument('-ld', '--log_dir', metavar='LD', type=str, default='./Csv/',
                        help='directory for CSV logs')
    parser.add_argument('-wd', '--weight_dir', metavar='MD', type=str, default='./Weights/',
                        help='directory for model weights')
    parser.add_argument('-pd', '--pkl_dir', metavar='PD', type=str, default='./Pkl/',
                        help='directory for config pickle files')
    parser.add_argument('-cd', '--cuda_dv', metavar='CD', type=str, default='0',
                        help='CUDA device ID')

    args = parser.parse_args()
    return args


class Config():
    """Configuration class for lightweight GNN model"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

        # Create task name with GNN_Light identifier
        self.task = 'VRP%d_%s_GNN_Light' % (self.n_customer, self.mode)
        self.dump_date = datetime.now().strftime('%m%d_%H_%M')

        # Create directories
        for x in [self.log_dir, self.weight_dir, self.pkl_dir]:
            os.makedirs(x, exist_ok=True)

        self.pkl_path = self.pkl_dir + self.task + '.pkl'
        self.n_samples = self.batch * self.batch_steps


def dump_pkl(args, verbose=True, param_log=True):
    """Save configuration to pickle file"""
    cfg = Config(**vars(args))
    with open(cfg.pkl_path, 'wb') as f:
        pickle.dump(cfg, f)
        print('--- Config saved to %s ---\n' % cfg.pkl_path)

        if verbose:
            print('Configuration:')
            for k, v in vars(cfg).items():
                print(f'  {k}: {v}')

        if param_log:
            param_path = '%sparam_%s_%s.csv' % (cfg.log_dir, cfg.task, cfg.dump_date)
            with open(param_path, 'w') as pf:
                pf.write(''.join('%s,%s\n' % item for item in vars(cfg).items()))
            print(f'\nParameters logged to {param_path}')


def load_pkl(pkl_path, verbose=True):
    """Load configuration from pickle file"""
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f'Config file not found: {pkl_path}')

    with open(pkl_path, 'rb') as f:
        cfg = pickle.load(f)
        if verbose:
            print('Loading config from: %s\n' % pkl_path)
            for k, v in vars(cfg).items():
                print(f'{k}: {v}')
        os.environ['CUDA_VISIBLE_DEVICE'] = cfg.cuda_dv

    return cfg


def train_parser():
    """Parser for training script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', metavar='P', type=str,
                        default='Pkl/VRP20_train_GNN_Light.pkl',
                        help='Path to config pickle file')
    args = parser.parse_args()
    return args


def test_parser():
    """Parser for testing script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', metavar='P', type=str, required=True,
                        help='Path to model weights (.pt file)')
    parser.add_argument('-b', '--batch', metavar='B', type=int, default=2,
                        help='batch size for inference')
    parser.add_argument('-n', '--n_customer', metavar='N', type=int, default=20,
                        help='number of customer nodes')
    parser.add_argument('-s', '--seed', metavar='S', type=int, default=123,
                        help='random seed')
    parser.add_argument('-t', '--txt', metavar='T', type=str,
                        help='test data file (optional)')
    parser.add_argument('-d', '--decode_type', metavar='D', default='sampling',
                        type=str, choices=['greedy', 'sampling'],
                        help='decoding strategy')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser_light()
    dump_pkl(args)