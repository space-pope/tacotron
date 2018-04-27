import argparse
import os

from hparams import hparams
from train import train
from util import infolog


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/data'))
  parser.add_argument('--input', default='training/train.txt')
  parser.add_argument('--model', default='gst_tacotron_2')
  parser.add_argument('--name', help='name of the run. used for logging. defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='global step to restore from checkpoint.')
  parser.add_argument('--gpu_fraction', type=float, default=0.7, help='fraction of gpu memory to use')
  parser.add_argument('--max_hours', type=float, default=0,
                      help='maximum number of hours of training data to use')
  parser.add_argument('--max_steps', type=int, default=0,
                      help='maximum number of training steps to run')
  parser.add_argument('--max_epochs', type=int, default=0,
                      help='maximum number of training epochs to run')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='slack webhook url to get periodic reports.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='tensorflow c++ log level.')
  parser.add_argument('--git', action='store_true', help='if set, verify that the client is clean.')
  args = parser.parse_args()
  os.environ['tf_cpp_min_log_level'] = str(args.tf_log_level)
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  for run_params in args.hparams.split(';'):
      run_log_dir = os.path.join(log_dir, run_params)
      os.makedirs(run_log_dir, exist_ok=True)
      infolog.init(os.path.join(run_log_dir, 'train.log'), run_name,
                   args.slack_url)
      args.hparams = run_params
      hparams.parse(args.hparams)
      train(run_log_dir, args)


if __name__ == '__main__':
  main()
