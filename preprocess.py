import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import ailabs, blizzard, librivox, ljspeech, vctk
from hparams import hparams


def preprocess_blizzard(args):
  in_dir = os.path.join(args.base_dir, 'Blizzard2012')
  out_dir = os.path.join('.', args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = blizzard.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_ljspeech(args):
  in_dir = os.path.join(args.base_dir, 'LJSpeech-1.1')
  out_dir = os.path.join('.', args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = ljspeech.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_ailabs(args):
  in_dir = os.path.join(args.base_dir, 'M-AILABS-en_US',
          'by_book', 'female', 'judy_bieber')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = ailabs.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_librivox(args):
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = librivox.build_from_path(args.base_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_vctk(args):
  in_dir = os.path.join(args.base_dir, 'vctk')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  if args.speaker_id:
    print("Generating training data for speaker {}".format(args.speaker_id))
    metadata = vctk.single_speaker(args.speaker_id, in_dir, out_dir,
                                   args.num_workers, tqdm=tqdm)
  else:
    metadata = vctk.build_from_path(in_dir, out_dir, args.num_workers,
                                    tqdm=tqdm)
  write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')
  frames = sum([m[2] for m in metadata])
  hours = frames * hparams.frame_shift_ms / (3600 * 1000)
  print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  print('Max input length:  %d' % max(len(m[3]) for m in metadata))
  print('Max output length: %d' % max(m[2] for m in metadata))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default='/var/pylon/data/speech')
  parser.add_argument('--output', default='training')
  parser.add_argument('--dataset', required=True,
                      choices=['ailabs', 'blizzard', 'librivox', 'ljspeech',
                               'vctk'])
  parser.add_argument('--speaker_id', default=None)
  parser.add_argument('--num_workers', type=int, default=cpu_count()-3)
  args = parser.parse_args()
  if args.dataset == 'blizzard':
    preprocess_blizzard(args)
  elif args.dataset == 'ailabs':
    preprocess_ailabs(args)
  elif args.dataset == 'librivox':
    hparams.sample_rate = 22050
    preprocess_librivox(args)
  elif args.dataset == 'ljspeech':
    preprocess_ljspeech(args)
  elif args.dataset == 'vctk':
    hparams.sample_rate = 24000
    preprocess_vctk(args)


if __name__ == "__main__":
  main()
