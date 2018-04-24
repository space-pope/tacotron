from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import pandas as pd
import os
from os import path
from glob import glob
from hparams import hparams
from util import audio


_max_out_length = 700

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  prompts = []
  audio_files = []
  speakers = []

  # read label-info
  df = pd.read_table(path.join(in_dir, 'speaker-info.txt'), usecols=['ID'],
                     index_col=False, delim_whitespace=True)

  # assign speaker IDs
  speaker_ids = {str(uid): i for i, uid in enumerate(df.ID.values)}

  # read file IDs
  file_ids = []
  for d in [path.join(in_dir, 'txt', 'p{}'.format(uid)) for uid in df.ID.values]:
    file_ids.extend([f[-12:-4] for f in sorted(glob(d + '/*.txt'))])

  print("{} total files".format(len(file_ids)))
  return _process_files(file_ids, in_dir, out_dir, num_workers, tqdm)


def single_speaker(speaker_id, in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  # read file IDs
  s_dir = path.join(in_dir, 'txt', 'p{}'.format(speaker_id))
  file_ids = [f[-12:-4] for f in sorted(glob(s_dir + '/*.txt'))]
  print("{} total files".format(len(file_ids)))
  return _process_files(file_ids, in_dir, out_dir, num_workers, tqdm)


def _process_files(file_ids, in_dir, out_dir, num_workers, tqdm):
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  for f in file_ids:

    # wave file name
    audio_file = path.join(in_dir, 'wav48', f[:4], '{}.wav'.format(f))
    txt_file = path.join(in_dir, 'txt', f[:4], '{}.txt'.format(f))

    with open(txt_file, 'r') as tff:
      text = tff.read().strip()

    task = partial(_process_utterance, out_dir, f, audio_file, text)
    futures.append(executor.submit(task))

  results = [future.result() for future in tqdm(futures)]
  return [r for r in results if r is not None]



def _process_utterance(out_dir, file_id, wav_path, text):
  # Load the wav file and trim silence from the ends:
  wav = audio.load_wav(wav_path)
  max_samples = _max_out_length * hparams.frame_shift_ms / 1000 * hparams.sample_rate
  if len(wav) > max_samples:
    return None
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
  spectrogram_filename = 'vctk-spec-{}.npy'.format(file_id)
  mel_filename = 'vctk-mel-{}.npy'.format(file_id)
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
  return (spectrogram_filename, mel_filename, n_frames, text)

