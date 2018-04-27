import math
import numpy as np
import os
import random
import tensorflow as tf
import threading
import time
import traceback
from text import cmudict, text_to_sequence
from util.infolog import log


_batches_per_group = 32
_p_cmudict = 0.5
_p_train_data = 0.8
_pad = 0


class DataFeeder(threading.Thread):
  '''Feeds batches of data into a queue on a background thread.'''

  def __init__(self, coordinator, metadata_filename, hparams):
    super(DataFeeder, self).__init__()
    self._coord = coordinator
    self._hparams = hparams
    self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    self._offset = 0

    # Load metadata:
    self._datadir = os.path.dirname(metadata_filename)
    with open(metadata_filename, encoding='utf-8') as f:
      self._metadata = [line.strip().split('|') for line in f]
      # round up the test samples to the nearest batch
      num_samples = len(self._metadata)
      test_samples = int(num_samples * (1.0 - _p_train_data))
      test_samples += hparams.batch_size - (test_samples % hparams.batch_size)
      self._max_train_offset = num_samples - test_samples
      hours = sum((int(x[2]) for x in self._metadata[:self._max_train_offset])) \
              * hparams.frame_shift_ms / (3600 * 1000)
      log('Loaded metadata for %d examples (%.2f hours)' % (self._max_train_offset, hours))
      self.epoch_length = math.ceil(self._max_train_offset / hparams.batch_size)
      self._test_meta = self._metadata[self._max_train_offset:]
      self.test_batches = test_samples // hparams.batch_size

    # Create placeholders for inputs and targets. Don't specify batch size because we want to
    # be able to feed different sized batches at eval time.
    self._placeholders = [
      tf.placeholder(tf.int32, [None, None], 'inputs'),
      tf.placeholder(tf.int32, [None], 'input_lengths'),
      tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
      tf.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets')
    ]

    # Create queue for buffering data:
    queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32], name='input_queue')
    self._enqueue_op = queue.enqueue(self._placeholders)
    self.inputs, self.input_lengths, self.mel_targets, self.linear_targets = queue.dequeue()
    self.inputs.set_shape(self._placeholders[0].shape)
    self.input_lengths.set_shape(self._placeholders[1].shape)
    self.mel_targets.set_shape(self._placeholders[2].shape)
    self.linear_targets.set_shape(self._placeholders[3].shape)

    # Create a second queue for test data
    self._test_placeholders = [
      tf.placeholder(tf.int32, [None, None], 'test_inputs'),
      tf.placeholder(tf.int32, [None], 'test_input_lengths'),
      tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'test_mel_targets'),
    ]
    test_queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32], name='test_queue')
    # self._enqueue_test_op = test_queue.enqueue(self._test_placeholders)
    self.test_inputs, self.test_input_lengths, self.test_mel_targets = test_queue.dequeue()
    self.test_inputs.set_shape(self._test_placeholders[0].shape)
    self.test_input_lengths.set_shape(self._test_placeholders[1].shape)
    self.test_mel_targets.set_shape(self._test_placeholders[2].shape)

    # Load CMUDict: If enabled, this will randomly substitute some words in the training data with
    # their IPA equivalents, which will allow you to also pass IPA to the model for
    # synthesis (useful for proper nouns, etc.)
    if hparams.use_cmudict:
      cmudict_path = os.path.join(self._datadir, 'cmudict-0.7b-ipa.txt')
      if not os.path.isfile(cmudict_path):
        raise Exception('If use_cmudict=True, you must download ' +
          'https://raw.githubusercontent.com/menelik3/cmudict-ipa/master/cmudict-0.7b-ipa.txt'  % cmudict_path)
      self._cmudict = cmudict.CMUDict(cmudict_path, keep_ambiguous=False)
      log('Loaded CMUDict with %d unambiguous entries' % len(self._cmudict))
    else:
      self._cmudict = None


  def limit_data(self, max_hours, hparams):
    cutoff = self._max_train_offset
    total = 0
    adjusted_max = max_hours * 3600000 / hparams.frame_shift_ms
    for i, x in enumerate(self._metadata):
      total += int(x[2])
      if total >= adjusted_max:
        cutoff = i
        break
    self._metadata = self._metadata[:cutoff + 1]
    self._max_train_offset = int(cutoff * _p_train_data)
    hours = total * hparams.frame_shift_ms / 3600000
    log('Limited metadata to %d examples (%.2f hours)' % (cutoff, hours))


  def start_in_session(self, session):
    self._session = session
    self.start()


  def run(self):
    try:
      while not self._coord.should_stop():
        self._enqueue_next_group()
    except Exception as e:
      traceback.print_exc()
      self._coord.request_stop(e)


  def _enqueue_next_group(self):
    start = time.time()

    # Read a group of examples:
    n = self._hparams.batch_size
    r = self._hparams.outputs_per_step
    examples = [self._get_next_example() for i in range(n * _batches_per_group)]

    batches = _batch_examples(examples, n)
    log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
    for batch in batches:
      feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
      self._session.run(self._enqueue_op, feed_dict=feed_dict)

  def _get_next_example(self):
    '''Loads a single example (input, mel_target, linear_target, cost) from disk'''
    if self._offset >= self._max_train_offset:
      self._offset = 0
      random.shuffle(self._metadata)
    example = self._load_example(self._offset, self._metadata)
    self._offset += 1
    return example

  def _load_example(self, index, metadata):
    meta = metadata[index]
    text = meta[3]
    if self._cmudict and random.random() < _p_cmudict:
      text = ' '.join([self._maybe_get_ipa(word) for word in text.split(' ')])

    input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
    linear_target = np.load(os.path.join(self._datadir, meta[0]))
    mel_target = np.load(os.path.join(self._datadir, meta[1]))
    return (input_data, mel_target, linear_target, len(linear_target))

  def _maybe_get_ipa(self, word):
    strip_emphasis = random.random() < 0.7
    ipa = self._cmudict.lookup(word, strip_emphasis)
    return '{%s}' % ipa[0] if ipa is not None and random.random() < 0.5 else word

  def fetch_test_data(self):
    start = time.time()
    # Read a group of examples:
    n = self._hparams.batch_size
    r = self._hparams.outputs_per_step
    examples = [self._load_example(i, self._test_meta)
                for i in range(0, len(self._test_meta))]
    batches = _batch_examples(examples, n)

    for batch in batches:
      yield dict(zip(self._test_placeholders, _prepare_batch(batch, r)[:-1]))
    self._test_meta = self._metadata[self._max_train_offset:]


def _batch_examples(examples, n):
  # Bucket examples based on similar output sequence length for efficiency:
  examples.sort(key=lambda x: x[-1])
  batches = [examples[i:i+n] for i in range(0, len(examples), n)]
  random.shuffle(batches)
  return batches


def _prepare_batch(batch, outputs_per_step):
  random.shuffle(batch)
  inputs = _prepare_inputs([x[0] for x in batch])
  input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
  mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
  linear_targets = _prepare_targets([x[2] for x in batch], outputs_per_step)
  return (inputs, input_lengths, mel_targets, linear_targets)


def _prepare_inputs(inputs):
  max_len = max((len(x) for x in inputs))
  return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, alignment):
  max_len = max((len(t) for t in targets)) + 1
  return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
  return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
  return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)


def _round_up(x, multiple):
  remainder = x % multiple
  return x if remainder == 0 else x + multiple - remainder
