import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio


class Synthesizer:
  def __init__(self, teacher_forcing_generating=False):
    self.teacher_forcing_generating = teacher_forcing_generating
  def load(self, checkpoint_path, reference_mel=None, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    if reference_mel is not None:
      reference_mel = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'reference_mel')
    # Only used in teacher-forcing generating mode
    if self.teacher_forcing_generating:
      mel_targets = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'mel_targets')
    else:
      mel_targets = None

    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      if model_name == 'tacotron':
        self.model.initialize(inputs, input_lengths, mel_targets=mel_targets)
      else:
        self.model.initialize(inputs, input_lengths, mel_targets=mel_targets, reference_mel=reference_mel)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])
      self.mel_spectogram = self.model.mel_outputs[0]

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text, index=None, mel_targets=None, reference_mel=None):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
    }

    if index:
      style_params = {
        'inference_token': index,
        'style_weights': style_weights
      }
      style_weights = [[0.5 if t == index else 0.5/(hparams.num_gst - 1) for t in range(hparams.num_gst)]
                       for _ in range(hparams.num_heads)]
      feed_dict[self.model.inference_token] = style_params['inference_token']
      feed_dict[self.model.style_weights] = style_params['style_weights']

    if mel_targets is not None:
      mel_targets = np.expand_dims(mel_targets, 0)
      feed_dict.update({self.model.mel_targets: np.asarray(mel_targets, dtype=np.float32)})
    if reference_mel is not None:
      reference_mel = np.expand_dims(reference_mel, 0)
      feed_dict.update({self.model.reference_mel: np.asarray(reference_mel, dtype=np.float32)})
    wav, mel = self.session.run([self.wav_output, self.mel_spectogram], feed_dict=feed_dict)
   # wav, style_embeddings = self.session.run([self.wav_output, self.model.style_embeddings], feed_dict=feed_dict)
    wav = wav[:audio.find_endpoint(wav)]
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue(), mel

