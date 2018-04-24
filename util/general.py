""" General utility methods. """
from models.tacotron import Tacotron
from models.gst_tacotron_2 import GSTTacotron2


def create_model(name, hparams):
  if name == 'tacotron':
    return Tacotron(hparams)
  elif name == 'gst_tacotron_2':
    return GSTTacotron2(hparams)
  else:
    raise Exception('Unknown model: ' + name)
