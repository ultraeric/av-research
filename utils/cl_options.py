"""Command line arguments parser configuration."""
import argparse  # default python library for command line argument parsing
import os

parser = argparse.ArgumentParser(  # pylint: disable=invalid-name
    description='Train DNNs.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu', default=-1, type=int, help='Cuda GPU ID')
parser.add_argument('--config', default='default.json', type=str, help='Path of config file to use in ' +
                                                                       'experiment from ./configs')

ARGS = parser.parse_args()

# Check for $DISPLAY being blank
if 'DISPLAY' not in os.environ:
    ARGS.display = False
