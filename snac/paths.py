"""
Functions that return strings for paths and such.

- Need to set environment variables:
    - SNAC (path to SNAC repo)

- Function naming convention:
  - "_filename" name of file only
  - "_path" full path to a directory
  - "_filepath" full path to a file

- Expected directory structure:
    $SNEC_MODEL/model/Data
"""

import os


def config_filepath(name='snec'):
    """
    Return path to config file
    parameters
    ----------
    name : str
    """
    if name is None:
        name = 'snec'

    try:
        snac_path = os.environ['SNAC']
    except KeyError:
        raise EnvironmentError('Environment variable SNAC not set. '
                               'Set path to code directory, e.g., '
                               "'export SNAC=${HOME}/path/to/SNAC'")

    return os.path.join(snac_path, 'snac', 'config', f'{name}.ini')


def data_path():
    """
    return path to $SNAC/snac/data
    """

    try:
        snac_path = os.environ['SNAC']
    except KeyError:
        raise EnvironmentError('Environment variable SNAC not set. '
                               'Set path to code directory, e.g., '
                               "'export SNAC=${HOME}/path/to/SNAC'")

    return os.path.join(snac_path, 'snac', 'data')


# ===============================================================
#                      Models
# ===============================================================
def model_path(model):
    """
    Return path to model directory
    parameters
    ----------
    model : str
        name of snec model directory
    """
    try:
        snec_models_path = os.environ['SNEC_MODELS']
    except KeyError:
        raise EnvironmentError('Environment variable SNEC_MODELS not set. '
                               'Set path to directory containing SNEC models, e.g., ')

    return os.path.join(snec_models_path, model)


def temp_path(model):
    """
    Path to directory for temporary file saving
    """
    m_path = model_path(model)
    return os.path.join(m_path, 'temp')


def output_path(model, output_dir='Data'):
    """
    Return path to model output directory
    """
    m_path = model_path(model)
    return os.path.join(m_path, output_dir)


# ===============================================================
#                      Dat files
# ===============================================================
def dat_filename(quantity):
    """
    Return filename for .dat file
    """
    return f'{quantity}.dat'


def dat_filepath(model, quantity):
    """
    Return filepath to .dat file
    parameters
    ----------
    model : str
    quantity : str
    """
    filename = dat_filename(quantity)
    d_path = output_path(model)
    return os.path.join(d_path, filename)


def dat_temp_filename(model):
    """
    Return filename for temporary (cached) dat file
    """
    return f'{model}_dat.pickle'


def dat_temp_filepath(model):
    """
    Return filepath to reduced dat table
    """
    path = temp_path(model)
    filename = dat_temp_filename(model)
    return os.path.join(path, filename)  


# ===============================================================
#                      Profiles
# ===============================================================
def profile_filename(quantity):
    """
    Return filename for .dat file
    Parameters:
    -----------
    quantity : str
    """
    return f'{quantity}.xg'


def profile_filepath(model, quantity):
    """
    Return filepath to .dat file

    parameters
    ----------
    model : str
    quantity : str
    """
    filename = profile_filename(quantity)
    d_path = output_path(model)
    return os.path.join(d_path, filename)


def profile_temp_filename(model):
    """
    Return filename for temporary (cached) dat file
    """
    return f'{model}_profile.pickle'


def profile_temp_filepath(model):
    """
    Return filepath to reduced dat table
    """
    path = temp_path(model)
    filename = profile_temp_filename(model)
    return os.path.join(path, filename) 