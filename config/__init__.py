import os
from .config import Config
from os.path import dirname, realpath


ROOT_PATH = dirname(dirname(dirname(realpath(__file__))))
SAVE_PATH = os.path.join('/hy-tmp/', 'save')
DATA_PATH = os.path.join('/hy-tmp/', 'data_file')
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
MODEL_PATH = os.path.join(DATA_PATH, 'model')
PRETRAIN_PATH = os.path.join(MODEL_PATH, 'pretrain')
EMBEDDING_PATH = os.path.join(DATA_PATH, 'embedding')
