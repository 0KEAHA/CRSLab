from crslab.config import Config
from crslab.data.dataset.huatuo.huatuo import HuatuoDataset
from loguru import logger
import sys
import json

DEBUG = True
import os
os.environ["LOGURU_LEVEL"] = "DEBUG"
logger.add(sys.stderr, level="DEBUG" if DEBUG else "INFO", format="{time} {level} {message}")

config_file = './config/crs/kbrd/huatuo.yaml'

conf = Config(config_file)
print("Load config from {}".format(config_file))

r = HuatuoDataset(conf,conf['tokenize'])

train_data = r.train_data

with open('test_huatuo_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)