from crslab.config import Config
from crslab.data.dataset.huatuo.huatuo import HuatuoDataset
from crslab.evaluator.metrics.gen import F1Metric, BleuMetric, DistMetric
from loguru import logger
import sys
import json

DEBUG = True
import os

guess = """
/API
当前位置 are here

 the by the. of Read’ll now professional question to Q

Ask

:

By一篇疼痛痛，等症状可能是后加重，可能是反痛等症状
是否有
:
是否有反、反等症状？


恶心、呕吐等症状
是否有user
是否有畏寒、发热等症状？
user
是否有畏寒、发热。


可能是反酸等症状？
user
是否有反酸等症状


可能是进行胃镜检查，明确胃癌等疾病。
"""

answer = ['建议进行胃镜检查，排除胃癌等疾病。']

for i in range(1, 5):
    print(f"bleu@{i}:", BleuMetric.compute(guess, answer, i))



# os.environ["LOGURU_LEVEL"] = "DEBUG"
# logger.add(sys.stderr, level="DEBUG" if DEBUG else "INFO", format="{time} {level} {message}")

# config_file = './config/crs/kbrd/huatuo.yaml'

# conf = Config(config_file)
# print("Load config from {}".format(config_file))

# r = HuatuoDataset(conf,conf['tokenize'])

# train_data = r.train_data

# with open('test_huatuo_train_data.json', 'w', encoding='utf-8') as f:
#     json.dump(train_data, f, ensure_ascii=False, indent=4)