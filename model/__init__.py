

import torch
from loguru import logger

from model.kbrd import KBRDModel

Model_register_table = {
    'KBRD': KBRDModel

}


def get_model(config, model_name, device, vocab, side_data=None):
    if model_name in Model_register_table:
        model = Model_register_table[model_name](config, device, vocab, side_data)
        logger.info(f'[Build model {model_name}]')
        if config.opt["gpu"] == [-1]:
            return model
        else:
            if len(config.opt["gpu"]) > 1:
                if model_name == 'PMI' or model_name == 'KBRD':
                    logger.info(f'[PMI/KBRD model does not support multi GPUs yet, using single GPU now]')
                    return model.to(device)
                else:
                    return torch.nn.DataParallel(model, device_ids=config["gpu"])
            else:
                return model.to(device)

    else:
        raise NotImplementedError('Model [{}] has not been implemented'.format(model_name))
