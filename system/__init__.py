



from loguru import logger

from .kbrd_qwen import KBRDQwenSystem


system_register_table = {
    'KBRD_Qwen': KBRDQwenSystem
}


def get_system(PretrainModel, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
               interact=False, debug=False, tensorboard=False):
    """
    return the system class
    """
    model_name = opt['model_name']
    if model_name in system_register_table:
        system = system_register_table[model_name](PretrainModel, opt, train_dataloader, valid_dataloader, test_dataloader, vocab,
                                                   side_data, restore_system, interact, debug, tensorboard)
        logger.info(f'[Build system {model_name}]')
        return system
    else:
        raise NotImplementedError('The system with model [{}] in dataset [{}] has not been implemented'.
                                  format(model_name, opt['dataset']))