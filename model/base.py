

from abc import ABC, abstractmethod

from torch import nn




class BaseModel(ABC, nn.Module):
    """Base class for all models"""

    def __init__(self, opt, device, dpath=None, resource=None):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.device = device

        self.build_model()

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """build model"""
        pass

    def recommend(self, batch, mode):
        """calculate loss and prediction of recommendation for batch under certain mode

        Args:
            batch (dict or tuple): batch data
            mode (str, optional): train/valid/test.
        """
        pass

    def converse(self, batch, mode):
        """calculate loss and prediction of conversation for batch under certain mode

        Args:
            batch (dict or tuple): batch data
            mode (str, optional): train/valid/test.
        """
        pass

    def guide(self, batch, mode):
        """calculate loss and prediction of guidance for batch under certain mode

        Args:
            batch (dict or tuple): batch data
            mode (str, optional): train/valid/test.
        """
        pass
