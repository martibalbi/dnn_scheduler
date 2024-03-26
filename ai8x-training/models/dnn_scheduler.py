"""
DNN model for TASA Scheduling on MAX78000

Optionally quantize/clamp activations
"""
from torch import nn

import ai8x

class DNN_Scheduler(nn.Module):

    def __init__(self, num_classes=420, num_channels=1, dimensions=(12,), bias=False, **kwargs):
        super().__init__()

        self.layer1 = ai8x.FusedLinearReLU(in_features=dimensions[0],out_features=800,bias=bias)
        self.layer2 = ai8x.FusedLinearReLU(in_features=800,out_features=1024,bias=bias)
        self.layer3 = ai8x.FusedLinearReLU(in_features=1024,out_features=1024,bias=bias)
        self.layer4 = ai8x.FusedLinearReLU(in_features=1024,out_features=800,bias=bias)
        self.layer5 = ai8x.Linear(in_features=800, out_features=num_classes,bias=bias,wide=True,**kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5

        return x


def dnn_scheduler(pretrained=False, **kwargs):
    """
    Constructs a DNN_Scheduler model.
    """
    assert not pretrained
    return DNN_Scheduler(**kwargs)

models = [
    {
        'name': 'dnn_scheduler',
        'min_input': 1,
        'dim': 1,
    },
]
