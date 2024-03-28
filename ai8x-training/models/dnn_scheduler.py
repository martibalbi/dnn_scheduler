"""
DNN model for TASA Scheduling on MAX78000

Optionally quantize/clamp activations
"""
from torch import nn

import ai8x

class DNN_Scheduler(nn.Module):

    def __init__(self, num_classes=420, num_channels=1, dimensions=(12,1), bias=False, scale=1., **kwargs):
        super().__init__()

        self.layer1 = ai8x.FusedLinearReLU(in_features=dimensions[0],out_features=int(400*scale),bias=bias)
        self.layer2 = ai8x.FusedLinearReLU(in_features=int(400*scale),out_features=int(512*scale),bias=bias)
        self.layer3 = ai8x.FusedLinearReLU(in_features=int(512*scale),out_features=int(512*scale),bias=bias)
        self.layer4 = ai8x.FusedLinearReLU(in_features=int(512*scale),out_features=int(512*scale),bias=bias)
        self.layer5 = ai8x.Linear(in_features=int(512*scale), out_features=420,bias=bias,wide=True,**kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


def dnn_scheduler(pretrained=False, **kwargs):
    """
    Constructs a DNN_Scheduler model.
    """
    assert not pretrained
    return DNN_Scheduler(**kwargs)

def dnn_scheduler_s_0_5(pretrained=False, **kwargs):
    """
    Constructs a DNN_Scheduler model.
    """
    assert not pretrained
    return DNN_Scheduler(scale=0.5, **kwargs)

models = [
    {
        'name': 'dnn_scheduler',
        'min_input': 1,
        'dim': 1,
    },
    {
        'name': 'dnn_scheduler_s_0_5',
        'min_input': 1,
        'dim': 1,
    },
]
