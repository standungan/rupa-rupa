import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


def l2_norm(x, axis=1):
    norm_x = torch.norm(x, 2, axis, True)
    output = torch.div(x, norm_x)
    return output

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(Bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1,1), stride, bias=False),
                nn.BatchNorm2d(depth)
                )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3,3), (1,1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3,3), stride, 1, bias=False),
            nn.BatchNorm2d(depth)
        )
    
    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        mod_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return mod_input * x

class Bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(Bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1,1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3,3), (1,1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3,3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''Named tuple untuk penamaan masing2 block pada Resnet'''

def get_block_bottleneck(in_channel, depth, num_units, stride=2):
    block = [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]
    return block

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block_bottleneck(in_channel=64, depth=64, num_units=3),
            get_block_bottleneck(in_channel=64, depth=128, num_units=4),
            get_block_bottleneck(in_channel=128, depth=256, num_units=14),
            get_block_bottleneck(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block_bottleneck(in_channel=64, depth=64, num_units=3),
            get_block_bottleneck(in_channel=64, depth=128, num_units=13),
            get_block_bottleneck(in_channel=128, depth=256, num_units=30),
            get_block_bottleneck(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block_bottleneck(in_channel=64, depth=64, num_units=3),
            get_block_bottleneck(in_channel=64, depth=128, num_units=8),
            get_block_bottleneck(in_channel=128, depth=256, num_units=36),
            get_block_bottleneck(in_channel=256, depth=512, num_units=3),
        ]
    return blocks

class Resnet(nn.Module):
    def __init__(self, embedding_size=512, num_layers=50, dropout_ratio=0.5, mode='ir_se'):
        super(Resnet, self).__init__()
        assert num_layers in [50,100,152] # jumlah layer yang bisa digunakan
        assert mode in ['ir', 'ir_se'] # tipe bottleneck yang bisa digunakan IR (Inverted Residual), IR_SE ()
        
        #buat block layers
        blocks = get_blocks(num_layers)
        
        #buat module IR atau IR_SE
        if mode == 'ir':
            unit_module = Bottleneck_IR
        elif mode == 'ir_se':
            unit_module = Bottleneck_IR_SE

        #buat layer : Input Layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        #buat layer : Body Layer
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride)
                )
        self.body_layer = nn.Sequential(*modules)

        #buat layer : Output Layer
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(dropout_ratio),
            Flatten(),
            nn.Linear(512 * 7 * 7, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body_layer(x)
        x = self.output_layer(x)
        return l2_norm(x)

if __name__ == "__main__":
    
    sample = torch.rand(2, 3, 128, 128)

    resnet50_IRSE = Resnet(num_layers=50, dropout_ratio=0.2, mode='ir_se')
    output = resnet50_IRSE(sample)
    
    print(output.shape)

    