import torch.nn as nn
import math
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None,
            dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.LeakyReLU(inplace=True)

        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3,
                    stride=stride, padding=1, output_padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3,
                    stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class ACNet(nn.Module):
    def __init__(self, num_class):
        super(ACNet, self).__init__()

        layers = [3,4,6,3]
        block = Bottleneck
        transblock = TransBasicBlock
        
        # rgb branch
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1],stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # depth branch
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.relu_d = nn.LeakyReLU(inplace=True)
        
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = 64
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1],stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        # merge branch
        self.atten_rgb_0 = self.channel_attention(64)
        self.atten_depth_0 = self.channel_attention(64)
        self.maxpool_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.atten_rgb_1 = self.channel_attention(64*4)
        self.atten_depth_1 = self.channel_attention(64*4)
        self.atten_rgb_2 = self.channel_attention(128*4)
        self.atten_depth_2 = self.channel_attention(128*4)
        self.atten_rgb_3 = self.channel_attention(256*4)
        self.atten_depth_3 = self.channel_attention(256*4)
        self.atten_rgb_4 = self.channel_attention(512*4)
        self.atten_depth_4 = self.channel_attention(512*4)

        self.inplanes = 64
        self.layer1_m = self._make_layer(block, 64, layers[0])
        self.layer2_m = self._make_layer(block, 128, layers[1],stride=2)
        self.layer3_m = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_m = self._make_layer(block, 512, layers[3], stride=2)

        # agant module
        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64*4, 64)
        self.agant2 = self._make_agant_layer(128*4, 128)
        self.agant3 = self._make_agant_layer(256*4, 256)
        self.agant4 = self._make_agant_layer(512*4, 512)
        
        # transpose layer
        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)

        # final block
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.final_deconv = nn.ConvTranspose2d(self.inplanes, num_class,
                kernel_size=2, stride=2, padding=0, bias=True)

        self.out5_conv = nn.Conv2d(256, num_class, kernel_size=1, stride=1,
                bias=True)
        self.out4_conv = nn.Conv2d(128, num_class, kernel_size=1, stride=1,
                bias=True)
        self.out3_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1,
                bias=True)
        self.out2_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1,
                bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def channel_attention(self, num_channel, ablation=False):
        pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
        activation = nn.Sigmoid()
        return nn.Sequential(*[pool, conv, activation])
    
    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                    stride=1, padding=0, bias=False),
                nn.BatchNorm2d(planes),
                nn.LeakyReLU(inplace=True)
                )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                    nn.ConvTranspose2d(self.inplanes, planes,
                        kernel_size=2, stride=stride,
                        padding=0, bias=False),
                    nn.BatchNorm2d(planes)
                    )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes,
                        kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes)
                    )

        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def encoder(self, rgb, depth):
        rgb = self.conv1(rgb)
        rgb = self.bn1(rgb)
        rgb = self.relu(rgb)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu_d(depth)

        atten_rgb = self.atten_rgb_0(rgb)
        atten_depth = self.atten_depth_0(depth)
        m0 = rgb.mul(atten_rgb) + depth.mul(atten_depth)

        rgb = self.maxpool(rgb)
        depth = self.maxpool_d(depth)
        m = self.maxpool_m(m0)

        rgb = self.layer1(rgb)
        depth = self.layer1_d(depth)
        m = self.layer1_m(m)
        atten_rgb = self.atten_rgb_1(rgb)
        atten_depth = self.atten_depth_1(depth)
        m1 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)
   
        rgb = self.layer2(rgb)
        depth = self.layer2_d(depth)
        m = self.layer2_m(m1)
        atten_rgb = self.atten_rgb_2(rgb)
        atten_depth = self.atten_depth_2(depth)
        m2 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

        rgb = self.layer3(rgb)
        depth = self.layer3_d(depth)
        m = self.layer3_m(m2)
        atten_rgb = self.atten_rgb_3(rgb)
        atten_depth = self.atten_depth_3(depth)
        m3 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

        rgb = self.layer4(rgb)
        depth = self.layer4_d(depth)
        m = self.layer4_m(m3)
        atten_rgb = self.atten_rgb_4(rgb)
        atten_depth = self.atten_depth_4(depth)
        m4 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

        return m0, m1, m2, m3, m4

    def decoder(self, fuse0, fuse1, fuse2, fuse3, fuse4):
        agant4 = self.agant4(fuse4)
        
        x = self.deconv1(agant4)
        if self.training:
            out5 = self.out5_conv(x)
        x = x + self.agant3(fuse3)

        x = self.deconv2(x)
        if self.training:
            out4 = self.out4_conv(x)
        x = x + self.agant2(fuse2)

        x = self.deconv3(x)
        if self.training:
            out3 = self.out3_conv(x)
        x = x + self.agant1(fuse1)

        x = self.deconv4(x)
        if self.training:
            out2 = self.out2_conv(x)
        x = x + self.agant0(fuse0)

        x = self.final_conv(x)
        out = self.final_deconv(x)

        if self.training:
            return out, out2, out3, out4, out5
        return out

    def forward(self, rgb, depth):
        fuses = self.encoder(rgb, depth)
        m = self.decoder(*fuses)
        return m
