import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,\
            padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,\
            bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class QNet_origin(nn.Module):

    def __init__(self, num_channel, conf, rc, layers=[2, 2, 2, 2], preconf=False):
        super(QNet_origin, self).__init__()

        '''
        self.pre_goal = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1,
                padding=1),
                nn.LeakyReLU(inplace=True)
                )
        if conf: 
            self.pre_conf = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1,
                    padding=1),
                nn.LeakyReLU(inplace=True)
                )

        '''
        self.conf = conf
        self.rc = rc

        self.num_channel = num_channel

        block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        assert not preconf, "out-of-date preconf layers"
        if self.conf and preconf:
            self.preconf = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        else:
            self.preconf = None
        self.conv1 = nn.Conv2d(num_channel * 2 + int(self.conf) + int(self.rc), self.inplanes, kernel_size=7, stride=2,
                               padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        # self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        self.l1 = nn.Linear(128 * 128, 1)
        #  self.conv5 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        #  self.conv6 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        #  self.conv7 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        '''
        obs = x[:, :self.num_channel,...]
        goal = x[:, self.num_channel:2*self.num_channel, ...]
        goal = self.pre_goal(goal)

        if self.conf:
            conf = x[:, 2*self.num_channel:, ...]
            conf = self.pre_conf(conf)
            x = torch.cat((obs, goal, conf), dim=1)
        else:
            x = torch.cat((obs, goal), dim=1)
'''
        if self.preconf is not None:
            conf = x[:, -1:, ...]
            normed_conf = self.preconf(conf)
            x = torch.cat((x[:, :-1, ...], normed_conf), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #   x = self.layer5(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        x = self.conv3(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        x = self.conv4(x)
        #  x = self.relu(x)
        #  x = F.interpolate(x, scale_factor=2, mode="bilinear",
        #          align_corners=True)
        #  x = self.conv5(x)
        #  x = self.relu(x)
        #  x = F.interpolate(x, scale_factor=2, mode="bilinear",
        #          align_corners=True)

        #  x = self.conv6(x)
        #  x = self.relu(x)
        #  x = F.interpolate(x, scale_factor=2, mode='bilinear',
        #          align_corners=True)

        #  x = self.conv7(x)
        q = self.l1(torch.flatten(x[0][0]))

        return [x, q]

class State_predictor_origin(nn.Module):

    def __init__(self, num_channel, conf, rc, layers=[2, 2, 2, 2], preconf=False):
        super(State_predictor_origin, self).__init__()
        self.conf = conf
        self.rc = rc

        self.num_channel = num_channel

        block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        assert not preconf, "out-of-date preconf layers"
        if self.conf and preconf:
            self.preconf = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        else:
            self.preconf = None
        self.conv1 = nn.Conv2d(num_channel * 2 + int(self.conf) + int(self.rc) + 2, self.inplanes, kernel_size=7,
                               stride=2,
                               padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        #  self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(128, 83, kernel_size=1, stride=1)
        #  self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        #  self.conv5 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        #  self.conv6 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        #  self.conv7 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, action_batch):
        if self.preconf is not None:
            conf = x[:, -2, ...]
            normed_conf = self.preconf(conf)
            x = torch.cat((x[:, :-1, ...], normed_conf), dim=1)
        x = torch.cat((x, action_batch), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #   x = self.layer5(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        x = self.conv3(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)

        return x

class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

        self.relu = GELU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.l1(inputs)
        output = self.relu(output)
        output = self.l2(output)
        return self.layer_norm(output + residual)

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(d_model, d_k * n_heads)
        self.WK = nn.Linear(d_model, d_k * n_heads)
        self.WV = nn.Linear(d_model, d_v * n_heads)

        self.linear = nn.Linear(n_heads * d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = ScaledDotProductAttention(d_k=self.d_k)(Q=q_s, K=k_s, V=v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.n_heads * self.d_v)  # concat happens here
        output = self.linear(context)
        return self.layer_norm(output + Q)

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model,d_k=d_k, d_v=d_v, n_heads=n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff)

    def forward(self, enc_inputs):
        enc_outputs = self.enc_self_attn(Q=enc_inputs, K=enc_inputs, V=enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs

class Encoder(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, n_layers):
        super(Encoder, self).__init__()

        self.pos_emb = PositionalEncoding(d_model=d_model, dropout=0)
        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        enc_outputs = self.pos_emb(x)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)
        return enc_outputs

class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model,d_k=d_k,
            d_v=d_v, n_heads=n_heads)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model,d_k=d_k,
            d_v=d_v, n_heads=n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)

    def forward(self, dec_inputs, enc_outputs):
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs

class Decoder(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, n_layers):
        super(Decoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model=d_model, dropout=0)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.pos_emb(dec_inputs)
        for layer in self.layers:
            dec_outputs = layer(dec_inputs=dec_outputs, enc_outputs=enc_outputs)
        return dec_outputs

class MapAttention(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, n_layers):
        super(MapAttention, self).__init__()
        self.encoder = Encoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers)
        self.decoder = Decoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        return dec_outputs

class QNet(nn.Module):

    def __init__(self, num_channel, conf, rc, layers=[2, 2, 2, 2], preconf=False):
        super(QNet, self).__init__()
        
        '''
        self.pre_goal = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1,
                padding=1),
                nn.LeakyReLU(inplace=True)
                )
        if conf: 
            self.pre_conf = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1,
                    padding=1),
                nn.LeakyReLU(inplace=True)
                )
        
        '''
        self.conf = conf
        self.rc = rc
        
        self.num_channel = num_channel

        block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        assert not preconf, "out-of-date preconf layers"
        if self.conf and preconf:
            self.preconf = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        else:
            self.preconf = None
        self.conv1 = nn.Conv2d(num_channel*2 + int(self.conf) + int(self.rc), self.inplanes, kernel_size=7, stride=2,
                padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        #self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        self.conv_relation_1 = nn.Conv2d(512, 512, kernel_size=2, stride=2)
        self.map_attention = MapAttention(d_model=1024, d_ff=1024, d_k=1024, d_v=1024, n_heads=2, n_layers=2)

        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        #self.l1 = nn.Linear(128 * 128, 1)
        #  self.conv5 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        #  self.conv6 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        #  self.conv7 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, relationship_bank):

        '''
        obs = x[:, :self.num_channel,...]
        goal = x[:, self.num_channel:2*self.num_channel, ...]
        goal = self.pre_goal(goal)

        if self.conf:
            conf = x[:, 2*self.num_channel:, ...]
            conf = self.pre_conf(conf)
            x = torch.cat((obs, goal, conf), dim=1)
        else:
            x = torch.cat((obs, goal), dim=1)
'''
        if self.preconf is not None:
            conf = x[:, -1:, ...]
            normed_conf = self.preconf(conf)
            x = torch.cat((x[:, :-1, ...], normed_conf), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)      #(1, 512, 32, 32)
     #   x = self.layer5(x)
        relation = self.conv_relation_1(relationship_bank.permute(0,3,1,2))  #(1, 512, 32, 32)
        x_ = rearrange(x, 'b c h w -> b c (h w)')
        relation = rearrange(relation, 'b c h w -> b c (h w)')
        enc_outputs = self.map_attention(x_, relation)

        x_ = enc_outputs.reshape(x.size(0), 512, 32, 32)

        x = x + x_

        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True)
        x = self.conv3(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True)
        x = self.conv4(x)
      #  x = self.relu(x)
      #  x = F.interpolate(x, scale_factor=2, mode="bilinear",
      #          align_corners=True)
      #  x = self.conv5(x)
      #  x = self.relu(x)
      #  x = F.interpolate(x, scale_factor=2, mode="bilinear",
      #          align_corners=True)

      #  x = self.conv6(x)
      #  x = self.relu(x)
      #  x = F.interpolate(x, scale_factor=2, mode='bilinear',
      #          align_corners=True)

      #  x = self.conv7(x)
        #q = self.l1(torch.flatten(x[0][0]))
        #return [x, q]
        return x

class State_predictor(nn.Module):

    def __init__(self, num_channel, conf, rc, layers=[2, 2, 2, 2], preconf=False):
        super(State_predictor, self).__init__()
        self.conf = conf
        self.rc = rc

        self.num_channel = num_channel

        block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        assert not preconf, "out-of-date preconf layers"
        if self.conf and preconf:
            self.preconf = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        else:
            self.preconf = None
        self.conv1 = nn.Conv2d(num_channel * 2 + int(self.conf) + int(self.rc) + 2, self.inplanes, kernel_size=7, stride=2,
                               padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        #  self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        self.conv_relation_1 = nn.Conv2d(512, 512, kernel_size=2, stride=2)
        self.map_attention = MapAttention(d_model=1024, d_ff=1024, d_k=1024, d_v=1024, n_heads=2, n_layers=2)

        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(128, 83, kernel_size=1, stride=1)
        #  self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        #  self.conv5 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        #  self.conv6 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        #  self.conv7 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, action_batch, relationship_bank):
        if self.preconf is not None:
            conf = x[:, -2, ...]
            normed_conf = self.preconf(conf)
            x = torch.cat((x[:, :-1, ...], normed_conf), dim=1)
        x = torch.cat((x, action_batch), dim = 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #   x = self.layer5(x)
        relation = self.conv_relation_1(relationship_bank.permute(0,3,1,2)) #(1, 512, 32, 32)
        x_ = rearrange(x, 'b c h w -> b c (h w)')
        relation = rearrange(relation, 'b c h w -> b c (h w)')
        enc_outputs = self.map_attention(x_, relation)

        x_ = enc_outputs.reshape(x.size(0), 512, 32, 32)

        x = x + x_

        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        x = self.conv3(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)

        return x

class QNet512(nn.Module):

    def __init__(self, num_channel, layers=[2, 2, 2, 2, 2]):
        super(QNet512, self).__init__()

        block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        
        self.conv1 = nn.Conv2d(num_channel, self.inplanes, kernel_size=7, stride=2,
                padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
       # self.layer5 = self._make_layer(block, 512, layers[4], stride=2)

        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
       # self.conv5 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
       # self.conv6 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        #print(x.shape)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
       # x = self.layer5(x)
     #   print(x.shape)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True)
        x = self.conv3(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True)
        x = self.conv4(x)
      #  x = self.relu(x)
      #  x = F.interpolate(x, scale_factor=2, mode="bilinear",
      #          align_corners=True)
      #  x = self.conv5(x)
      #  x = self.relu(x)
      #  x = F.interpolate(x, scale_factor=2, mode="bilinear",
      #          align_corners=True)

      #  x = self.conv6(x)

        return x
class Q_discrete(nn.Module):

    def __init__(self, num_channel, conf, layers=[2, 2, 2, 2], preconf=False):
        super(Q_discrete, self).__init__()
        
        '''
        self.pre_goal = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1,
                padding=1),
                nn.LeakyReLU(inplace=True)
                )
        if conf: 
            self.pre_conf = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1,
                    padding=1),
                nn.LeakyReLU(inplace=True)
                )
        
        '''
        self.conf = conf
        
        self.num_channel = num_channel

        block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        if self.conf and preconf:
            self.preconf = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        else:
            self.preconf = None
        self.conv1 = nn.Conv2d(num_channel*2 + int(self.conf), self.inplanes, kernel_size=7, stride=2,
                padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
      #  self.layer5 = self._make_layer(block, 512, layers[4], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 8)

      #  self.conv2 = nn.Conv2d(512, 128, kernel_size=3, stride=2)
      #  self.conv3 = nn.Conv2d(128, 32, kernel_size=3, stride=2)
      #  self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=2)
      #  self.conv5 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
      #  self.conv6 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
      #  self.conv7 = nn.Conv2d(32, 1, kernel_size=1, stride=1) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        '''
        obs = x[:, :self.num_channel,...]
        goal = x[:, self.num_channel:2*self.num_channel, ...]
        goal = self.pre_goal(goal)

        if self.conf:
            conf = x[:, 2*self.num_channel:, ...]
            conf = self.pre_conf(conf)
            x = torch.cat((obs, goal, conf), dim=1)
        else:
            x = torch.cat((obs, goal), dim=1)
'''
        if self.preconf is not None:
            conf = x[:, -1:, ...]
            normed_conf = self.preconf(conf)
            x = torch.cat((x[:, :-1, ...], normed_conf), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
     #   x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
if __name__ == "__main__":
    Q_net = Q_discrete(41, True)
    inp = torch.rand(3, 83, 128, 128)
    output = Q_net(inp)
    print(output.shape)

