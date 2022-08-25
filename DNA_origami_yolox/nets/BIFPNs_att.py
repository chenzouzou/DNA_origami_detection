#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn
import math
from torch.nn import functional as F

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[256, 512, 1024], act="silu", depthwise=False, ):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1,
                         act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            # ---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            # ---------------------------------------------------#
            x = self.stems[k](x)
            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            cls_feat = self.cls_convs[k](x)
            # ---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](cls_feat)

            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            reg_feat = self.reg_convs[k](x)
            # ---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](reg_feat)
            # ---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)




class Reuse(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), epsilon=1e-4, depthwise=False,
                 act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.epsilon = epsilon
        self.swish = Swish()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.Conv_P1t2 = BaseConv(256, 512, 3, 2, act=act)
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()
        self.Conv_P2t3 = BaseConv(512, 1024, 3, 2, act=act)
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.Conv_P3U = BaseConv(1024, 512, 1, 1, act=act)
        self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_w2_relu = nn.ReLU()
        self.Conv_P2U = BaseConv(512, 256, 1, 1, act=act)
        self.p1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_w1_relu = nn.ReLU()
        self.P2td = CSPLayer(1024,512,round(3),False,depthwise = depthwise,act = act)
        self.P3out = CSPLayer(2048, 1024, round(3), False, depthwise=depthwise, act=act)
        self.P2out = CSPLayer(1536, 512, round(3), False, depthwise=depthwise, act=act)
        self.P1out = CSPLayer(512, 256, round(3), False, depthwise=depthwise, act=act)







    def forward(self, input):
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        P3_in = input[2]
        P2_in = input[1]
        P1_in = input[0]
        # -------------------------------------------#
        #   80, 80, 256 -> 40, 40, 512
        # -------------------------------------------#
        P1t2 = self.Conv_P1t2(P1_in)

        # -------------------------------------------#
        #   40, 40, 512 + 40, 40, 512 ->40, 40, 512
        # -------------------------------------------#

        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)

        P2_td = self.swish(weight[0] * P2_in + weight[1] * P1t2)
      
        # -------------------------------------------#
        #   40, 40, 512 -> 20, 20, 1024
        # -------------------------------------------#
        P2t3 = self.Conv_P2t3(P2_td)

        # -------------------------------------------#
        #   20, 20, 1024 + 20, 20, 1024 ->20, 20, 1024
        # -------------------------------------------#

        p3_w1 = self.p3_w1_relu(self.p2_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        P3_out = self.swish(weight[0] * P3_in + weight[1] * P2t3)  
     


        # -------------------------------------------#
        #   20, 20, 1024  ->40, 40, 512
        # -------------------------------------------#

        P3_U = self.Conv_P3U(self.upsample(P3_out))

        # -------------------------------------------#
        #   40, 40, 512 + 40, 40, 512 + 40, 40, 512 = 40, 40, 512
        # -------------------------------------------#

        p2_w2 = self.p2_w2_relu(self.p2_w2)
        weight = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
    
        P2_out = self.swish(weight[0] * P2_td + weight[1] * P2_in + weight[2] * P3_U)
        


        # -------------------------------------------#
        #   40, 40, 512  ->80, 80, 256
        # -------------------------------------------#

        P2_U = self.Conv_P2U(self.upsample(P2_out))


        # -------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 ->80, 80, 256
        # -------------------------------------------#
        p1_w1 = self.p1_w1_relu(self.p1_w1)
        weight = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)

       
        P1_out = self.swish(weight[0] * P1_in + weight[1] * P2_U)
         


        return P1_out, P2_out, P3_out








class YOLOPAFPN_new(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"),epsilon=1e-4,depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.epsilon = epsilon
        self.swish = Swish()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.reuse1 = Reuse()
        #self.reuse2 = Reuse()
        #self.reuse3 = Reuse()
        #self.reuse4 = Reuse()
        self.in_features = in_features
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")



    def forward(self, input):
        out_features = self.backbone.forward(input)
        [feat1, feat2, feat3] = [out_features[f] for f in self.in_features]
        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        P1_out,P2_out,P3_out = self.reuse1([feat1, feat2, feat3])
        #P1_out_new2, P2_out_new2, P3_out_new2 = self.reuse2([P1_out,P2_out,P3_out])
        #P1_out_new3, P2_out_new3, P3_out_new3 = self.reuse3([P1_out_new2, P2_out_new2, P3_out_new2])
        #P1_out_new4, P2_out_new4, P3_out_new4 = self.reuse4([P1_out_new3, P2_out_new3, P3_out_new3])


        return (P1_out, P2_out, P3_out)







##------------BiFPN的函数依赖------------------------##

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x




class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False

        self.backbone = YOLOPAFPN_new(depth, width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs = self.backbone.forward(x)
        outputs = self.head.forward(fpn_outs)
        return outputs
