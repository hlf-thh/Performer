import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class addlinear(nn.Module):

    def __init__(self, drop_p=0., drop_path_ratio=0., in_dim=64, out_dim=512):
        super(addlinear, self).__init__()
        self.fc = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=drop_p)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, x):
        x1, x2 = x
        out = x1 + self.drop_path(self.drop(self.act(self.fc(x2))))

        return out


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 expan_radio=1,
                 is_last=False,
                 basic_dim=64,
                 basic_head=1):
        super(Block, self).__init__()
        """"------------------------------------ours----------------------------------------"""
        self.is_last = is_last
        self.fc1 = nn.Linear(in_features=dim, out_features=basic_dim*expan_radio)
        self.act1 = act_layer()
        self.norm1 = norm_layer(basic_dim*expan_radio, eps=1e-6)
        self.attn = Attention(dim=basic_dim*expan_radio, num_heads=basic_head*expan_radio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.norm2 = norm_layer(basic_dim*expan_radio, eps=1e-6)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.addlin = addlinear(drop_path_ratio=drop_path_ratio, in_dim=basic_dim*expan_radio, out_dim=dim, drop_p=drop_ratio)

    def forward(self, x):
        ret = self.act1(self.fc1(x))
        ret = self.norm2(ret + self.drop_path(self.attn(self.norm1(ret))))

        # 定义成一个函数,方面模块化编程！
        out = self.addlin((x, ret))

        if self.is_last:
            # 如果是当前层最后一块输出两个
            return x, ret
        else:
            return out



class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=512, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Performer(nn.Module):

    def __init__(self, res_block, res_block_num, vit_block_num,
                 img_size=224, patch_size=16, embed_dim=512,drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, num_classes=1000, alpha=0.5, beta=0.5):
        super(Performer, self).__init__()
        self.attn_drop_ratio = attn_drop_ratio
        self.drop_path_ratio = drop_path_ratio
        self.drop_ratio = drop_ratio
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.beta = beta
        """判断resnet用的 忽略即可"""
        self.in_channel = 64
        self.groups = 1
        self.width_per_group = 64
        """
        vit的预处理部分
        """
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        """
        resnet的预处理部分
        """
        self.conv1 = nn.Conv2d(3, embed_dim//8, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim//8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        """
        resnet的block部分
        """
        self.layer1 = self._make_layer(res_block, embed_dim // 8, res_block_num[0])
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.layer2 = self._make_layer(res_block, embed_dim // 4, res_block_num[1], stride=2)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.layer3 = self._make_layer(res_block, embed_dim // 2, res_block_num[2], stride=2)
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.layer4 = self._make_layer(res_block, embed_dim, res_block_num[3], stride=2)
        self.avgpool4 = nn.AdaptiveAvgPool2d((1, 1))

        """vit的block部分"""
        self.block1 = self._make_block(vit_block_num[0], j=1)
        self.simgod1 = nn.Sigmoid()
        self.addlin1 = addlinear(drop_p=drop_ratio, drop_path_ratio=drop_path_ratio, in_dim=embed_dim // 8,
                                 out_dim=embed_dim)
        self.block2 = self._make_block(vit_block_num[1], j=2)
        self.simgod2 = nn.Sigmoid()
        self.addlin2 = addlinear(drop_p=drop_ratio, drop_path_ratio=drop_path_ratio, in_dim=embed_dim // 4,
                                 out_dim=embed_dim)
        self.block3 = self._make_block(vit_block_num[2], j=3)
        self.simgod3 = nn.Sigmoid()
        self.addlin3 = addlinear(drop_p=drop_ratio, drop_path_ratio=drop_path_ratio, in_dim=embed_dim // 2,
                                 out_dim=embed_dim)
        self.block4 = self._make_block(vit_block_num[3], j=4)
        self.addlin4 = addlinear(drop_p=drop_ratio, drop_path_ratio=drop_path_ratio, in_dim=embed_dim,
                                 out_dim=embed_dim)


        """最后一层全连接层"""
        self.fc = nn.Linear(in_features=embed_dim, out_features=num_classes, bias=False)
        """
        resnet的block部分
        """
        """预处理"""
        self.apply(_init_vit_weights)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def _make_block(self, block_num, j):
        blocks = []
        for i in range(block_num-1):
            blocks.append(Block(dim=self.embed_dim, basic_dim=self.embed_dim//8, expan_radio=2**(j-1),
                                   drop_ratio=self.drop_ratio, attn_drop_ratio=self.attn_drop_ratio,
                                   drop_path_ratio=self.drop_path_ratio, is_last=False))

        blocks.append(Block(dim=self.embed_dim, basic_dim=self.embed_dim//8, expan_radio=2**(j-1),
                                   drop_ratio=self.drop_ratio, attn_drop_ratio=self.attn_drop_ratio,
                                   drop_path_ratio=self.drop_path_ratio, is_last=True))

        return nn.Sequential(*blocks)

    """vit 预处理"""
    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)
        return x

    def forward(self, x):
        """预处理"""
        vit_out = self.forward_features(x)
        res_out = self.relu(self.bn(self.conv1(x)))
        res_out = self.maxpool(res_out)


        """ ret是 cnn到transformer """
        """ tret是 transformer到cnn """


        """layer1"""
        res_out = self.layer1(res_out)
        ret1 = self.avgpool1(res_out)
        ret1 = torch.flatten(ret1, 1)

        vit1, vit_out = self.block1(vit_out)
        tret1 = vit_out[:, 0]
        """cnn"""
        vit_out[:, 0] = self.alpha * ret1 + self.beta * vit_out[:, 0]
        vit_out = self.addlin1((vit1, vit_out))
        """transformer"""
        tret1 = tret1.reshape(tret1.shape[0], tret1.shape[1], 1, 1)
        tret1 = self.simgod1(tret1)
        res_out = res_out*tret1

        """layer2"""
        res_out = self.layer2(res_out)
        ret2 = self.avgpool2(res_out)
        ret2= torch.flatten(ret2, 1)

        vit2, vit_out = self.block2(vit_out)
        tret2 = vit_out[:, 0]
        """cnn"""
        vit_out[:, 0] = self.alpha * ret2 + self.beta * vit_out[:, 0]
        vit_out = self.addlin2((vit2, vit_out))
        """transformer"""
        tret2 = tret2.reshape(tret2.shape[0], tret2.shape[1], 1, 1)
        tret2 = self.simgod2(tret2)
        res_out = res_out*tret2

        """layer3"""
        res_out = self.layer3(res_out)
        ret3 = self.avgpool3(res_out)
        ret3 = torch.flatten(ret3, 1)

        vit3, vit_out = self.block3(vit_out)
        tret3 = vit_out[:, 0]
        """cnn"""
        vit_out[:, 0] = self.alpha * ret3 + self.beta * vit_out[:, 0]
        vit_out = self.addlin3((vit3, vit_out))
        """transformer"""
        tret3 = tret3.reshape(tret3.shape[0], tret3.shape[1], 1, 1)
        tret3 = self.simgod3(tret3)
        res_out = res_out*tret3

        """layer4"""
        res_out = self.layer4(res_out)
        ret4 = self.avgpool4(res_out)
        ret4 = torch.flatten(ret4, 1)

        vit4, vit_out = self.block4(vit_out)
        """cnn"""
        vit_out[:, 0] = self.alpha * ret4 + self.beta * vit_out[:, 0]
        vit_out = self.addlin4((vit4, vit_out))

        """结束"""
        out = vit_out[:, 0]
        out = self.fc(out)
        return out

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def Performer_small(num_classes=1000, alpha=0.5, beta=0.5, p=0.):
    return Performer(res_block=BasicBlock, res_block_num=[2, 2, 6, 2], vit_block_num=[2, 2, 6, 2], num_classes=num_classes,
                  alpha=alpha, beta=beta,  embed_dim=512, drop_ratio=p,
                 attn_drop_ratio=p, drop_path_ratio=p)



