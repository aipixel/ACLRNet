import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from einops.einops import rearrange
from functools import reduce

class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)

class RDB(nn.Module):
    def __init__(self, G0, C, G):  # 64 4 24
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0 + i * G, G))

        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0 + C * G, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x

class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):  # G0=64, C=4, G=24, n_RDB=4
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0 * n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out, buffer_cat


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // 16, 1, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // 16, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



#SKFF
class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1, M=3, r=16, L=32):
        super(SKConv,self).__init__()
        d = max(in_channels//r, L)
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3, stride=stride,padding=1+i, dilation=1+i, groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))

        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)
        self.softmax=nn.Softmax(dim=1)
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z降维
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax
        #the part of selection
        a_b=list(a_b.chunk(self.M, dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度M切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加
        return V

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )

    def __call__(self, x):
        out = self.body(x)
        return out + x

class SPAM(nn.Module):  # 视差注意力机制
    def __init__(self, channels, nb):
        super(SPAM, self).__init__()
        self.bq = nn.Conv2d(nb * channels, channels, 1, 1, 0, groups=nb, bias=True)
        self.bs = nn.Conv2d(nb * channels, channels, 1, 1, 0, groups=nb, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(nb * channels)
        self.bn = nn.BatchNorm2d(nb * channels)
        self.skff = SKConv(nb * channels, nb * channels, stride=1, M=4, r=16, L=32)

    def __call__(self, x_left, x_right, catfea_left, catfea_right):
        b, c0, h0, w0 = x_left.shape
        Q = self.bq(self.skff(self.bn(catfea_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.skff(self.bn(catfea_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),  # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))  # (B*H) * C * Wr

        M_right_to_left = self.softmax(score)  # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))  # (B*H) * Wr * Wl
        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                            ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)
        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)  # B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                             ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)  # B, C0, H0, W0

        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) + x_rightT * V_right_tanh.repeat(1, c0, 1, 1)
        return out_left, out_right



def M_Relax(M, num_pixels):
    _, u, v = M.shape
    M_list = []
    M_list.append(M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i + 1, 0))
        pad_M = pad(M[:, :-1 - i, :])
        M_list.append(pad_M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i + 1))
        pad_M = pad(M[:, i + 1:, :])
        M_list.append(pad_M.unsqueeze(1))
    M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
    return M_relaxed

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = bilinear_grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def manual_pad(x, pady, padx):
    pad = (padx, padx, pady, pady)
    return F.pad(x.clone().detach(), pad, "replicate")

# Ref: https://zenn.dev/pinto0309/scraps/7d4032067d0160
def bilinear_grid_sample(im, grid, align_corners=False):
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = torch.nn.functional.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0, device=im.device), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1, device=im.device), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0, device=im.device), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1, device=im.device), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0, device=im.device), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1, device=im.device), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0, device=im.device), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1, device=im.device), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):

        Q = self.feature_map(queries)  #left
        K = self.feature_map(keys)  #right

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V  #torch.Size([1, 4, 16, 16])
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()



class LoFTREncoderLayer(nn.Module):   #
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attention = LinearAttention()

        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):

        bs = x.size(0)
        query, key, value = x, source, source

        # attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)
        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class NonLocalAttention(nn.Module):
    def __init__(self, d_model, nhead, layer_names, attention):
        super(NonLocalAttention, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        encoder_layer = LoFTREncoderLayer(d_model, nhead, attention)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        if len(feat0.shape) == 4:
            feat0 = rearrange(feat0, 'b c h w -> b (h w) c')
            feat1 = rearrange(feat1, 'b c h w -> b (h w) c')

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1


class NOAM(nn.Module):
    def __init__(self, fmap1, fmap2, att=None):
        super(NOAM, self).__init__()
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.att = att
        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device)
        self.groups = 8

    def forward(self, left_feature, right_feature, extra_offset):
        N, C, H, W = left_feature.shape

        if self.att is not None:
            # 'n (h w) c -> n c h w'
            left_feature, right_feature = self.att(left_feature, right_feature)
            # 'n (h w) c -> n c h w'
            left_feature, right_feature = [x.reshape(N, H, W, C).permute(0, 3, 1, 2) for x in
                                           [left_feature, right_feature]]

        lefts = torch.split(left_feature, left_feature.shape[1] // self.groups, dim=1)
        rights = torch.split(right_feature, right_feature.shape[1] // self.groups, dim=1)

        C = C // self.groups

        search_num = 9
        extra_offset = extra_offset.reshape(N, search_num, 2, H, W).permute(0, 1, 3, 4, 2)  # [N, search_num, 1, 1, 2]

        corrs = []
        for i in range(len(lefts)):
            left_feature, right_feature = lefts[i], rights[i]

            psizey, psizex = 3, 3
            dilatey, dilatex = 1, 1

            ry = psizey // 2 * dilatey
            rx = psizex // 2 * dilatex
            x_grid, y_grid = torch.meshgrid(torch.arange(-rx, rx + 1, dilatex, device=self.fmap1.device),
                                            torch.arange(-ry, ry + 1, dilatey, device=self.fmap1.device))

            offsets = torch.stack((x_grid, y_grid))
            offsets = offsets.reshape(2, -1).permute(1, 0)
            for d in sorted((0, 2, 3)):
                offsets = offsets.unsqueeze(d)
            offsets = offsets.repeat_interleave(N, dim=0)
            offsets = offsets + extra_offset

            coords = self.coords.permute(0, 2, 3, 1)  # [N, H, W, 2]
            coords = torch.unsqueeze(coords, 1) + offsets
            coords = coords.reshape(N, -1, W, 2)  # [N, search_num*H, W, 2]
            right_feature = bilinear_sampler(
                right_feature, coords
            )
            right_feature = right_feature.reshape(N, C, -1, H, W)
            left_feature = left_feature.unsqueeze(2).repeat_interleave(right_feature.shape[2], dim=2)

            corr = torch.mean(left_feature * right_feature, dim=1)
            corrs.append(corr)   #n，9，h，w

        final_corr = torch.cat(corrs, dim=1)
        return final_corr



class Net(nn.Module):
    def __init__(self, upscale_factor, in_nc=3, out_nc=3, ng0=64, ng=64, nbc=8, nb=4):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Conv2d(in_nc, ng0, 3, 1, 1, bias=True)

        self.deep_feature = RDG(G0=ng0, C=nbc, G=ng, n_RDB=nb)
        self.deep_feature1 = RDG(G0=ng0, C=nbc, G=ng, n_RDB=nb)

        self.spam1 = SPAM(channels=ng0, nb=nb)
        self.spam2 = SPAM(channels=ng0, nb=nb)

        self.fusion = nn.Sequential(
            CALayer(ng0 * 2),
            nn.Conv2d(ng0 * 2, ng0, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.fusion1 = nn.Sequential(
            CALayer(ng0 * 2),
            nn.Conv2d(ng0 * 2, ng0, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.reconstruct = RDG(G0=ng0, C=nbc, G=ng, n_RDB=nb)
        self.reconstruct1 = RDG(G0=ng0, C=nbc, G=ng, n_RDB=nb)

        if self.upscale_factor == 1:
            self.upscale = nn.Sequential(
            nn.Conv2d(ng0, ng0, 1, 1, 0, bias=True),
            nn.Conv2d(ng0, out_nc, 3, 1, 1, bias=True))
        else:
            self.upscale = nn.Sequential(
            nn.Conv2d(ng0, ng0 * self.upscale_factor ** 2, 1, 1, 0, bias=True),
            nn.PixelShuffle(self.upscale_factor),
            nn.Conv2d(ng0, out_nc, 3, 1, 1, bias=True))


        self.cov1x1 = nn.Conv2d(72, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_offset = nn.Conv2d(64, 9 * 2, kernel_size=3, stride=1, padding=1)
        self.cross_att_fn = NonLocalAttention(d_model=64, nhead=4, layer_names=["cross", "self"] * 1, attention="linear")


    def forward(self, x_left, x_right):
        x_left_upscale = F.interpolate(x_left, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        buffer_left, catfea_left = self.deep_feature(buffer_left)
        buffer_right, catfea_right = self.deep_feature(buffer_right)
        buffer_leftT, buffer_rightT = self.spam1(buffer_left, buffer_right, catfea_left, catfea_right)

        buffer_leftF = self.fusion(torch.cat([buffer_left, buffer_leftT], dim=1))
        buffer_rightF = self.fusion(torch.cat([buffer_right, buffer_rightT], dim=1))

        offset_left = self.conv_offset(buffer_leftF)
        offset_left = (torch.sigmoid(offset_left) - 0.5) * 2.0
        offset_right = self.conv_offset(buffer_rightF)
        offset_right = (torch.sigmoid(offset_right) - 0.5) * 2.0

        noam_left = NOAM(buffer_leftF, buffer_rightF, att=self.cross_att_fn)
        noam_right = NOAM(buffer_rightF, buffer_leftF, att=self.cross_att_fn)

        buffer_leftF, _ = self.reconstruct(
            self.cov1x1(noam_left(buffer_leftF, buffer_rightF, extra_offset=offset_left)))
        buffer_rightF, _ = self.reconstruct(
            self.cov1x1(noam_right(buffer_rightF, buffer_leftF, extra_offset=offset_right)))

        buffer_left, catfea_left = self.deep_feature1(buffer_leftF)
        buffer_right, catfea_right = self.deep_feature1(buffer_rightF)

        buffer_leftT, buffer_rightT = self.spam2(buffer_left, buffer_right, catfea_left, catfea_right)

        buffer_leftF = self.fusion1(torch.cat([buffer_left, buffer_leftT], dim=1))
        buffer_rightF = self.fusion1(torch.cat([buffer_right, buffer_rightT], dim=1))

        buffer_leftF, _ = self.reconstruct1(buffer_leftF)
        buffer_rightF, _ = self.reconstruct1(buffer_rightF)

        out_left = self.upscale(buffer_leftF) + x_left_upscale
        out_right = self.upscale(buffer_rightF) + x_right_upscale

        return out_left, out_right



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(upscale_factor=4).to(device)
    n, c, h, w = 1, 3, 30, 120
    x = torch.randn(n, c, h, w).to(device)
    print('x.shape:', x.shape)
    model.eval()
    y1, y2, = model(x, x)
    print(y1.shape, y2.shape)

