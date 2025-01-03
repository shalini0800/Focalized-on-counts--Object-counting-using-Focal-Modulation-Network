from functools import partial

import torch
import torch.nn as nn

import math
import time
# import cProfile
# import pstats

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from models.Focal_vit16 import focalnet_base_iso
from models.Focal_vit16_decoder import focalnet_base_iso_decoder
# from models.FocalNet.FocalNet import FocalNet
# from utils.consts import *

from einops import rearrange, reduce, repeat


class MaskedAutoencoderViTNoCT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=96, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(depth)])
        # self.norm = norm_layer(embed_dim)

        f_model = focalnet_base_iso()
        # f_model = FocalNet()
        self.blocks = nn.ModuleList([f_model])
        
        self.norm = norm_layer(embed_dim)

        self.mse_loss = nn.MSELoss()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # # MAE decoder specifics
        # self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])

        # self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        f_model_decoder = focalnet_base_iso_decoder()

        self.decoder_blocks = nn.ModuleList([f_model_decoder])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # --------------------------------------------------------------------------
        self.mask_sampler = torch.distributions.bernoulli.Bernoulli(0.5)
        self.norm_pix_loss = norm_pix_loss

        # self.initialize_weights()

    # def initialize_weights(self):
    #     # initialization
    #     # initialize (and freeze) pos_embed by sin-cos embedding
    #     pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
    #     self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    #     decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
    #     self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    #     # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    #     w = self.patch_embed.proj.weight.data
    #     torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    #     # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    #     torch.nn.init.normal_(self.mask_token, std=.02)

    #     # initialize nn.Linear and nn.LayerNorm
    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         # we use xavier_uniform following official JAX ViT:
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    # def patchify(self, imgs):
    #     """
    #     imgs: (N, 3, H, W)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    #     h = w = imgs.shape[2] // p
    #     x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    #     x = torch.einsum('nchpwq->nhwpqc', x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    #     return x

    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     h = w = int(x.shape[1]**.5)
    #     assert h * w == x.shape[1]
        
    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    #     return imgs

    # def random_masking(self, x, mask_ratio):
    #     """
    #     Perform per-sample random masking by per-sample shuffling.
    #     Per-sample shuffling is done by argsort random noise.
    #     x: [N, L, D], sequence
    #     """
    #     N, L, D = x.shape  # batch, length, dim
    #     len_keep = int(L * (1 - mask_ratio))
        
    #     noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
    #     # sort noise for each sample
    #     ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)

    #     # keep the first subset
    #     ids_keep = ids_shuffle[:, :len_keep]
    #     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    #     # generate the binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask, dim=1, index=ids_restore)

    #     return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        # # embed patches
        # x = self.patch_embed(x)

        # # add pos embed w/o cls token
        # x = x + self.pos_embed

        # # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # apply Transformer blocks
        # print('before Encoder: ', x.shape, torch.sum(torch.isnan(x)))
        for blk in self.blocks:
            x = blk(x)
        # print('After block Encoder: ', x.shape, torch.sum(torch.isnan(x)))
        x = self.norm(x)
        # print('After Norm Encoder: ', x.shape, torch.sum(torch.isnan(x)))

        # return x, mask, ids_restore
        return x

    def forward_decoder(self, x):
    # def forward_decoder(self, x, ids_restore):
        # # embed tokens
        # x = self.decoder_embed(x)

        # # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        # x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = x_ # append cls token

        # # add pos embed
        # x = x + self.decoder_pos_embed

        # # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        # x = self.decoder_norm(x)
        # print('before Decoder: ', x.shape)
        # print(x.device)
        B, HW, C = x.shape
        H = W = int(math.sqrt(HW))
        
        for blk in self.decoder_blocks:
            x = blk(x, H, W)
        # print('After Decoder block: ', x.shape)
        #x = self.decoder_norm(x)
        # print('After Decoder Norm block: ', x.shape)
        # # predictor projection
        # x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # target = self.patchify(imgs)
        # target = imgs
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        # loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        # Move tensors to GPU
        imgs = imgs#.to(device= 'cuda')
        pred = pred#.to(device= imgs.device)
        mask = mask#.to(device= imgs.device)
        
        b, c, h, w = imgs.shape
        h = self.patch_size
        w = self.patch_size
        p1 = int(384/h)
        p2 = int(384/w)

        
        rearranged_pred = rearrange((pred-imgs)**2,'b c (p1 h) (p2 w) -> b c (p1 p2) h w', h=h, w=w)
        masked_loss = torch.einsum('bl,bclhw->bclhw',(1-mask),rearranged_pred)
        loss = masked_loss.sum()/(1-mask).sum()
        #loss = self.mse_loss(pred, imgs)

        # loss_up = nn.MSELoss(reduction='mean')
        # out_loss = loss_up(imgs, pred)
        # out_loss.backward()

        # # For mean loss on all patches
        # N, L = mask.shape
        # mask_s = torch.ones([N, L], device=imgs.device)
        # loss = (loss * mask_s).sum() / mask_s.sum()

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    # def forward(self, imgs, mask_ratio=0.5):
    #     latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    #     pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
    #     loss = self.forward_loss(imgs, pred, mask)
    #     return loss, pred, mask

    def random_masking(self, imgs):
        b, c, h, w = imgs.shape
        h = self.patch_size
        w = self.patch_size
        p1 = int(384/h)
        p2 = int(384/w)
        mod_imgs = rearrange(imgs,'b c (p1 h) (p2 w) -> b c (p1 p2) h w', h=h, w=w)#.to(device= imgs.device)
        # print('mod_imgs: ', mod_imgs.device)
        mask = self.mask_sampler.sample((b, p1*p2)).to(device= imgs.device)
        # print('mask: ', mask.device)
        mod_imgs = torch.einsum('bl,bclhw->bclhw',mask,mod_imgs)#.to(device= imgs.device)
        # print('mod_imgs: ', mod_imgs.device)
        mod_imgs = rearrange(mod_imgs,'b c (p1 p2) h w -> b c (p1 h) (p2 w)', p1=p1)
        return mod_imgs, mask
        

    def forward(self, imgs):
        ## masking
        # imgs = imgs.device
        # print('imgs: ', imgs.device)

        
        # b, c, h, w = imgs.shape
        # h = self.patch_size
        # w = self.patch_size
        # p1 = int(384/h)
        # p2 = int(384/w)
        # mod_imgs = rearrange(imgs,'b c (p1 h) (p2 w) -> b c (p1 p2) h w', h=h, w=w)#.to(device= imgs.device)
        # # print('mod_imgs: ', mod_imgs.device)
        # mask = self.mask_sampler.sample((b, p1*p2)).to(device= imgs.device)
        # # print('mask: ', mask.device)
        # mod_imgs = torch.einsum('bl,bclhw->bclhw',mask,mod_imgs)#.to(device= imgs.device)
        # # print('mod_imgs: ', mod_imgs.device)
        # mod_imgs = rearrange(mod_imgs,'b c (p1 p2) h w -> b c (p1 h) (p2 w)', p1=p1)#.to(device= imgs.device)
        # # print('mod_imgs: ', mod_imgs.device)

        # with cProfile.Profile() as profile:
        start = time.time()
        mod_imgs, mask = self.random_masking(imgs)
        end = time.time()
        print("End- start for Random: ", (end-start))

        start = time.time()
        latent = self.forward_encoder(mod_imgs)#.to(device= imgs.device)
        end = time.time()
        print("End- start for forward encoder: ", (end-start))
        
        start = time.time()
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        end = time.time()
        print("End- start for forward decoder: ", (end-start))

        start = time.time()
        loss = self.forward_loss(imgs, pred, mask)
        end = time.time()
        print("End- start for loss: ", (end-start))

        # results = pstats.Stats(profile)
        # results.sort_stats(pstats.SortKey.TIME)
        # results.print_stats()
        # print(results.print_stats())
        # mod_imgs = mod_imgs#.to(device= imgs.device)
        # print(mod_imgs.device)

        # latent = self.forward_encoder(mod_imgs)#.to(device= imgs.device)
        # # print('latent: ', latent.device)
        # # latent = self.forward_encoder(mod_imgs)
        # # print('latent',torch.sum(torch.isnan(latent)))
        # pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        # # print(pred.device)
        # # print(torch.sum(torch.isnan(pred.view(-1))), torch.sum(torch.isnan(imgs.view(-1))))
        # loss = self.forward_loss(imgs, pred, mask)
        # # print(loss.device)
        # # print(loss.shape, pred.shape, imgs.shape, latent.shape)
        # # print(loss)
        # # print(torch.sum(torch.isnan(pred.view(-1))), torch.sum(torch.isnan(imgs.view(-1))))
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViTNoCT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViTNoCT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViTNoCT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
