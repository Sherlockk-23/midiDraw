#===================================================================================================================
#
# X Trasformer Module
#
# Partial x-transformers code With useful modifications
#
# Version 1.0
#
# Original source code courtesy of lucidrains
# https://github.com/lucidrains/x-transformers
#
# Original source code retrieved on 10/10/2023
#
# Project Los Angeles
# Tegridy Code 2023

#===================================================================================================================

# Critical dependencies
#
# !pip install torch
# !pip install einops

#===================================================================================================================

from functools import partial
from typing import Optional, Tuple

import os
os.environ['USE_FLASH_ATTENTION'] = '1'

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

# Flash attention
from torch.nn.attention import SDPBackend, sdpa_kernel
torch.backends.cuda.enable_flash_sdp(True)

from collections import namedtuple
from functools import wraps
from packaging import version
from dataclasses import dataclass

from einops import rearrange, repeat

# constants

EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

@dataclass
class Intermediates:
    qk_similarities: Optional[Tensor] = None
    pre_softmax_attn: Optional[Tensor] = None
    post_softmax_attn: Optional[Tensor] = None
    cached_kv: Optional[Tuple[Tensor, Tensor]] = None

    def to_tuple(self):
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def compact(arr):
    return [*filter(exists, arr)]

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# functions for creating causal mask
# need a special one for onnx cpu (no support for .triu)

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def onnx_create_causal_mask(i, j, device):
    r = torch.arange(i, device = device)
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    causal_mask = F.pad(causal_mask, (j - i, 0), value = False)
    return causal_mask

# main class

class Attend(nn.Module):
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        heads = None,
        talking_heads = False,
        sparse_topk = None,
        scale = None,
        qk_norm = False,
        flash = False,
        add_zero_kv = False,
        onnxable = False
    ):
        super().__init__()
        self.scale = scale
        self.qk_norm = qk_norm

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        self.attn_fn = partial(F.softmax, dtype = torch.float32) if not qk_norm else F.softmax

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads

        assert not (flash and talking_heads), 'talking heads not compatible with flash attention'

        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
            self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)

        # sparse topk

        assert not (flash and sparse_topk), 'sparse topk not compatible with flash attention'
        self.sparse_topk = sparse_topk

        # add a key / value token composed of zeros
        # in case this helps controlling outliers, proposed by https://www.evanmiller.org/attention-is-off-by-one.html

        self.add_zero_kv = add_zero_kv

        # flash attention

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        major, minor = device_properties.major, device_properties.minor

        if (major, minor) == (8, 0):
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        elif (major, minor) == (9, 0):
            print_once('H100 GPU detected, using flash attention')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def flash_attn(
        self,
        q, k, v,
        mask = None,
        attn_bias = None
    ):
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention

        if self.qk_norm:
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        # in the case of kv caching with one token (q_len == 1), just turn off causal masking
        # in speculative decoding, this may go up to 5-6, so right aligned causal mask will be needed there

        if q_len == 1 and causal:
            causal = False

        # expand key padding mask

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim = -1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, heads, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        # Legacy code...
        # with torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=True):
        # with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):

        # PyTorch 2.3-2.4 SDPA backend code...
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
        # with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):

        # New PyTorch 2.5 SDPA backend code:
        # with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out, Intermediates()

    def forward(
        self,
        q, k, v,
        mask = None,
        attn_bias = None,
        prev_attn = None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        causal = self.causal

        # handle kv cached decoding

        if n == 1 and causal:
            causal = False

        # handle grouped multi-query attention

        if kv_heads == 1:
            k, v = map(lambda t: rearrange(t, 'b 1 n d -> b n d'), (k, v))
        elif kv_heads < heads:
            k, v = map(lambda t: repeat(t, 'b kvh n d -> b (r kvh) n d', r = heads // kv_heads), (k, v))

        # handle zero kv, as means for allowing network to attend to nothing

        if self.add_zero_kv:
            k, v = map(lambda t: F.pad(t, (0, 0, 1, 0), value = 0.), (k, v))

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value = 0.)

        if self.flash:
            assert not exists(prev_attn), 'residual attention not compatible with flash attention'
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        if exists(prev_attn):
            dots = dots + prev_attn

        qk_similarities = dots.clone()

        if self.talking_heads:
            dots = self.pre_softmax_talking_heads(dots)

        if exists(attn_bias):
            dots = dots + attn_bias

        i, j, dtype = *dots.shape[-2:], dots.dtype

        mask_value = -torch.finfo(dots.dtype).max

        if exists(self.sparse_topk) and self.sparse_topk < j:
            top_values, _ = dots.topk(self.sparse_topk, dim = -1)
            sparse_topk_mask = dots < top_values[..., -1:]
            mask = (mask & sparse_topk_mask) if exists(mask) else sparse_topk_mask

        if exists(mask):
            dots = dots.masked_fill(~mask, mask_value)

        if causal:
            causal_mask = self.create_causal_mask(i, j, device = device)
            dots = dots.masked_fill(causal_mask, mask_value)

        pre_softmax_attn = dots.clone()

        attn = self.attn_fn(dots, dim = -1)
        attn = attn.type(dtype)

        post_softmax_attn = attn.clone()

        attn = self.attn_dropout(attn)

        if self.talking_heads:
            attn = self.post_softmax_talking_heads(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        intermediates = Intermediates(
            qk_similarities = qk_similarities,
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        return out, intermediates

#===================================================================================================================

from math import ceil, log
from typing import Optional, Union, Tuple, Callable

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, pack, unpack

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def identity(t, *args, **kwargs):
    return t

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else (t,) * length

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# for variable lengthed prefixes

def align_right(t, lens, pad_id = 0):
    batch, seq_len, device, dtype = *t.shape, t.device, t.dtype

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device = device, dtype = torch.long)

    t = F.pad(t, (max_pad_len, 0), value = 0)
    offset = max_pad_len - pad_lens

    aligned = t[batch_arange, prompt_len_arange + offset[..., None]]
    return aligned

# nucleus

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, frac_num_tokens = 0.1, k = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# top_a

def top_a(logits, min_p_pow = 2.0, min_p_ratio = 0.02):
    probs = F.softmax(logits, dim = -1)
    max_probs = torch.amax(probs, dim = -1, keepdim = True)
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float('-inf'), logits)

# contrastive decoding function

def contrastive_decode_fn(
    expert_logits,
    amateur_logits,
    alpha = 0.1,
    beta = 0.5
):
    """
    Appendix A Algorithm 2
    https://arxiv.org/abs/2309.09117
    """

    cutoff = log(alpha) + expert_logits.amax(dim = -1, keepdim = True)
    diffs = (1 + beta) * expert_logits - beta * amateur_logits
    contrastive_decode_logits = diffs.masked_fill(expert_logits < cutoff, -torch.finfo(expert_logits.dtype).max)
    return contrastive_decode_logits

# autoregressive wrapper class

class AutoregressiveWrapper(Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0,
        mask_prob = 0.,
        add_attn_z_loss = False,
        return_cache=False,
        penalty_reward_tokens=None
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

        # whether to add router z-loss
        self.add_attn_z_loss = add_attn_z_loss
        self.return_cache = return_cache
        
        
        self.penalty_reward_tokens = penalty_reward_tokens

    # [MY]
    @torch.inference_mode()
    @eval_decorator
    def generateWithPic(
        self,  
        prompts,
        img, # get something like (50, None, 1)
        seq_len,
        eos_token = None,
        temperature = 1.,
        prompt_lens: Optional[Tensor] = None,
        filter_logits_fn: Callable = top_k,
        restrict_to_max_seq_len = True,
        amateur_model: Optional[Union[Module, Tuple[Module]]] = None,
        filter_kwargs: dict = dict(),
        contrastive_decode_kwargs: Union[dict, Tuple[dict]] = dict(
            beta = 0.5,
            alpha = 0.1
        ),
        cache_kv = True,
        verbose=True,
        return_prime=False,
        **kwargs
    ):
        print("prompts shape:", prompts.shape)
        print("img shape:", img.shape)
        max_seq_len, device = self.max_seq_len, prompts.device

        prompts, ps = pack([prompts], '* n')

        b, t = prompts.shape
        
        # [MY] rescale img to (50, *, 1)
        assert(img.shape[0] == 50), "img should be (50, *)"
        # img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # img_H = 50
        # img_scale = img_H / img.shape[0]
        # # img = F.interpolate(img, size = (img_H, int(img.shape[2] * img_scale)), mode = 'bilinear', align_corners = False)
        # # use a numpy function to resize, not F.interpolate
        # img = img.squeeze(0).squeeze(0).numpy()
        # # img = cv2.resize(img, (int(img.shape[1] * img_scale), img_H), interpolation=cv2.INTER_LINEAR)
        img_H = img.shape[0]
        img_W = img.shape[1]
        img = img.reshape(img_H, img_W)

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id = self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        if verbose:
          print("Generating sequence of max length:", seq_len)

        # kv caches

        cache = None

        # if doing contrastive decoding, turn off filter automatically

        if exists(amateur_model):
            amateur_model = cast_tuple(amateur_model)
            contrastive_decode_kwargs = cast_tuple(contrastive_decode_kwargs)

            assert len(amateur_model) == len(contrastive_decode_kwargs)

            amateur_caches = [None] * len(amateur_model)
            filter_logits_fn = identity

            for i, module in enumerate(amateur_model):
                if isinstance(module, AutoregressiveWrapper):
                    amateur_model[i] = module.net

                module.eval()

        # sampling up to seq_len
        
        # [MY] init pointers 
        # [0,256], [256, 2304], [2304, 2304+128]
        default_middle_y = 2304 + 65 
        # if prompts[0,-4] > 2304+20 and prompts[0,-4] < 2304+80:
        #     middle_y = prompts[0,-4]
        middle_y = 0
        cnt_cal_middle = 0
        for i in range(2, len(prompts[0])-1, 3):
            if prompts[0, i] > 2304 and prompts[0, i] < 2304 + 128:
                middle_y += prompts[0, i]
                cnt_cal_middle += 1
        if cnt_cal_middle > 0:
            middle_y = int(middle_y // cnt_cal_middle)
        middle_y = max(middle_y, default_middle_y-15)
        middle_y = min(middle_y, default_middle_y+15)
        # print("middle_y:", middle_y, "cnt_cal_middle:", cnt_cal_middle)


        ave_dur = 0 
        dur_cnt = 0
        for i in range(1, len(prompts[0])-1, 3):
            if prompts[0, i] > 256 and prompts[0, i] < 2304:
                ave_dur += ((prompts[0, i]-256)//8)*16
                dur_cnt += 1
        if dur_cnt > 0:
            ave_dur = int(ave_dur // dur_cnt)

        # print("ave_dur:", ave_dur)



        img_px, img_py = find_nxt_pixel(img, 0, img_H)
        added_x = img_px
        ntime = 0
        ndur = 0
        npit = 108
        handle_none =False

        num_midis_same_time = 0

        for sl in range(seq_len):

            if restrict_to_max_seq_len:
                x = out[:, -max_seq_len:]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            logits, new_cache = self.net(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                **kwargs
            )

            if cache_kv and self.net.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]

            # handle contrastive decoding, Li et al.
            # https://arxiv.org/abs/2210.15097

            if exists(amateur_model):
                for i, (amateur, amateur_cache, amateur_contrastive_decode_kwargs) in enumerate(zip(amateur_model, amateur_caches, contrastive_decode_kwargs)):
                    amateur_logits, next_amateur_cache = amateur(
                        x,
                        return_intermediates = True,
                        cache = amateur_cache,
                        seq_start_pos = seq_start_pos,
                        **kwargs
                    )

                    amateur_logits = amateur_logits[:, -1]

                    assert amateur_logits.shape == logits.shape, 'logits dimension are not the same between amateur and expert model'
                    logits = contrastive_decode_fn(logits, amateur_logits, **amateur_contrastive_decode_kwargs)

                    if cache_kv and amateur.can_cache_kv:
                        amateur_caches[i] = next_amateur_cache
                        
                        
            # [MY] here to reweight logists
            print("="*70)
            print("sl:", sl)
            print("img_px:", img_px, "img_py:", img_py, "ntime:", ntime, "ndur:", ndur, "npit:", npit)
            gen = False
            if img_px != None:
                now_x = int(ntime/1000*4)
                if img_px > now_x:
                    if img_px - now_x > 10:
                        # print("in branch 1")
                        pass
                    elif sl%3==0:
                        # print("in branch 2")
                        logits[0, :5] *= -10 # force a jump
                elif img_px < now_x:
                    # print("in branch 3")
                    while img_px < now_x:
                        img_px, img_py = find_nxt_pixel(img, img_px, img_py)
                        if img_px == None:
                            break
                if img_px == now_x:
                    if sl%3==0:
                        # print("in branch 4")
                        if num_midis_same_time <=5:
                            logits[0, :3] *=10 # force unmove
                        else:
                            logits[0, :3] *= -10 # force a jump
                    elif sl%3==1:
                        # print("in branch 5")
                        pass
                        # [TODO]
                    else:
                        if img_py<=npit:
                            # print("in branch 6")
                            if torch.rand(1) < 0.95:
                                tar_y = img_py - (img_H//2) + middle_y
                                logits[:, tar_y-1:tar_y+3] *= 10
                                new_img_px, new_img_py, _ = get_next_pos(img, img_px, img_py-1)
                                new_img_px, new_img_py = find_nxt_pixel(img, new_img_px, new_img_py)
                                if new_img_px != None:
                                    added_x = new_img_px - img_px
                                img_px, img_py = new_img_px, new_img_py
                        else:
                            # print("in branch 7")
                            new_img_px, new_img_py = find_nxt_pixel(img, img_px, img_py)
                            while new_img_px == now_x and new_img_py > npit:
                                new_img_px, new_img_py = find_nxt_pixel(img, new_img_px, new_img_py)
                                if new_img_px == None:
                                    break
                            if new_img_px != None:
                                added_x = new_img_px - img_px
            elif handle_none == False:
                if sl%3 == 0:
                    logits[0,:10]*=-10 # force a jump
                    handle_none = True
            # filter by top_k, top_p (nucleus), top_a, or custom

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)
            
            # [MY] here utilize output
            if True:
                if sl%3==0:
                    ntime += sample[0] * 16
                    ntime = int(ntime)
                    if sample[0]!=0:
                        npit = 108
                        num_midis_same_time = 0
                    else:
                        num_midis_same_time += 1
                    # [TBD]
                elif sl%3==1:
                    ndur = ((sample[0]-256) // 8) * 16
                    ndur = (int(ndur))
                else:
                    if sample[0]< 2304 or sample[0] > 2304 + 128:
                        continue
                    draw_y = sample[0] -middle_y + img_H//2
                    npit = int(draw_y)
                    img[draw_y-2:draw_y+3, int(ntime/1000*4):int((ntime+ndur)/1000*4)+1] = 0
                    # print("drawing in img at:", draw_y, int(ntime/1000*4), int((ntime+ndur)/1000*4))
            

            out = torch.cat((out, sample), dim=-1)

            if verbose:
              if sl % 32 == 0:
                print(sl, '/', seq_len)

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)

                if is_eos_tokens.any(dim = -1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    out = out.masked_fill(mask, self.pad_value)

                    if verbose: 
                      print('Model called the end of sequence at:', sl, '/', seq_len)

                    break

        if return_prime:
          return out[:, :]
        
        else:
          return out[:, t:]
      
      
    @torch.inference_mode()
    @eval_decorator
    def generateWithPicRPM(
        self,  
        prompts,
        img, # get something like (50, None, 1)
        seq_len,
        eos_token = None,
        temperature = 1.,
        prompt_lens: Optional[Tensor] = None,
        filter_logits_fn: Callable = top_k,
        restrict_to_max_seq_len = True,
        amateur_model: Optional[Union[Module, Tuple[Module]]] = None,
        filter_kwargs: dict = dict(),
        contrastive_decode_kwargs: Union[dict, Tuple[dict]] = dict(
            beta = 0.5,
            alpha = 0.1
        ),
        cache_kv = True,
        verbose=True,
        return_prime=False,
        **kwargs
    ):
        print("prompts shape:", prompts.shape)
        print("img shape:", img.shape)
        max_seq_len, device = self.max_seq_len, prompts.device

        prompts, ps = pack([prompts], '* n')

        b, t = prompts.shape
        
        # [MY] rescale img to (50, *, 1)
        assert(img.shape[0] == 50), "img should be (50, *)"
        # img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # img_H = 50
        # img_scale = img_H / img.shape[0]
        # # img = F.interpolate(img, size = (img_H, int(img.shape[2] * img_scale)), mode = 'bilinear', align_corners = False)
        # # use a numpy function to resize, not F.interpolate
        # img = img.squeeze(0).squeeze(0).numpy()
        # # img = cv2.resize(img, (int(img.shape[1] * img_scale), img_H), interpolation=cv2.INTER_LINEAR)
        img_H = img.shape[0]
        img_W = img.shape[1]
        img = img.reshape(img_H, img_W)

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id = self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        if verbose:
          print("Generating sequence of max length:", seq_len)

        # kv caches

        cache = None

        # if doing contrastive decoding, turn off filter automatically

        if exists(amateur_model):
            amateur_model = cast_tuple(amateur_model)
            contrastive_decode_kwargs = cast_tuple(contrastive_decode_kwargs)

            assert len(amateur_model) == len(contrastive_decode_kwargs)

            amateur_caches = [None] * len(amateur_model)
            filter_logits_fn = identity

            for i, module in enumerate(amateur_model):
                if isinstance(module, AutoregressiveWrapper):
                    amateur_model[i] = module.net

                module.eval()

        # sampling up to seq_len
        
        # [MY] init pointers 
        # [0,256], [256, 2304], [2304, 2304+128]
        default_middle_y = 2304 + 65 
        # if prompts[0,-4] > 2304+20 and prompts[0,-4] < 2304+80:
        #     middle_y = prompts[0,-4]
        middle_y = 0
        cnt_cal_middle = 0
        for i in range(2, len(prompts[0])-1, 3):
            if prompts[0, i] > 2304 and prompts[0, i] < 2304 + 128:
                middle_y += prompts[0, i]
                cnt_cal_middle += 1
        if cnt_cal_middle > 0:
            middle_y = int(middle_y // cnt_cal_middle)
        middle_y = max(middle_y, default_middle_y-15)
        middle_y = min(middle_y, default_middle_y+15)
        # print("middle_y:", middle_y, "cnt_cal_middle:", cnt_cal_middle)


        ave_dur = 0 
        dur_cnt = 0
        for i in range(1, len(prompts[0])-1, 3):
            if prompts[0, i] > 256 and prompts[0, i] < 2304:
                ave_dur += ((prompts[0, i]-256)//8)*16
                dur_cnt += 1
        if dur_cnt > 0:
            ave_dur = int(ave_dur // dur_cnt)

        # print("ave_dur:", ave_dur)



        img_px, img_py = find_nxt_pixel(img, 0, img_H)
        added_x = img_px
        ntime = 0
        ndur = 0
        npit = 108
        handle_none =False

        num_midis_same_time = 0

        for sl in range(seq_len):

            if restrict_to_max_seq_len:
                x = out[:, -max_seq_len:]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]
                        
                        
            penalty_reward_mask = torch.zeros((1, x.shape[1], 19463+1), dtype=torch.float16, device='cuda')
            print("reward_penalty_mask shape:", penalty_reward_mask.shape)

            
                        
                        
            # [MY] here to reweight logists
            print("="*70)
            print("sl:", sl)
            print("img_px:", img_px, "img_py:", img_py, "ntime:", ntime, "ndur:", ndur, "npit:", npit)
            gen = False
            if img_px != None:
                now_x = int(ntime/1000*4)
                if img_px > now_x:
                    if img_px - now_x > 10:
                        print("in branch 1")
                        pass
                    elif sl%3==0:
                        print("in branch 2")
                        # logits[0, :5] *= -10 # force a jump
                        penalty_reward_mask[0, :, :5] = -2 # force a jump
                elif img_px < now_x:
                    print("in branch 3")
                    while img_px < now_x:
                        img_px, img_py = find_nxt_pixel(img, img_px, img_py)
                        if img_px == None:
                            break
                if img_px == now_x:
                    if sl%3==0:
                        print("in branch 4")
                        if num_midis_same_time <=5:
                            # logits[0, :3] *=10 # force unmove
                            penalty_reward_mask[0,:, :3] = 2
                        else:
                            # logits[0, :3] *= -10 # force a jump
                            penalty_reward_mask[0,:, :3] = -2
                    elif sl%3==1:
                        print("in branch 5")
                        pass
                        # [TODO]
                    else:
                        if img_py<=npit:
                            print("in branch 6")
                            if torch.rand(1) < 0.95:
                                tar_y = img_py - (img_H//2) + middle_y
                                # logits[:, tar_y-1:tar_y+3] *= 10
                                penalty_reward_mask[0, :, tar_y-1:tar_y+3] = 2
                                new_img_px, new_img_py, _ = get_next_pos(img, img_px, img_py-1)
                                new_img_px, new_img_py = find_nxt_pixel(img, new_img_px, new_img_py)
                                if new_img_px != None:
                                    added_x = new_img_px - img_px
                                img_px, img_py = new_img_px, new_img_py
                        else:
                            print("in branch 7")
                            new_img_px, new_img_py = find_nxt_pixel(img, img_px, img_py)
                            while new_img_px == now_x and new_img_py > npit:
                                new_img_px, new_img_py = find_nxt_pixel(img, new_img_px, new_img_py)
                                if new_img_px == None:
                                    break
                            if new_img_px != None:
                                added_x = new_img_px - img_px
            elif handle_none == False:
                if sl%3 == 0:
                    # logits[0,:10]*=-10 # force a jump
                    penalty_reward_mask[0, :, :10] = -2
                    handle_none = True
            # filter by top_k, top_p (nucleus), top_a, or custom
            
            # penalty_reward_mask = torch.zeros_like(penalty_reward_mask)
            print("sum in penalty_reward_mask:", penalty_reward_mask.sum())
            # print("argmax in penalty_reward_mask:", penalty_reward_mask.argmax(dim=-1).item(), "with value:", penalty_reward_mask[0, -1, penalty_reward_mask.argmax(dim=-1).item()].item())
            # print("argmin in penalty_reward_mask:", penalty_reward_mask.argmin(dim=-1).item(), "with value:", penalty_reward_mask[0, -1, penalty_reward_mask.argmin(dim=-1).item()].item())
            
            logits, new_cache = self.net(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                penalty_reward_mask=penalty_reward_mask, # [TODO]
                **kwargs
            )

            if cache_kv and self.net.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]
            
            print("logits shape:", logits.shape)
            print("logits:", logits[0, :20])
            print("arg max in logits:", logits.argmax(dim=-1).item(), "with value:", logits[0, logits.argmax(dim=-1).item()].item())
            print("arg min in logits:", logits.argmin(dim=-1).item(), "with value:", logits[0, logits.argmin(dim=-1).item()].item())
            

            # handle contrastive decoding, Li et al.
            # https://arxiv.org/abs/2210.15097

            if exists(amateur_model):
                for i, (amateur, amateur_cache, amateur_contrastive_decode_kwargs) in enumerate(zip(amateur_model, amateur_caches, contrastive_decode_kwargs)):
                    amateur_logits, next_amateur_cache = amateur(
                        x,
                        return_intermediates = True,
                        cache = amateur_cache,
                        seq_start_pos = seq_start_pos,
                        **kwargs
                    )

                    amateur_logits = amateur_logits[:, -1]

                    assert amateur_logits.shape == logits.shape, 'logits dimension are not the same between amateur and expert model'
                    logits = contrastive_decode_fn(logits, amateur_logits, **amateur_contrastive_decode_kwargs)

                    if cache_kv and amateur.can_cache_kv:
                        amateur_caches[i] = next_amateur_cache

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)
            
            print("sampled token:", sample[0].item())
            
            # [MY] here utilize output
            if True:
                if sl%3==0:
                    ntime += sample[0] * 16
                    ntime = int(ntime)
                    if sample[0]!=0:
                        npit = 108
                        num_midis_same_time = 0
                    else:
                        num_midis_same_time += 1
                    # [TBD]
                elif sl%3==1:
                    ndur = ((sample[0]-256) // 8) * 16
                    ndur = (int(ndur))
                else:
                    if sample[0]< 2304 or sample[0] > 2304 + 128:
                        continue
                    draw_y = sample[0] -middle_y + img_H//2
                    npit = int(draw_y)
                    img[draw_y-2:draw_y+3, int(ntime/1000*4):int((ntime+ndur)/1000*4)+1] = 0
                    # print("drawing in img at:", draw_y, int(ntime/1000*4), int((ntime+ndur)/1000*4))
            

            out = torch.cat((out, sample), dim=-1)

            if verbose:
              if sl % 32 == 0:
                print(sl, '/', seq_len)

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)

                if is_eos_tokens.any(dim = -1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    out = out.masked_fill(mask, self.pad_value)

                    if verbose: 
                      print('Model called the end of sequence at:', sl, '/', seq_len)

                    break

        if return_prime:
          return out[:, :]
        
        else:
          return out[:, t:]


    @torch.inference_mode()
    @eval_decorator
    def generate(
        self,
        prompts,
        seq_len,
        eos_token = None,
        temperature = 1.,
        prompt_lens: Optional[Tensor] = None,
        filter_logits_fn: Callable = top_k,
        restrict_to_max_seq_len = True,
        amateur_model: Optional[Union[Module, Tuple[Module]]] = None,
        filter_kwargs: dict = dict(),
        contrastive_decode_kwargs: Union[dict, Tuple[dict]] = dict(
            beta = 0.5,
            alpha = 0.1
        ),
        cache_kv = True,
        verbose=True,
        return_prime=False,
        **kwargs
    ):
        max_seq_len, device = self.max_seq_len, prompts.device

        prompts, ps = pack([prompts], '* n')

        b, t = prompts.shape

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id = self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        if verbose:
          print("Generating sequence of max length:", seq_len)

        # kv caches

        cache = None

        # if doing contrastive decoding, turn off filter automatically

        if exists(amateur_model):
            amateur_model = cast_tuple(amateur_model)
            contrastive_decode_kwargs = cast_tuple(contrastive_decode_kwargs)

            assert len(amateur_model) == len(contrastive_decode_kwargs)

            amateur_caches = [None] * len(amateur_model)
            filter_logits_fn = identity

            for i, module in enumerate(amateur_model):
                if isinstance(module, AutoregressiveWrapper):
                    amateur_model[i] = module.net

                module.eval()

        # sampling up to seq_len

        for sl in range(seq_len):

            if restrict_to_max_seq_len:
                x = out[:, -max_seq_len:]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            # print("before into net, x shape:", x.shape)
            logits, new_cache = self.net(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                **kwargs
            )
            # print("after into net, logits shape:", logits.shape)

            if cache_kv and self.net.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]

            # handle contrastive decoding, Li et al.
            # https://arxiv.org/abs/2210.15097

            if exists(amateur_model):
                for i, (amateur, amateur_cache, amateur_contrastive_decode_kwargs) in enumerate(zip(amateur_model, amateur_caches, contrastive_decode_kwargs)):
                    amateur_logits, next_amateur_cache = amateur(
                        x,
                        return_intermediates = True,
                        cache = amateur_cache,
                        seq_start_pos = seq_start_pos,
                        **kwargs
                    )

                    amateur_logits = amateur_logits[:, -1]

                    assert amateur_logits.shape == logits.shape, 'logits dimension are not the same between amateur and expert model'
                    logits = contrastive_decode_fn(logits, amateur_logits, **amateur_contrastive_decode_kwargs)

                    if cache_kv and amateur.can_cache_kv:
                        amateur_caches[i] = next_amateur_cache

            # filter by top_k, top_p (nucleus), top_a, or custom

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if verbose:
              if sl % 32 == 0:
                print(sl, '/', seq_len)

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)

                if is_eos_tokens.any(dim = -1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    out = out.masked_fill(mask, self.pad_value)

                    if verbose: 
                      print('Model called the end of sequence at:', sl, '/', seq_len)

                    break

        if return_prime:
          return out[:, :]
        
        else:
          return out[:, t:]

        # out, = unpack(out, ps, '* n')

        # return out

    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc

    def forward(self, x, penalty_reward_mask = None, **kwargs):
        seq, ignore_index, add_attn_z_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss

        inp, target = x[:, :-1], x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        if self.mask_prob > 0.:
            rand = torch.randn(inp.shape, device = x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim = -1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_kv_mask = mask)
            
        # print("inp shape:", inp.shape) # inp shape: torch.Size([2, 8192])

        logits, cache = self.net(
            inp,
            return_intermediates = True,
            return_attn_z_loss = add_attn_z_loss,
            penalty_reward_mask = penalty_reward_mask,
            **kwargs
        )

        # print("logits shape:", logits.shape) # logits shape: torch.Size([2, 8192, 19464])

        acc = self.compute_accuracy(logits, target)
        
        # here to add something

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )
        
        # print("loss :", loss) # ([])
        
        if penalty_reward_mask is not None:
            # 假设penalty_reward_mask是一个二维张量，形状为(batch_size, vocab_size)
            # logits的形状为(batch_size, seq_len, vocab_size)
            # 根据penalty_reward_mask的值，对logits进行加权求和，计算损失
            # 这里只是一个简单的示例，具体的计算方式可以根据实际需求调整
            # print("penalty_reward_mask shape:", penalty_reward_mask.shape) # (4, 19464)
            # print("logits shape:", logits.shape) # (4, 8192, 19464)
            # print("argmax penalty_reward_mask:", torch.argmax(torch.abs(penalty_reward_mask), dim=-1), "with value ", 
            #       torch.max(torch.abs(penalty_reward_mask), dim=-1).values)
            # print("argmax in logits:", torch.argmax(logits[:,-1,:], dim=-1),
            #         "with value ", torch.max(logits[:,-1,:], dim=-1).values)
            penalty_reward_loss = torch.mean(
                (logits[:,:,:] * (-penalty_reward_mask)))* 10  # 假设权重为10，可以根据实际需求调整
            # print("penalty_reward_loss:", penalty_reward_loss)
            loss = loss + penalty_reward_loss 


        
        if add_attn_z_loss:
            loss = loss + cache.attn_z_loss
        
        if self.return_cache:
            return loss, acc, cache
            
        elif penalty_reward_mask is not None:
            return loss, penalty_reward_loss, acc
        
        else:
            return loss, acc

#===============================================================================

import math
from random import random

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from functools import partial, wraps
from inspect import isfunction
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Callable, Optional

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# constants

DEFAULT_DIM_HEAD = 64

@dataclass
class LayerIntermediates:
    hiddens: Optional[List[Tensor]] = None
    attn_intermediates: Optional[List[Intermediates]] = None
    layer_hiddens: Optional[List[Tensor]] = None
    attn_z_loss: Optional[Tensor] = None
    mems: Optional[Tensor] = None

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

def divisible_by(num, den):
    return (num % den) == 0

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, *args, **kwargs):
        return self.val

class not_equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x != self.val

class equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x == self.val

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor helpers

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# auxiliary loss helpers

def calc_z_loss(
    pre_softmax_attns: List[Tensor],
    mask = None,
    weight = 1.
):
    # the same loss applied to the mixture of experts router logits in https://arxiv.org/abs/2202.08906
    # in the paper, in a tiny footnote, they mention using it on attention logits with stabilizing effects
    # also used in PaLM as one of the measures

    lse = 0.

    for attn in pre_softmax_attns:
        lse = lse + attn.logsumexp(dim = -1)

    loss = torch.square(lse)
    loss = reduce(loss, 'b h n -> b n', 'sum')

    if not exists(mask):
        return loss.mean() * weight

    loss = loss[mask].sum() / mask.sum().clamp(min = 1e-5)
    return loss * weight

# init helpers

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# activations

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

# embedding

class TokenEmbedding(nn.Module):
    def __init__(self, dim, num_tokens, l2norm_embed = False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x)
        return l2norm(token_emb) if self.l2norm_embed else token_emb

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale

class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else None,
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else None,
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = 0)
        self.register_buffer('bias', bias, persistent = False)

        return self.bias

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward(self, seq_len):
        device = self.inv_freq.device
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not exists(self.scale):
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs, scale = 1):
    # print("apply_rotary_pos_emb, t shape:", t.shape, "freqs shape:", freqs.shape)
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim = -1)

# norms

class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        scale_fn = lambda t: t * self.value

        if not isinstance(out, tuple):
            return scale_fn(out)

        return (scale_fn(out[0]), *out[1:])

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1) * (dim ** -0.5))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True)
        return x / norm.clamp(min = self.eps) * self.g

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.g

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale

# residual and residual gates

class Residual(nn.Module):
    def __init__(self, dim, scale_residual = False, scale_residual_constant = 1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual

class GRUGating(nn.Module):
    def __init__(self, dim, scale_residual = False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)

# token shifting

def shift(t, amount, mask = None):
    if amount == 0:
        return t
    else:
        amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return pad_at_dim(t, (amount, -amount), dim = - 2, value = 0.)

class ShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

# feedforward

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        mult_bias = False
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate) * self.mult_bias

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        glu_mult_bias = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias = glu_mult_bias)
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias = not no_bias),
                activation
            )

        self.ff = Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)

# attention. it is all we need

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = DEFAULT_DIM_HEAD,
        heads = 8,
        causal = False,
        flash = False,
        talking_heads = False,
        head_scale = False,
        sparse_topk = None,
        num_mem_kv = 0,
        dropout = 0.,
        on_attn = False,
        gate_value_heads = False,
        gate_values = False,
        zero_init_output = False,
        max_attend_past = None,
        qk_norm = False,
        qk_norm_groups = 1,
        qk_norm_scale = 10,
        qk_norm_dim_scale = False,
        one_kv_head = False,
        kv_heads = None,
        shared_kv = False,
        value_dim_head = None,
        tensor_product = False,      # https://arxiv.org/abs/2208.06061
        add_zero_kv = False,         # same as add_zero_attn in pytorch
        rotary_embed_values = False,
        onnxable = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal
        self.max_attend_past = max_attend_past

        assert not (exists(kv_heads) and one_kv_head), 'either attn_one_kv_head is set to True (in which case kv_heads is set to 1), or attn_kv_heads is set, but not both'

        value_dim_head = default(value_dim_head, dim_head)
        kv_heads = default(kv_heads, heads)

        kv_heads = 1 if one_kv_head else kv_heads
        assert divisible_by(heads, kv_heads)

        self.kv_heads = kv_heads

        q_dim = dim_head * heads
        k_dim = dim_head * kv_heads
        v_dim = value_dim_head * kv_heads
        out_dim = value_dim_head * heads

        self.to_q = nn.Linear(dim, q_dim, bias = False)
        self.to_k = nn.Linear(dim, k_dim, bias = False)

        # shared key / values, for further memory savings during inference
        assert not (shared_kv and value_dim_head != dim_head), 'key and value head dimensions must be equal for shared key / values'
        self.to_v = nn.Linear(dim, v_dim, bias = False) if not shared_kv else None

        # relations projection from tp-attention
        self.to_r = nn.Linear(dim, v_dim, bias = False) if tensor_product else None

        # add GLU gating for aggregated values, from alphafold2
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, out_dim)
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 10)

        # add per head gating of the output values, from 'Attend to nothing' paper
        self.to_v_head_gate = None
        if gate_value_heads:
            self.to_v_head_gate = nn.Linear(dim, heads)
            nn.init.constant_(self.to_v_head_gate.weight, 0)
            nn.init.constant_(self.to_v_head_gate.bias, 10)

        # cosine sim attention
        self.qk_norm = qk_norm
        self.qk_norm_groups = qk_norm_groups
        self.qk_norm_scale = qk_norm_scale

        # whether to use the rmsnorm (equivalent to cosine sim attention when scale is equal to 1) - https://arxiv.org/abs/2302.05442
        self.qk_norm_dim_scale = qk_norm_dim_scale

        self.qk_norm_q_scale = self.qk_norm_k_scale = 1
        if qk_norm and qk_norm_dim_scale:
            self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))

        assert (not qk_norm) or divisible_by(dim_head, qk_norm_groups), 'dimension per attention head must be divisible by the qk norm groups'
        assert not (qk_norm and (dim_head // qk_norm_groups) <= 2), 'the group dimension may be too small (2 was too small in my tests, but 4 still works, surprisingly)'

        # attend class - includes core attention algorithm + talking heads

        self.attend = Attend(
            heads = heads,
            causal = causal,
            talking_heads = talking_heads,
            dropout = dropout,
            sparse_topk = sparse_topk,
            qk_norm = qk_norm,
            scale = qk_norm_scale if qk_norm else self.scale,
            add_zero_kv = add_zero_kv,
            flash = flash,
            onnxable = onnxable
        )

        # head scaling
        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(out_dim, dim * 2, bias = False), nn.GLU()) if on_attn else nn.Linear(out_dim, dim, bias = False)

        # whether to rotate positions into values, for absolute positions in addition to relative
        self.rotary_embed_values = rotary_embed_values

        # init output projection 0
        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        rel_pos = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None,
        return_intermediates = False,
        cache: Optional[Intermediates] = None,
    ):
        b, n, _, h, kv_h, head_scale, device, has_context = *x.shape, self.heads, self.kv_heads, self.head_scale, x.device, exists(context)
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input
        r_input = x

        if exists(mem):
            k_input, mem_packed_shape = pack([mem, k_input], 'b * d')
            v_input, _ = pack([mem, v_input], 'b * d')

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input) if exists(self.to_v) else k
        r = self.to_r(r_input) if exists(self.to_r) else None

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        k, v, r = map(lambda t: maybe(rearrange)(t, 'b n (h d) -> b h n d', h = kv_h), (k, v, r))

        if exists(cache) and not has_context:
            ck, cv = cache.cached_kv

            if exists(mem):
                mk, k = unpack(k, mem_packed_shape, 'b h * d')
                mv, v = unpack(v, mem_packed_shape, 'b h * d')

            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

            if exists(mem):
                k = torch.cat((mk, k), dim = -2)
                v = torch.cat((mv, v), dim = -2)

        if return_intermediates:
            mem_len = mem.shape[-2] if exists(mem) else 0
            cached_kv = (k[..., mem_len:, :], v[..., mem_len:, :])

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups = self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))
            scale = self.qk_norm_scale

            q = q * self.qk_norm_q_scale
            k = k * self.qk_norm_k_scale

        if exists(rotary_pos_emb) and not has_context:
            freqs, xpos_scale = rotary_pos_emb
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if exists(xpos_scale) else (1., 1.)

            q = apply_rotary_pos_emb(q, freqs, q_xpos_scale)
            k = apply_rotary_pos_emb(k, freqs, k_xpos_scale)

            if self.rotary_embed_values:
                v = apply_rotary_pos_emb(v, freqs, k_xpos_scale)

        input_mask = context_mask

        if not exists(input_mask) and not has_context:
            input_mask = mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), (self.mem_k, self.mem_v))

            if self.qk_norm:
                mem_k = l2norm(mem_k)
                mem_k = mem_k * self.qk_norm_k_scale

            k = torch.cat((mem_k, k), dim = -2)
            v = torch.cat((mem_v, v), dim = -2)

            if exists(input_mask):
                input_mask = pad_at_dim(input_mask, (self.num_mem_kv, 0), dim = -1, value = True)

        i, j = map(lambda t: t.shape[-2], (q, k))

        # determine masking

        mask_value = max_neg_value(q)
        masks = []
        final_attn_mask = None

        if exists(input_mask):
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        if exists(attn_mask):
            assert 2 <= attn_mask.ndim <= 4, 'attention mask must have greater than 2 dimensions but less than or equal to 4'
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            masks.append(~attn_mask)

        if exists(self.max_attend_past):
            range_q = torch.arange(j - i, j, device = device)
            range_k = torch.arange(j, device = device)
            dist = rearrange(range_q, 'i -> 1 1 i 1') - rearrange(range_k, 'j -> 1 1 1 j')
            max_attend_past_mask = dist > self.max_attend_past
            masks.append(max_attend_past_mask)

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        # prepare relative positional bias, if needed

        attn_bias = None
        if exists(rel_pos):
            attn_bias = rel_pos(i, j)

        # attention is all we need

        out, intermediates = self.attend(
            q, k, v,
            mask = final_attn_mask,
            attn_bias = attn_bias,
            prev_attn = prev_attn
        )

        # https://arxiv.org/abs/2208.06061 proposes to add a residual for better gradients

        if exists(r):
            out = out * r + out

        # normformer scaling of heads

        if head_scale:
            out = out * self.head_scale_params

        # per head gating, from https://arxiv.org/abs/2306.12929

        if exists(self.to_v_head_gate):
            head_gate = self.to_v_head_gate(x)
            out = out * rearrange(head_gate, 'b n h -> b h n 1').sigmoid()

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # alphafold2 styled gating of the values

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * gates.sigmoid()

        # combine the heads

        out = self.to_out(out)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        if not return_intermediates:
            return out

        intermediates.cached_kv = cached_kv

        return out, intermediates

class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rmsnorm = False,
        use_simple_rmsnorm = False,
        alibi_pos_bias = False,
        alibi_num_heads = None,
        rel_pos_bias = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        dynamic_pos_bias = False,
        dynamic_pos_bias_log_distance = False,
        dynamic_pos_bias_mlp_depth = 2,
        dynamic_pos_bias_norm = False,
        rotary_pos_emb = False,
        rotary_emb_dim = None,
        rotary_xpos = False,
        rotary_interpolation_factor = 1.,
        rotary_xpos_scale_base = 512,
        rotary_base_rescale_factor = 1.,
        custom_layers = None,
        sandwich_coef = None,
        par_ratio = None,
        weight_tie_layers = False,   # Albert - https://arxiv.org/abs/1909.11942
        layers_execute_order = None, # generalizes weight tying, can do arbitrary layer execution orders
        residual_attn = False,
        cross_residual_attn = False,
        macaron = False,
        pre_norm = True,
        pre_norm_has_final_norm = True,
        gate_residual = False,
        scale_residual = False,
        scale_residual_constant = 1.,
        shift_tokens = 0,
        sandwich_norm = False,
        resi_dual = False,
        resi_dual_scale = 1.,
        zero_init_branch_output = False,
        layer_dropout = 0.,
        cross_attn_tokens_dropout = 0.,
        **kwargs
    ):
        super().__init__()
        rotary_pos_emb = rotary_pos_emb or rotary_xpos

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.has_pos_emb = rel_pos_bias or rotary_pos_emb

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)

        assert not (rotary_xpos and not causal), 'rotary xpos is not compatible with bidirectional attention'
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim, use_xpos = rotary_xpos, scale_base = rotary_xpos_scale_base, interpolation_factor = rotary_interpolation_factor, base_rescale_factor = rotary_base_rescale_factor) if rotary_pos_emb else None

        assert not (alibi_pos_bias and rel_pos_bias), 'you can only choose Alibi positional bias or T5 relative positional bias, not both'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        # relative positional bias

        flash_attn = attn_kwargs.get('flash', False)
        assert (int(rel_pos_bias) + int(dynamic_pos_bias) + int(alibi_pos_bias)) <= 1, 'you can only choose up to one of t5, alibi, or dynamic positional bias'

        self.rel_pos = None
        if rel_pos_bias:
            assert not flash_attn, 'flash attention not compatible with t5 relative positional bias'
            self.rel_pos = RelativePositionBias(scale = dim_head ** 0.5, causal = causal, heads = heads, num_buckets = rel_pos_num_buckets, max_distance = rel_pos_max_distance)
        elif dynamic_pos_bias:
            assert not flash_attn, 'flash attention not compatible with dynamic positional bias'
            self.rel_pos = DynamicPositionBias(dim = dim // 4, heads = heads, log_distance = dynamic_pos_bias_log_distance, depth = dynamic_pos_bias_mlp_depth, norm = dynamic_pos_bias_norm)
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            self.rel_pos = AlibiPositionalBias(heads = alibi_num_heads, total_heads = heads)

        assert (int(sandwich_norm) + int(resi_dual)) <= 1, 'either sandwich norm or resiDual is selected, but not both'
        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'

        if resi_dual:
            pre_norm = False

        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.resi_dual = resi_dual
        assert 0 < resi_dual_scale <= 1., 'resiDual prenorm residual must be scaled by a factor greater than 0 and less than or equal to 1.'
        self.resi_dual_scale = resi_dual_scale

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        assert not (flash_attn and (residual_attn or cross_residual_attn)), 'flash attention is not compatible with residual attention'

        self.cross_attend = cross_attend

        assert (int(use_scalenorm) + int(use_rmsnorm) + int(use_simple_rmsnorm)) <= 1, 'you can only use either scalenorm, rmsnorm, or simple rmsnorm'

        if use_scalenorm:
            norm_class = ScaleNorm
        elif use_rmsnorm:
            norm_class = RMSNorm
        elif use_simple_rmsnorm:
            norm_class = SimpleRMSNorm
        else:
            norm_class = nn.LayerNorm

        norm_fn = partial(norm_class, dim)

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output':  True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output':  True}

        # setup weight tying, which is a special case of `layer_execute_order`

        assert not (weight_tie_layers and any([*map(exists, (custom_layers, par_ratio, sandwich_coef))]))

        if weight_tie_layers:
            assert not exists(layers_execute_order)
            layers_execute_order = tuple(range(len(default_block))) * depth
            depth = 1

        # calculate layer block order

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn  = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.layers_execute_order = default(layers_execute_order, tuple(range(len(layer_types))))

        assert all([i < len(self.layer_types) for i in self.layers_execute_order])

        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # stochastic depth

        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types))

        # structured dropout for cross attending

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # whether it has post norm

        self.final_norm = norm_fn() if pre_norm or resi_dual else nn.Identity()

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == 'a':
                layer = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            residual_fn = GRUGating if gate_residual else Residual
            residual = residual_fn(dim, scale_residual = scale_residual, scale_residual_constant = scale_residual_constant)

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = nn.ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.layers.append(nn.ModuleList([
                norms,
                layer,
                residual
            ]))

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        self_attn_kv_mask = None,
        mems = None,
        seq_start_pos: Optional[Tensor] = None,
        cache: Optional[LayerIntermediates] = None,
        cache_age = 1,
        return_hiddens = False
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'

        # initialize accums

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        # handle left padded sequences

        if exists(seq_start_pos):
            seq_arange = torch.arange(x.shape[-2], device = x.device, dtype = torch.long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        rotary_pos_emb = None

        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems)))
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length)

        # assume cached key / values

        attn_cache = []

        if exists(cache):
            assert not self.training and self.causal and not any([*map(exists, (mask, attn_mask))])

            if cache_age > 0:
                x = x[:, -cache_age:] # for spec decoding, may be greater than 1

            attn_cache = cache.attn_intermediates

        iter_attn_cache = iter(attn_cache)

        # outer residual - for resiDual paper

        outer_residual = x * self.resi_dual_scale

        # get layers to be executed

        layer_variables = (
            self.layer_types,
            self.layers,
            self.layer_dropouts
        )

        layer_variables = tuple(tuple(layer_variable[i] for i in self.layers_execute_order) for layer_variable in layer_variables)

        # go through the attention and feedforward layers

        for ind, (layer_type, (norm, block, residual_fn), layer_dropout) in enumerate(zip(*layer_variables)):
            is_last = ind == (len(self.layers) - 1)

            if self.training and layer_dropout > 0. and random() < layer_dropout:
                continue

            if layer_type == 'a':
                if return_hiddens:
                    hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            if layer_type == 'c':
                if self.training and self.cross_attn_tokens_dropout > 0.:
                    context, context_mask = dropout_seq(context, context_mask, self.cross_attn_tokens_dropout)

            inner_residual = x

            if return_hiddens:
                layer_hiddens.append(x)

            pre_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_norm):
                x = pre_norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask = mask, context_mask = self_attn_kv_mask, attn_mask = attn_mask, rel_pos = self.rel_pos, rotary_pos_emb = rotary_pos_emb, prev_attn = prev_attn, cache = next(iter_attn_cache, None), mem = layer_mem, return_intermediates = True)
            elif layer_type == 'c':
                out, inter = block(x, context = context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn, cache = next(iter_attn_cache, None), return_intermediates = True)
            elif layer_type == 'f':
                out = block(x)

            if self.resi_dual:
                outer_residual = outer_residual + out * self.resi_dual_scale

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual)

            if layer_type in ('a', 'c') and return_hiddens:
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.resi_dual:
            x = x + self.final_norm(outer_residual)
        else:
            x = self.final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens = hiddens,
            attn_intermediates = intermediates,
            layer_hiddens = layer_hiddens
        )

        return x, intermediates

class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal = False, **kwargs)

class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal = True, **kwargs)

class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend = True, only_cross = True, **kwargs)

class ViTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        attn_layers,
        channels = 3,
        num_classes = None,
        post_emb_norm = False,
        num_register_tokens = 0,
        emb_dropout = 0.
    ):
        super().__init__()
        assert isinstance(attn_layers, Encoder), 'attention layers must be an Encoder'
        assert divisible_by(image_size, patch_size), 'image dimensions must be divisible by the patch size'
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        has_register_tokens = num_register_tokens > 0
        self.has_register_tokens = has_register_tokens

        if has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.patch_to_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.post_emb_norm = nn.LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers

        self.mlp_head = nn.Linear(dim, num_classes) if exists(num_classes) else nn.Identity()

    def forward(
        self,
        img,
        return_embeddings = False
    ):
        b, p = img.shape[0], self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        n = x.shape[1]

        x = x + self.pos_embedding[:, :n]

        x = self.post_emb_norm(x)
        x = self.dropout(x)

        if self.has_register_tokens:
            r = repeat(self.register_tokens, 'n d -> b n d', b = b)
            x, ps = pack((x, r), 'b * d')

        x = self.attn_layers(x)

        if self.has_register_tokens:
            x, _ = unpack(x, ps, 'b * d')

        if not exists(self.mlp_head) or return_embeddings:
            return x

        x = x.mean(dim = -2)
        return self.mlp_head(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class TransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        emb_dim=None,
        max_mem_len=0,
        shift_mem_down=0,
        emb_dropout=0.,
        post_emb_norm=False,
        num_memory_tokens=None,
        memory_tokens_interspersed_every=None,
        tie_embedding=False,
        logits_dim=None,
        use_abs_pos_emb=True,
        scaled_sinu_pos_emb=False,
        l2norm_embed=False,
        emb_frac_gradient=1.,  # GLM-130B and Cogview successfully used this, set at 0.1
        attn_z_loss_weight=1e-4,
        logits_dropout=False,
        penalty_reward_tokens=None  # New parameter for penalty and reward tokens
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed
        self.token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed=l2norm_embed)

        if not (use_abs_pos_emb and not attn_layers.has_pos_emb):
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed=l2norm_embed)

        self.emb_frac_gradient = emb_frac_gradient
        self.post_emb_norm = nn.LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        self.init_()

        logits_dim = default(logits_dim, num_tokens)
        self.to_logits = nn.Linear(dim, logits_dim) if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()

        # memory tokens (like [cls]) from Memory Transformers paper

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # whether can do cached kv decoding

        self.can_cache_kv = self.num_memory_tokens == 0

        # New attributes for penalty and reward tokens
        self.penalty_reward_tokens = penalty_reward_tokens
        if penalty_reward_tokens is not None:
            self.penalty_reward_net = nn.Sequential(
                nn.Linear(dim + logits_dim, 2*dim),  # 假设dim是模型的维度，可以根据需要调整
                nn.ReLU(),
                nn.Linear(2*dim, dim*2),
                nn.ReLU(),
                nn.Linear(dim*2, dim)  # 输出维度与模型的维度相同
            )

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.token_emb.emb.weight, std=1e-5)
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            return

        nn.init.kaiming_normal_(self.token_emb.emb.weight)

    def forward(
        self,
        x,
        return_embeddings=False,
        return_logits_and_embeddings=False,
        return_intermediates=False,
        mask=None,
        return_mems=False,
        return_attn=False,
        mems=None,
        pos=None,
        prepend_embeds=None,
        sum_embeds=None,
        return_attn_z_loss=False,
        attn_z_loss_weight=1e-4,
        seq_start_pos=None,
        cache: Optional[LayerIntermediates] = None,
        penalty_reward_mask=None,  # New parameter for penalty and reward mask
        **kwargs
    ):
        b, n, device, num_mems, has_memory_tokens, emb_frac_gradient = *x.shape, x.device, self.num_memory_tokens, self.num_memory_tokens > 0, self.emb_frac_gradient
        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos=pos, seq_start_pos=seq_start_pos) if not external_pos_emb else pos
        x = self.token_emb(x) + pos_emb

        # for summing embeddings passed externally - needs this for self-conditioning in non-autoregressive training

        if exists(sum_embeds):
            x = x + sum_embeds

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)
        
        

        # whether to append embeds, as in PaLI, for image embeddings
        
        # print(f"x after post_emb_norm: {x.shape}")

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as text model dimensions'

            x = torch.cat((prepend_embeds, x), dim=-2)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # embedding dropout

        x = self.emb_dropout(x)
        
        # print("x after embedding dropout:", x.shape)  # Debugging line to check shape after dropout

        x = self.project_emb(x)

        if has_memory_tokens:
            mem_every = self.memory_tokens_interspersed_every

            if exists(mem_every):
                assert mem_every > 0
                assert isinstance(self.attn_layers, Decoder), 'only for decoder'
                next_seq_len = math.ceil(n / mem_every) * mem_every

                x = pad_at_dim(x, (0, next_seq_len - n), dim=-2, value=0.)
                x = rearrange(x, 'b (n m) d -> (b n) m d', m=mem_every)

            mem = repeat(self.memory_tokens, 'n d -> b n d', b=x.shape[0])
            x, mem_packed_shape = pack((mem, x), 'b * d')

            # auto-handle masking after appending memory tokens
            if not exists(mem_every) and exists(mask):
                mask = pad_at_dim(mask, (num_mems, 0), dim=-1, value=True)

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b=b)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        # print("x before attention layers:", x.shape) # (2, 8192, 2048)
        
        # New logic for penalty and reward tokens
        # print("before penalty_reward_mask layers:", x.shape)  # Debugging line to check shape before attention layers
        if penalty_reward_mask is not None:
            # pass
            penalty_reward_emb = self.penalty_reward_net(torch.cat([x[:, :, :], penalty_reward_mask], dim=-1))
            x[:, :, :] += penalty_reward_emb  # Assuming penalty_reward_mask is of shape (b, 1)


        # print("before attention layers:", x.shape)  # Debugging line to check shape before attention layers
        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, cache=cache, return_hiddens=True, seq_start_pos=seq_start_pos, **kwargs)

        # print("x after attention layers:", x.shape) # (2, 8192, 2048)
        
        if has_memory_tokens:
            if exists(mem_every):
                x = rearrange(x, 'b (n m) d -> (b n) m d', m=(mem_every + num_mems))

            mem, x = unpack(x, mem_packed_shape, 'b * d')

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b=b)

            x = x[:, :n]
        #     print("x after unpacking memory tokens:", x.shape)  # Debugging line to check shape after unpacking memory tokens
            
        # print("x before to_logits:", x.shape)  # Debugging line to check shape before logits
            # print(f"x after unpacking memory tokens: {x.shape}") # (2, 8192, 2048)
            
        
        # print("x contains NaN before to_logits:", torch.isnan(x).any())  # Debugging line to check NaN after attention layers
        # print("x contains Inf before to_logits:", torch.isinf(x).any())  # Debugging line to check Inf after attention layers

        if return_logits_and_embeddings:
            print("branch 1")
            out = (self.to_logits(x), x)
        elif return_embeddings:
            print("branch 2")
            out = x
        else:
            # print("branch 3")
            # print("to logits:", self.to_logits)
            # print("contains NaN in to_logits:", torch.isnan(self.to_logits.weight).any())  No Nan
            # print("contains Inf in to_logits:", torch.isinf(self.to_logits.weight).any())  No Inf
            # print("x contains NaN before to_logits:", torch.isnan(x).any())  # Debugging line to check NaN before logits
            # print("x contains Inf before to_logits:", torch.isinf(x).any())  # Debugging line to check Inf before logits
            # print("min in x", x.min())
            # print("max in x", x.max())
            # print("min in to_logits weight", self.to_logits.weight.min())
            # print("max in to_logits weight", self.to_logits.weight.max())
            # here is a Floating point exception to be debugged
            # see if x is all float
            # print("x dtype:", x.dtype)  # float32
            # print("to_logits weight dtype:", self.to_logits.weight.dtype)  # float32
            
            out = self.to_logits(x)
        
        # print("out after to_logits:", out.shape)  # Debugging line to check shape after logits

        if return_attn_z_loss:
            pre_softmax_attns = list(map(lambda t: t.pre_softmax_attn, intermediates.attn_intermediates))
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight=attn_z_loss_weight)
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out

class ContinuousTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        attn_layers,
        dim_in = None,
        dim_out = None,
        emb_dim = None,
        max_mem_len = 0,
        post_emb_norm = False,
        emb_dropout = 0.,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len

        if not (use_abs_pos_emb and not attn_layers.has_pos_emb):
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        self.post_emb_norm = nn.LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_in = nn.Linear(dim_in, dim) if exists(dim_in) else nn.Identity()

        self.attn_layers = attn_layers

        self.project_out = nn.Linear(dim, dim_out) if exists(dim_out) else nn.Identity()

    def forward(
        self,
        x,
        return_embeddings = False,
        return_intermediates = False,
        return_mems = False,
        mask = None,
        return_attn = False,
        mems = None,
        pos = None,
        prepend_embeds = None,
        **kwargs
    ):
        x = self.project_in(x)
        x = x + self.pos_emb(x, pos = pos)

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            _, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, return_hiddens = True, **kwargs)

        out = self.project_out(x) if not return_embeddings else x

        if return_intermediates:
            return out, intermediates

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), hiddens))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out

class XTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        tie_token_emb = False,
        ignore_index = -100,
        pad_value = 0,
        cross_attn_tokens_dropout = 0.,
        **kwargs
    ):
        super().__init__()
        enc_kwargs, kwargs = groupby_prefix_and_trim('enc_', kwargs)
        dec_kwargs, kwargs = groupby_prefix_and_trim('dec_', kwargs)

        assert 'dim' not in enc_kwargs and 'dim' not in dec_kwargs, 'dimension of either encoder or decoder must be set with `dim` keyword'
        enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], enc_kwargs)
        enc_transformer_kwargs['emb_dropout'] = enc_kwargs.pop('emb_dropout', 0)
        enc_transformer_kwargs['num_memory_tokens'] = enc_kwargs.pop('num_memory_tokens', None)
        enc_transformer_kwargs['scaled_sinu_pos_emb'] = enc_kwargs.pop('scaled_sinu_pos_emb', False)
        enc_transformer_kwargs['use_abs_pos_emb'] = enc_kwargs.pop('use_abs_pos_emb', True)

        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        dec_transformer_kwargs['emb_dropout'] = dec_kwargs.pop('emb_dropout', 0)
        dec_transformer_kwargs['scaled_sinu_pos_emb'] = dec_kwargs.pop('scaled_sinu_pos_emb', False)
        dec_transformer_kwargs['use_abs_pos_emb'] = dec_kwargs.pop('use_abs_pos_emb', True)

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout  # how many tokens from the encoder to dropout when cross attending from decoder - seen in a couple papers, including Perceiver AR - this will also be very effective regularization when cross attending to very long memories

        self.encoder = TransformerWrapper(
            **enc_transformer_kwargs,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )

        self.decoder = TransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers = Decoder(dim = dim, cross_attend = True, **dec_kwargs)
        )

        if tie_token_emb:
            self.decoder.token_emb = self.encoder.token_emb

        self.decoder = AutoregressiveWrapper(self.decoder, ignore_index=ignore_index, pad_value=pad_value)

    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, seq_len, mask = None, attn_mask = None, **kwargs):
        encodings = self.encoder(seq_in, mask = mask, attn_mask = attn_mask, return_embeddings = True)
        return self.decoder.generate(seq_out_start, seq_len, context = encodings, context_mask = mask, **kwargs)

    def forward(self, src, tgt, mask = None, attn_mask = None, src_prepend_embeds = None):

        if exists(src_prepend_embeds) and exists(mask):
            mask = pad_at_dim(mask, (src_prepend_embeds.shape[-2], 0), dim = -1, value = True)

        enc = self.encoder(src, mask = mask, attn_mask = attn_mask, prepend_embeds = src_prepend_embeds, return_embeddings = True)

        if self.training and self.cross_attn_tokens_dropout > 0:
            enc, mask = dropout_seq(enc, mask, self.cross_attn_tokens_dropout)

        out = self.decoder(tgt, context = enc, context_mask = mask)
        return out



def get_next_pos(img, ox, oy):
    # from (ox, oy), in order ox+1, then oy+1, return nx, ny where img[nx, ny] is not 0
    ny, nx = oy-1, ox
    x_plus = 0
    # print(img.shape)
    img_H, img_W = img.shape[:2]
    if ny <0:
        ny = img_H-1
        nx += 1
        x_plus+=1
    if nx >= img_W:
        return None, None, None
    return nx, ny, x_plus

def find_nxt_pixel(img, ox, oy):
    assert img.ndim == 2
    # from (ox, oy), in order ox+1, then oy+1, return nx, ny where img[nx, ny] is not 0
    nx, ny, _ = get_next_pos(img, ox, oy)
    if nx is None:
        return None, None
    while img[ny, nx] == 0:
        nx, ny, _ = get_next_pos(img, nx, ny)
        if nx is None:
            return None, None
    return nx, ny