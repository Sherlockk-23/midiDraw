#@title Import modules
import os
import copy
import pickle
import secrets
import statistics
from time import time
import tqdm
os.environ['USE_FLASH_ATTENTION'] = '1'
import torch
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)
# import TMIDIX
# from midi_to_colab_audio import midi_to_colab_audio
# from x_transformer_1_23_2 import *
import random
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn import metrics
from IPython.display import Audio, display
from huggingface_hub import hf_hub_download
import cv2

import torch.nn.functional as F


def get_model_path(select_model_to_load, full_path_to_models_dir):
    if select_model_to_load == '786M-44L-Fast-Extra-Large':
        model_checkpoint_file_name = 'Giant_Music_Transformer_Extra_Large_Trained_Model_18001_steps_0.2657_loss_0.9272_acc.pth'
        model_path = full_path_to_models_dir+'/Extra Large/'+model_checkpoint_file_name
        mdim = 1024
        num_layers = 44
        mrpe = False
        if os.path.isfile(model_path):
            print('Model already exists...')
        else:
            hf_hub_download(repo_id='asigalov61/Giant-Music-Transformer',
                            filename=model_checkpoint_file_name,
                            local_dir='./content/Giant-Music-Transformer/Models/Extra Large',
                            )
    elif select_model_to_load == '585M-32L-Very-Fast-Large':
        model_checkpoint_file_name = 'Giant_Music_Transformer_Large_Trained_Model_36074_steps_0.3067_loss_0.927_acc.pth'
        model_path = full_path_to_models_dir+'/Large/'+model_checkpoint_file_name
        mdim = 1024
        num_layers = 32
        mrpe = False
        if os.path.isfile(model_path):
            print('Model already exists...')
        else:
            hf_hub_download(repo_id='asigalov61/Giant-Music-Transformer',
                            filename=model_checkpoint_file_name,
                            local_dir='./content/Giant-Music-Transformer/Models/Large',
                            )
    elif select_model_to_load == '482M-8L-Ultra-Fast-Medium':
        model_checkpoint_file_name = 'Giant_Music_Transformer_Medium_Trained_Model_42174_steps_0.5211_loss_0.8542_acc.pth'
        model_path = full_path_to_models_dir+'/Medium/'+model_checkpoint_file_name
        mdim = 2048
        num_layers = 8
        mrpe = True
        if os.path.isfile(model_path):
            print('Model already exists...')
        else:
            hf_hub_download(repo_id='asigalov61/Giant-Music-Transformer',
                            filename=model_checkpoint_file_name,
                            local_dir='./content/Giant-Music-Transformer/Models/Medium',
                            )
    return model_path, mdim, num_layers, mrpe

def trim_to_chord(lst, max_trim_len=30, enabled=False):

  if enabled:

    # Reverse the list
    reversed_lst = lst[::-1]
    idx = 0
    for i, value in enumerate(reversed_lst):
        # Check if the value is non-zero
        if 0 < value < 256:
            # Convert the index to be from the end of the list
            idx = len(lst) - i - 1
            break

    trimmed_list = lst[:idx]

    if (len(lst) - len(trimmed_list)) <= max_trim_len:

      return trimmed_list

    else:
      return lst

  else:
    return lst
  

 
def score2melody_chords_f(score, number_of_prime_tokens=8190, 
                          trim_all_outputs_to_last_chord=False):

  # INSTRUMENTS CONVERSION CYCLE
  events_matrix = []
  itrack = 1
  patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  while itrack < len(score):
      for event in score[itrack]:
          if event[0] == 'note' or event[0] == 'patch_change':
              events_matrix.append(event)
      itrack += 1

  events_matrix.sort(key=lambda x: x[1])

  events_matrix1 = []

  for event in events_matrix:
          if event[0] == 'patch_change':
                patches[event[2]] = event[3]

          if event[0] == 'note':
                event.extend([patches[event[3]]])

                if events_matrix1:
                    if (event[1] == events_matrix1[-1][1]):
                        if ([event[3], event[4]] != events_matrix1[-1][3:5]):
                            events_matrix1.append(event)
                    else:
                        events_matrix1.append(event)

                else:
                    events_matrix1.append(event)

  if len(events_matrix1) > 0:
      if min([e[1] for e in events_matrix1]) >= 0 and min([e[2] for e in events_matrix1]) >= 0:

          instruments_list_without_drums = list(set([y[3] for y in events_matrix1 if y[3] != 9]))
          instruments_list = list(set([y[3] for y in events_matrix1]))

          if len(events_matrix1) > 0 and len(instruments_list_without_drums) > 0:

              for e in events_matrix1:
                  e[1] = int(e[1] / 16)
                  e[2] = int(e[2] / 16)

              events_matrix1.sort(key=lambda x: x[6])
              events_matrix1.sort(key=lambda x: x[4], reverse=True)
              events_matrix1.sort(key=lambda x: x[1])


              melody_chords = []
              melody_chords2 = []

              if 9 in instruments_list:
                  drums_present = 19331 # Yes
              else:
                  drums_present = 19330 # No

              if events_matrix1[0][3] != 9:
                  pat = events_matrix1[0][6]
              else:
                  pat = 128

              melody_chords.extend([19461, drums_present, 19332+pat]) # Intro seq

              pe = events_matrix1[0]
              for e in events_matrix1:
                  delta_time = max(0, min(255, e[1]-pe[1]))
                  dur = max(0, min(255, e[2]))
                  cha = max(0, min(15, e[3]))

                  # Patches
                  if cha == 9: # Drums patch will be == 128
                      pat = 128

                  else:
                      pat = e[6]

                  # Pitches

                  ptc = max(1, min(127, e[4]))

                  # Velocities

                  # Calculating octo-velocity
                  vel = max(8, min(127, e[5]))
                  velocity = round(vel / 15)-1


                  dur_vel = (8 * dur) + velocity
                  pat_ptc = (129 * pat) + ptc

                  melody_chords.extend([delta_time, dur_vel+256, pat_ptc+2304])
                  melody_chords2.append([delta_time, dur_vel+256, pat_ptc+2304])

                  pe = e

  melody_chords = melody_chords[:number_of_prime_tokens]
  melody_chords_f = trim_to_chord(melody_chords,
                                  enabled=trim_all_outputs_to_last_chord)
  
  return melody_chords_f


def melody_chords2song_f(melody_chords_f):

  song_f = []

  time = 0
  dur = 0
  vel = 90
  pitch = 0
  channel = 0

  patches = [-1] * 16

  channels = [0] * 16
  channels[9] = 1

  for ss in melody_chords_f:

      if 0 <= ss < 256:

          time += ss * 16

      if 256 <= ss < 2304:

          dur = ((ss-256) // 8) * 16
          vel = (((ss-256) % 8)+1) * 15

      if 2304 <= ss < 18945:

          patch = (ss-2304) // 129

          if patch < 128:

              if patch not in patches:
                if 0 in channels:
                    cha = channels.index(0)
                    channels[cha] = 1
                else:
                    cha = 15

                patches[cha] = patch
                channel = patches.index(patch)
              else:
                channel = patches.index(patch)

          if patch == 128:
              channel = 9

          pitch = (ss-2304) % 129

          song_f.append(['note', time, dur, channel, pitch, vel, patch ])

  return song_f, patches


# generate params
# try_to_generate_outro = False #@param {type:"boolean"}
# try_to_introduce_drums = False # @param {type:"boolean"}
# number_of_tokens_to_generate = 2048 # @param {type:"slider", min:33, max:1024, step:3}
# number_of_batches_to_generate = 1 #@param {type:"slider", min:1, max:16, step:1}
# preview_length_in_tokens = 120 # @param {type:"slider", min:33, max:240, step:3}
# number_of_memory_tokens = 7203 # @param {type:"slider", min:300, max:8190, step:3}
# temperature = 1 # @param {type:"slider", min:0.1, max:1, step:0.05}
# model_sampling_top_p_value = 0.96 # @param {type:"slider", min:0.1, max:1, step:0.01}

# #@markdown Other settings

# allow_model_to_stop_generation_if_needed = False #@param {type:"boolean"}
# render_MIDI_to_audio = True # @param {type:"boolean"}

def cast_img_to_input(img):
    print("sum of image pixels:", img.sum())
    if img.ndim == 3:
        if img.shape[2] == 4:  # 检查是否有透明通道

            # 将透明通道转换为灰度值
            alpha_channel = img[:, :, 3]
            print("sum of alpha_channel pixels:", alpha_channel.sum())
            # 将透明通道的值应用到灰度图像上
            gray_image = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            print("sum of gray_image pixels:", gray_image.sum())
            gray_image = cv2.bitwise_and(gray_image, gray_image, mask=alpha_channel)
            print("sum of alpha_channel pixels:", img.sum())
            img = alpha_channel
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print("sum of cvtColor pixels:", img.sum())


    # img = 255-img
    img_sum = img.sum()
    if img_sum > (255*img.shape[0]*img.shape[1])//2:
        print("Image is too bright, inverting it")
        img = 255 - img

    # show image
    plt.imshow(img, cmap='gray')
    print("Image shape:", img.shape)
    print("Image sum:", img_sum)

    min_y = img.shape[0]
    min_x = img.shape[1]
    max_x = 0
    max_y = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] >0:
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
    img = img[max(0, min_y-10):min(max_y+10, img.shape[0]), min_x:max_x]
    print("Image shape after cropping:", img.shape)

    h, w = img.shape

    img = cv2.resize(img, (w*50//h, 50))
    h, w = img.shape
    img = cv2.resize(img, (int(w*2), h))
    if w*2 > 200:
        print("Image is too wide, cutting it to be thinner")
        img = img[:, :200]
    # upside down
    img = cv2.flip(img, 0)



    return img

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def generate_with_img(model, melody_chords_f, img, ctx, 
                      number_of_tokens_to_generate=2048, 
                      number_of_batches_to_generate=1,
                      temperature=1, 
                      model_sampling_top_p_value=0.96,
                      number_of_memory_tokens=7203,
                      allow_model_to_stop_generation_if_needed=False,
                      try_to_generate_outro=False,
                      try_to_introduce_drums=False,
                      only_out = False):

    if allow_model_to_stop_generation_if_needed:
        min_stop_token = 19462
    else:
        min_stop_token = None


    mel_cho = melody_chords_f[-number_of_memory_tokens:]

    if try_to_introduce_drums:
        last_note = mel_cho[-3:]
        drums = [36, 38, 47, 54]
        if 0 <= last_note[0] < 256:
            for d in random.choices(drums, k = random.randint(1, len(drums))):
                mel_cho.extend([0, min(((4*8)+5)+256, last_note[1]), ((128*129)+d)+2304])

    if try_to_generate_outro:
        mel_cho.extend([18945])

    torch.cuda.empty_cache()

    inp = [mel_cho] * number_of_batches_to_generate

    inp = torch.LongTensor(inp).cuda()


    # to grey
    img = cast_img_to_input(img)
    # ori_img = img.clone()
    ori_img = copy.deepcopy(img)

    with ctx:
        with torch.inference_mode():
            out = model.generateWithPic(inp,img,
                                number_of_tokens_to_generate,
                                filter_logits_fn=top_p,
                                filter_kwargs={'thres': model_sampling_top_p_value},
                                temperature=temperature,
                                return_prime=False,
                                eos_token=min_stop_token,
                                verbose=True)
            
    if only_out:
        return out
    else:
        return out, img, ori_img


def generate_without_img(model, melody_chords_f, ctx, 
                      number_of_tokens_to_generate=2048, 
                      number_of_batches_to_generate=1,
                      temperature=1, 
                      model_sampling_top_p_value=0.96,
                      number_of_memory_tokens=7203,
                      allow_model_to_stop_generation_if_needed=False,
                      try_to_generate_outro=False,
                      try_to_introduce_drums=False):

    if allow_model_to_stop_generation_if_needed:
        min_stop_token = 19462
    else:
        min_stop_token = None


    mel_cho = melody_chords_f[-number_of_memory_tokens:]

    if try_to_introduce_drums:
        last_note = mel_cho[-3:]
        drums = [36, 38, 47, 54]
        if 0 <= last_note[0] < 256:
            for d in random.choices(drums, k = random.randint(1, len(drums))):
                mel_cho.extend([0, min(((4*8)+5)+256, last_note[1]), ((128*129)+d)+2304])

    if try_to_generate_outro:
        mel_cho.extend([18945])

    torch.cuda.empty_cache()

    inp = [mel_cho] * number_of_batches_to_generate

    inp = torch.LongTensor(inp).cuda()


    with ctx:
        with torch.inference_mode():
            out = model.generate(inp,
                                number_of_tokens_to_generate,
                                filter_logits_fn=top_p,
                                filter_kwargs={'thres': model_sampling_top_p_value},
                                temperature=temperature,
                                return_prime=False,
                                eos_token=min_stop_token,
                                verbose=True)
    return out