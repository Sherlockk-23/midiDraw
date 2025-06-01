# from flask import Flask, request, send_file
# from flask_cors import CORS
from PIL import Image
import base64
import io

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
from GiantMusicTransformer import TMIDIX
from GiantMusicTransformer.midi_to_colab_audio import midi_to_colab_audio
from GiantMusicTransformer.x_transformer_1_23_2 import *
import random
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn import metrics
from IPython.display import Audio, display
from huggingface_hub import hf_hub_download

import cv2
from GiantMusicTransformer.my_utils import *


prompts = None

try_to_generate_outro = False #@param {type:"boolean"}
try_to_introduce_drums = False # @param {type:"boolean"}
number_of_tokens_to_generate = 1024 # @param {type:"slider", min:33, max:1024, step:3}
number_of_batches_to_generate = 1 #@param {type:"slider", min:1, max:16, step:1}
preview_length_in_tokens = 120 # @param {type:"slider", min:33, max:240, step:3}
number_of_memory_tokens = 7203 # @param {type:"slider", min:300, max:8190, step:3}
temperature = 1 # @param {type:"slider", min:0.1, max:1, step:0.05}
model_sampling_top_p_value = 0.96 # @param {type:"slider", min:0.1, max:1, step:0.01}
number_of_prime_tokens = 8190
#@markdown Other settings

allow_model_to_stop_generation_if_needed = False #@param {type:"boolean"}
render_MIDI_to_audio = True # @param {type:"boolean"}

trim_all_outputs_to_last_chord = False


def load_model():
    #@title Load Giant Music Transformer Pre-Trained Model
    select_model_to_load = "482M-8L-Ultra-Fast-Medium" 
    # @param ["482M-8L-Ultra-Fast-Medium","585M-32L-Very-Fast-Large","786M-44L-Fast-Extra-Large"]
    model_precision = "bfloat16" # @param ["bfloat16", "float16"]
    plot_tokens_embeddings = "None" 
    full_path_to_models_dir = "./GiantMusicTransformer/content/Giant-Music-Transformer/Models"

    model_path, mdim, num_layers, mrpe = get_model_path(select_model_to_load, full_path_to_models_dir)

    device_type = 'cuda'
    if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported():
        dtype = 'bfloat16'
    else:
        dtype = 'float16'

    if model_precision == 'float16':
        dtype = 'float16'
    ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    SEQ_LEN = 8192
    PAD_IDX = 19463
    model = TransformerWrapper(
            num_tokens = PAD_IDX+1,
            max_seq_len = SEQ_LEN,
            attn_layers = Decoder(dim = mdim,
                                depth = num_layers,
                                heads = 32,
                                rotary_pos_emb = mrpe,
                                attn_flash = True
                                )
    )

    model = AutoregressiveWrapper(model, ignore_index=PAD_IDX, pad_value=PAD_IDX)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    return model, ctx

def sample_prompt(data_path = "dataset/adl-piano-midi/aria-midi-v1-pruned-ext/data/ae"):
    all_midis = os.listdir(data_path)
    sample_midi = random.choice(all_midis)
    sample_midi_path = os.path.join(data_path, sample_midi)
    # sample_midi = TMIDIX.MidiFile(sample_midi_path)
    score = TMIDIX.midi2single_track_ms_score(open(sample_midi_path, 'rb').read(), recalculate_channels=False)
    full_prompt = score2melody_chords_f(score, number_of_prime_tokens=number_of_prime_tokens, 
                                                trim_all_outputs_to_last_chord=trim_all_outputs_to_last_chord)
    length = len(full_prompt)//3
    prompt_length = min(100, random.randint(30, length))
    st = random.randint(0, length-prompt_length)
    prompt = full_prompt[st*3:(st+prompt_length)*3]
    print(f"Sampled prompt from {sample_midi_path} with length {len(prompt)//3}")
    return prompt
    

from dataset.draw2midi import get_filtered_dataset

filtered_images = get_filtered_dataset("./dataset/quickdraw_dataset", drop_rate=0.03)
    
def sample_image():
    # Sample a random image from the dataset
    image = random.choice(filtered_images)
    image = image.reshape(100, 100).astype('uint8')
    return image

def generate_midi(model, ctx, idx):
    
    image = sample_image()
    prompts = sample_prompt()
    

    if image.sum() == 0:
        print("Image is empty. Please provide a valid image.")
        return {"status": "Image is empty. Please provide a valid image."}

    #generate output.mid from input.pn

    full_music = prompts if prompts is not None else []
    if len(full_music) == 0:
        print(prompts)
        if model is None:
            print("Model not loaded. Please load the model first.")
        print("No seed MIDI loaded. Please load a seed MIDI first.")
        return {"status": "No seed MIDI loaded. Please load a seed MIDI first."}

    print("Generating MIDI from input.png...")

    Fail  = True
    num_tries = 0
    while Fail:
        prompt_song = full_music[-number_of_memory_tokens+10:]
        out, outimg, castedimg= generate_with_img(
            model = model,
            melody_chords_f = prompt_song,
            img = image,
            ctx=ctx,
            number_of_tokens_to_generate = number_of_tokens_to_generate,
            number_of_batches_to_generate = number_of_batches_to_generate,
            number_of_memory_tokens = number_of_memory_tokens,
            temperature = temperature,
            model_sampling_top_p_value = model_sampling_top_p_value,
            try_to_generate_outro = try_to_generate_outro,
            try_to_introduce_drums = try_to_introduce_drums,
            allow_model_to_stop_generation_if_needed = allow_model_to_stop_generation_if_needed,
            )
        out0 = out.tolist()
        melody_chords_f = out0[0]
        melody_chords_f = trim_to_chord(melody_chords_f,
                                        enabled=trim_all_outputs_to_last_chord)

        song_f, patches = melody_chords2song_f(melody_chords_f)

        patches = [0 if x==-1 else x for x in patches]

        detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature = 'Giant Music Transformer',
                                                                output_file_name = f'./dataset/gen_midi/{idx}',
                                                                track_name='Project Los Angeles',
                                                                list_of_MIDI_patches=patches
                                                                )

        fname = f'./dataset/gen_midi/{idx}'

        
        # midi_audio = midi_to_colab_audio(fname + '.mid')
        # display(Audio(midi_audio, rate=16000, normalize=False))

        # TMIDIX.plot_ms_SONG(song_f,
        #                     plot_title=fname,
        #                     preview_length_in_notes=0
        #                     )
        # # plot image
        # outimg = cv2.flip(outimg, 0)
        # plt.imshow(outimg, cmap="gray")
        # plt.show()
        # castedimg = cv2.flip(castedimg, 0)
        # plt.imshow(castedimg, cmap="gray")
        # plt.show()

        print("sum of outimg:", outimg.sum())
        print("sum of castedimg:", castedimg.sum())
        print("ratio: ", outimg.sum() / castedimg.sum())

        if outimg.sum() / castedimg.sum() < 0.08:
            Fail = False
            full_music = full_music + melody_chords_f[-number_of_tokens_to_generate:]
            return True
        else:
            print("Image is not good enough, trying again...")
            num_tries += 1
            if num_tries > 10:
                print("Too many tries, stopping...")
                Fail = False
                return False
            
def main():
    model, ctx = load_model()
    num_to_gen = 180
    while num_to_gen > 0:
        idx = num_to_gen
        # num_to_gen -= 1
        print(f"Generating {idx}...")
        try:
            success = generate_midi(model, ctx, idx)
            if success:
                num_to_gen -= 1
        except Exception as e:
            print(f"Error generating {idx}: {e}")
            continue

main()