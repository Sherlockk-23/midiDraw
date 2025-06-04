from flask import Flask, request, send_file
from flask_cors import CORS
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
import TMIDIX
from midi_to_colab_audio import midi_to_colab_audio
from x_transformer_1_23_2 import *
import random
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn import metrics
from IPython.display import Audio, display
from huggingface_hub import hf_hub_download

import cv2
from my_utils import *

app = Flask(__name__)
CORS(app)

app.model = None
app.prompts = None
app.ctx = None

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

@app.route("/load-model", methods=['get', 'post'])
def load_model():
    
    #@title Load Giant Music Transformer Pre-Trained Model
    select_model_to_load = "482M-8L-Ultra-Fast-Medium" 
    if app.model is not None:
        return {"status": "Model already loaded", "model_name": select_model_to_load}
    # @param ["482M-8L-Ultra-Fast-Medium","585M-32L-Very-Fast-Large","786M-44L-Fast-Extra-Large"]
    model_precision = "bfloat16" # @param ["bfloat16", "float16"]
    plot_tokens_embeddings = "None" 
    full_path_to_models_dir = "./content/Giant-Music-Transformer/Models"

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

    app.model = model
    app.ctx = ctx

    return {"status": "Model loaded successfully", "model_name": select_model_to_load}


@app.route("/load-seed-midi", methods=['get', 'post'])
def load_seed_midi():
    #@title Load Seed MIDI
    select_seed_MIDI = "Giant-Music-Transformer-Piano-Seed-2" 
    # @param ["Upload your own custom MIDI", "Giant-Music-Transformer-Piano-Seed-1", "Giant-Music-Transformer-Piano-Seed-2", "Giant-Music-Transformer-Piano-Seed-3", "Giant-Music-Transformer-Piano-Seed-4", "Giant-Music-Transformer-Piano-Seed-5", "Giant-Music-Transformer-Piano-Seed-6", "Giant-Music-Transformer-MI-Seed-1", "Giant-Music-Transformer-MI-Seed-2", "Giant-Music-Transformer-MI-Seed-3", "Giant-Music-Transformer-MI-Seed-4", "Giant-Music-Transformer-MI-Seed-5", "Giant-Music-Transformer-MI-Seed-6"]
    number_of_prime_tokens = 8190 # @param {type:"slider", min:90, max:8190, step:3}
    trim_all_outputs_to_last_chord = False # @param {type:"boolean"}
    render_MIDI_to_audio = False # @param {type:"boolean"}

    f = ''

    if select_seed_MIDI != "Upload your own custom MIDI":
        print('Loading seed MIDI...')
        f = './Seeds/'+select_seed_MIDI+'.mid'

    if f != '':

        score = TMIDIX.midi2single_track_ms_score(open(f, 'rb').read(), recalculate_channels=False)

        melody_chords_f = score2melody_chords_f(score, number_of_prime_tokens=number_of_prime_tokens, 
                                                trim_all_outputs_to_last_chord=trim_all_outputs_to_last_chord)

        song = melody_chords_f

        song_f, patches = melody_chords2song_f(song)

        #=======================================================

        print('=' * 70)
        print('Composition stats:')
        print('Composition has', int(len(melody_chords_f) / 3), 'notes')
        print('Composition has', len(melody_chords_f), 'tokens')
        print('Composition MIDI patches:', sorted(list(set([((y-2304) // 129) for y in melody_chords_f if 2304 <= y < 18945]))))
        print('=' * 70)

        fname = './content/output/Giant-Music-Transformer-Seed-Composition'

        block_lines = [(song_f[-1][1] / 1000)]
        block_tokens = [min(len(melody_chords_f), number_of_prime_tokens)]
        pblock = []

        if render_MIDI_to_audio:
            midi_audio = midi_to_colab_audio(fname + '.mid')
            display(Audio(midi_audio, rate=16000, normalize=False))

        TMIDIX.plot_ms_SONG(song_f, plot_title=fname)

    else:
        print('=' * 70)

    prompts = melody_chords_f
    app.prompts = prompts

    return {
        "status": "Seed MIDI loaded successfully",
        "seed_midi_name": select_seed_MIDI,
        "number_of_tokens": len(prompts),
        "number_of_notes": int(len(prompts) / 3)
    }
    

def sample_prompt(data_path = "../dataset/adl-piano-midi/aria-midi-v1-pruned-ext/data/ae"):
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


@app.route("/generate-midi", methods=['get', 'post'])
def generate_midi():
    print("Generating MIDI from input.png...")
    data = request.get_json()
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    PNGimage = Image.open(io.BytesIO(image_bytes))
    PNGimage.save("Seeds/pics/input.png")
    
    # transform PNGimage to cv2 image
    image = cv2.imread("Seeds/pics/input.png", cv2.IMREAD_UNCHANGED)

    if image.sum() == 0:
        print("Image is empty. Please provide a valid image.")
        return {"status": "Image is empty. Please provide a valid image."}

    #generate output.mid from input.png

    model = app.model
    # prompts = sample_prompt()
    prompts = app.prompts
    ctx = app.ctx

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
    best_midi = None
    best_rate = 1
    while Fail:
        prompts = sample_prompt()
        prompt_song = full_music[-number_of_memory_tokens+10:]
        try:
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
        except Exception as e:
            print(f"Error during generation: {e}")
            num_tries += 1
            print("Retrying generation...")
            continue
        out0 = out.tolist()
        melody_chords_f = out0[0]
        melody_chords_f = trim_to_chord(melody_chords_f,
                                        enabled=trim_all_outputs_to_last_chord)

        song_f, patches = melody_chords2song_f(melody_chords_f)

        patches = [0 if x==-1 else x for x in patches]

        detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature = 'Giant Music Transformer',
                                                                output_file_name = './content/output/lala',
                                                                track_name='Project Los Angeles',
                                                                list_of_MIDI_patches=patches
                                                                )

        fname = './content/output/lala'

        
        midi_audio = midi_to_colab_audio(fname + '.mid')
        display(Audio(midi_audio, rate=16000, normalize=False))

        TMIDIX.plot_ms_SONG(song_f,
                            plot_title=fname,
                            preview_length_in_notes=0
                            )
        # plot image
        outimg = cv2.flip(outimg, 0)
        plt.imshow(outimg, cmap="gray")
        plt.show()
        castedimg = cv2.flip(castedimg, 0)
        plt.imshow(castedimg, cmap="gray")
        plt.show()

        print("sum of outimg:", outimg.sum())
        print("sum of castedimg:", castedimg.sum())
        print("ratio: ", outimg.sum() / castedimg.sum())

        if outimg.sum() / castedimg.sum() < 0.08:
            Fail = False
            full_music = full_music + melody_chords_f[-number_of_tokens_to_generate:]
        else:
            print("Image is not good enough, trying again...")
            num_tries += 1
            if outimg.sum() / castedimg.sum() < best_rate:
                best_rate = outimg.sum() / castedimg.sum()
                best_midi = melody_chords_f
            if num_tries > 5:
                print("Too many tries, stopping...")
                Fail = False
                full_music = full_music + melody_chords_f[-number_of_tokens_to_generate:]
                song_f, patches = melody_chords2song_f(best_midi)
                patches = [0 if x==-1 else x for x in patches]
                detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(
                    song_f,
                    output_signature='Giant Music Transformer',
                    output_file_name=fname,
                    track_name='Project Los Angeles',
                    list_of_MIDI_patches=patches
                )

    return send_file(fname + '.mid', mimetype="audio/midi")

if __name__ == "__main__":
    app.run(port=5000)
    # app.run(host='0.0.0.0', port=5000)
