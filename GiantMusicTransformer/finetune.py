# Load modules and make data dir

print('Loading modules...')

import os
import pickle
import random
import secrets
import tqdm
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

# %cd /content/tegridy-tools/tegridy-tools/

import TMIDIX

# %cd /content/tegridy-tools/tegridy-tools/X-Transformer

from x_transformer_1_23_2 import *

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# %cd /content/

# if not os.path.exists('/content/INTS'):
#     os.makedirs('/content/INTS')

import random

from huggingface_hub import hf_hub_download

print('Done')

print('Torch version:', torch.__version__)

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'


dataset_addr = "../root/autodl-tmp/Dataset"

#==========================================================================

filez = list()
for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames]
print('=' * 70)

random.shuffle(filez)
filez = filez[:6]  # Limit to 1000 files for faster training

print('Loaded', len(filez), 'data files')
print('=' * 70)


#@title Load Giant Music Transformer Pre-Trained Model

#@markdown Choose model

select_model_to_load = "482M-8L-Ultra-Fast-Medium" # @param ["585M-32L-Very-Fast-Large", "786M-44L-Fast-Extra-Large"]

#@markdown Model precision option

model_precision = "float16" # @param ["bfloat16", "float16"]

#@markdown bfloat16 == Half precision/faster speed (if supported, otherwise the model will default to float16)

#@markdown float16 == Full precision/fast speed

plot_tokens_embeddings = "None" # @param ["None", "Start Times", "Durations Velocities", "Piano Pitches", "Drums Pitches", "Aux"]

print('=' * 70)
print('Loading Giant Music Transformer', select_model_to_load,'Pre-Trained Model...')
print('Please wait...')
print('=' * 70)

full_path_to_models_dir = "./content/Giant-Music-Transformer/Models"

if select_model_to_load == '786M-44L-Fast-Extra-Large':

  model_checkpoint_file_name = 'Giant_Music_Transformer_Extra_Large_Trained_Model_18001_steps_0.2657_loss_0.9272_acc.pth'
  model_path = full_path_to_models_dir+'/Extra Large/'+model_checkpoint_file_name
  num_layers = 44
  mdim = 1024
  mrpe = False
  if os.path.isfile(model_path):
    print('Model already exists...')

  else:
    hf_hub_download(repo_id='asigalov61/Giant-Music-Transformer',
                    filename=model_checkpoint_file_name,
                    local_dir='./content/Giant-Music-Transformer/Models/Extra Large',
                    local_dir_use_symlinks=False)

elif select_model_to_load == '585M-32L-Very-Fast-Large':

  model_checkpoint_file_name = 'Giant_Music_Transformer_Large_Trained_Model_36074_steps_0.3067_loss_0.927_acc.pth'
  model_path = full_path_to_models_dir+'/Large/'+model_checkpoint_file_name
  num_layers = 32
  mdim = 1024
  mrpe = False
  if os.path.isfile(model_path):
    print('Model already exists...')

  else:
    hf_hub_download(repo_id='asigalov61/Giant-Music-Transformer',
                    filename=model_checkpoint_file_name,
                    local_dir='./content/Giant-Music-Transformer/Models/Large',
                    local_dir_use_symlinks=False)
    
else: 
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
                    local_dir_use_symlinks=False)

print('=' * 70)
print('Instantiating model...')

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda'

if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported():
  dtype = 'bfloat16'
else:
  dtype = 'float16'

if model_precision == 'float16':
  dtype = 'float16'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

SEQ_LEN = 2048

# instantiate the model

model = TransformerWrapper(
    num_tokens = 19464,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = mdim, depth = num_layers, heads = 32, rotary_pos_emb = mrpe, attn_flash = True),
    penalty_reward_tokens=True
)

model = AutoregressiveWrapper(model, ignore_index=19463)

model.cuda()
print('=' * 70)

print('Loading model checkpoint...')

# model.load_state_dict(torch.load(model_path))

pretrained_state_dict = torch.load(model_path)

def init_weights_linear(m):
    if isinstance(m, nn.Linear):
        # mean to 0, std to 0.02
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# 初始化新增的结构
model.net.penalty_reward_net.apply(init_weights_linear)

# 加载预训练模型的权重到当前模型
model_state_dict = model.state_dict()
pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
print('Loaded', len(pretrained_state_dict), 'keys from the pre-trained model state dict')
print("pretrained_state_dict keys:", list(pretrained_state_dict.keys())[:10])  # Print first 10 keys for verification
model_state_dict.update(pretrained_state_dict)
model.load_state_dict(model_state_dict)

print(model)

print('=' * 70)

model.eval()

print('Done!')
print('=' * 70)

print('Model will use', dtype, 'precision...')
print('=' * 70)

# Model stats
# summary(model, (SEQ_LEN,))
# Setup model

# constants

NUM_DATA_FILES_TO_LOAD_PER_ITER = 3

SEQ_LEN = 2048 # Models seq len (must be divisible by 4)
PAD_IDX = 19463 # Models pad index

#=========================================================
# Fine-Tuning params
#=========================================================

NUM_EPOCHS = 30 # This number depends on your dataset and desired final loss/acc

BATCH_SIZE = 4 # Original and therefore optimal is 4
GRADIENT_ACCUMULATE_EVERY = 4 # Original and therefore optimal is 4

LEARNING_RATE = 1e-4 # A 1/20th of the original 2e-4
NUM_LAST_LAYERS_TO_UNFREEZE = 8 # I think it should be at least 2 and probably no more than 8 here

#=========================================================

VALIDATE_EVERY  = 100
SAVE_EVERY = 200
GENERATE_EVERY  = 400
GENERATE_LENGTH = 1024
PRINT_STATS_EVERY = 2

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

# prep the model

# Freeze all layers in the model
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last few layers in attn_layers
for param in model.net.attn_layers.layers[-NUM_LAST_LAYERS_TO_UNFREEZE:].parameters():  # replace N with the number of layers you want to unfreeze
    param.requires_grad = True
    
for param in model.net.penalty_reward_net.parameters():  # replace N with the number of layers you want to unfreeze
    param.requires_grad = True


# Dataloader

class MusicDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):

        # consequtive sampling

        full_seq = torch.Tensor(self.data[index][:self.seq_len+1]).long()

        return full_seq.cuda()

    def __len__(self):
        return (len(self.data) // BATCH_SIZE) * BATCH_SIZE

# precision/optimizer/scaler

dtype = torch.float16

ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scaler = torch.cuda.amp.GradScaler(enabled=True)

print('Done!')


# Train the model
import random

# empty some cache 
torch.cuda.empty_cache()

CHUNKS_LENGTH = SEQ_LEN+1
MIN_NUMBER_OF_CHUNK_EVENTS = 512 # min number of tokens per chunk (must be divisible by 4)

train_losses = []
val_losses = []

train_accs = []
val_accs = []

nsteps = 0

for fa in range(0, len(filez), NUM_DATA_FILES_TO_LOAD_PER_ITER):
    

      #==========================================================================
        print('=' * 70)
        print('Loading data files', fa, '---', fa+NUM_DATA_FILES_TO_LOAD_PER_ITER-1)
        print('Please wait...')
        print('=' * 70)

        train_data = []

        chunks_counter = 0
        discarted_chunks_counter = 1

        for lfa in tqdm.tqdm(filez[fa:fa+NUM_DATA_FILES_TO_LOAD_PER_ITER]):

            train_d = pickle.load(open(lfa, 'rb'))
            random.shuffle(train_d)
            for t in train_d:
                for i in range(0, len(t), int((SEQ_LEN-512))):

                  #=========================================================================
                  # collecting all possible chunks of chunks length
                  
                  if len(t) < SEQ_LEN+512:
                    print('Chunk is too small, skipping...')
                    continue

                  if 0 <= max(t[i:i+CHUNKS_LENGTH]) < PAD_IDX: # final data integrity check
                    if len(t[i:i+CHUNKS_LENGTH]) == CHUNKS_LENGTH:
                      train_data.append(t[i:i+CHUNKS_LENGTH])

                    else:
                      if len(t[i:i+CHUNKS_LENGTH]) > MIN_NUMBER_OF_CHUNK_EVENTS:
                        td = t[i:i+CHUNKS_LENGTH] + [PAD_IDX] * (CHUNKS_LENGTH-len(t[i:i+CHUNKS_LENGTH])) # padding with pad index
                        train_data.append(td)
                      else:
                        discarted_chunks_counter += 1

                    chunks_counter += 1

                  else:
                    print('Bad data!!!')
                    break

                #=========================================================================
                # Collecting middle chunk if it larger than chunks length
                # print('len(t)', len(t))
                # print('SEQ_LEN', SEQ_LEN)
                # print("max(t)", max(t))
                # print("PAD_IDX", PAD_IDX)
                if 0 <= max(t) < PAD_IDX: # final data integrity check
                    if len(t) >= SEQ_LEN+8:
                        comp_middle = int(len(t) / 2)
                        sidx = int(comp_middle -(SEQ_LEN / 2))
                        train_data.append(t[sidx:sidx+CHUNKS_LENGTH])

                    else:
                        discarted_chunks_counter += 1

                    chunks_counter += 1

                else:
                  print('Bad data!!!')
                  break

        #==========================================================================

        print('Done!')
        print('=' * 70)
        print('Total number of imput chunks:', chunks_counter)
        print('Total number of good chunks:', len(train_data))
        print('Total number of discarted chunks:', discarted_chunks_counter, '/', round(100 * discarted_chunks_counter/chunks_counter, 3), '%')
        print('All data is good:', len(max(train_data, key=len)) == len(min(train_data, key=len)))
        print('=' * 70)
        print('Final data randomization...')
        random.shuffle(train_data)
        print('Done!')
        print('=' * 70)

        train_dataset = MusicDataset(train_data, SEQ_LEN)
        val_dataset   = MusicDataset(train_data, SEQ_LEN)
        train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
        val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

        NUM_BATCHES = (len(train_data) // BATCH_SIZE // GRADIENT_ACCUMULATE_EVERY) * NUM_EPOCHS
        
        print("len(train_data)", len(train_data))
        print("NUM_BATCHES", NUM_BATCHES)

        for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='Training'):
            model.train()
            
            # [TODO] add penalty reward mask to the model
            
            
            for __ in range(GRADIENT_ACCUMULATE_EVERY):
                
                x = next(train_loader)
                penalty_reward_mask = torch.zeros((BATCH_SIZE, SEQ_LEN, PAD_IDX+1), dtype=torch.float16, device='cuda')
                
                
                
                # print('x.shape', x.shape)
                # print('x[-1]', x[:, -1]) # one to predicr
                for b in range(BATCH_SIZE):
                    for t in range(SEQ_LEN):
                        tar = x[b, t+1] # last token
                        rd = random.random()
                        if tar >=0 and tar <= 256:   # time
                            if rd<0.1:
                                pass
                            elif rd<0.6:
                                mask_len = random.randint(1, 10)
                                penalty_reward_mask[b, t, :mask_len] = -2
                            else:
                                mask_len = random.randint(1, 10)
                                penalty_reward_mask[b, t, :mask_len] = 2
                        elif tar >= 256 and tar < 2304: # duration and velocity
                            # pass
                            masks = random.sample(range(1, 2048), random.randint(10, 55))
                            if rd<0.1:
                                pass
                            elif rd<0.6:
                                for m in masks:
                                    penalty_reward_mask[b, t, 256 + m] = -2
                            else:
                                for m in masks:
                                    penalty_reward_mask[b, t, 256 + m] = 2
                        elif tar >= 2304 and tar < 2304 + 128: # piano
                            if rd<0.1:
                                pass
                            elif rd<0.4:
                                masks = random.sample(range(0, 128), random.randint(5, 35))
                                for m in masks:
                                    penalty_reward_mask[b, t, 2304 + m] = -2
                            elif rd<0.6:
                                masks = random.sample(range(0, 128), random.randint(5, 35))
                                for m in masks:
                                    penalty_reward_mask[b, t, 2304 + m] = 2
                            elif rd<0.8:
                                middle_mask = random.randint(40, 90)
                                for m in range(middle_mask-3, middle_mask+3):
                                    penalty_reward_mask[b, t, 2304 + m] = -2
                            else:
                                middle_mask = random.randint(40, 90)
                                for m in range(middle_mask-3, middle_mask+3):
                                    penalty_reward_mask[b, t, 2304 + m] = 2
                            
                        
                            
                        
                        
                
                with ctx:
                    loss, pr_loss, acc = model(x, 
                                      penalty_reward_mask=penalty_reward_mask,
                                    #   return_acc=True
                                      )
                    # loss, acc = model(x)
                loss = loss / GRADIENT_ACCUMULATE_EVERY
                scaler.scale(loss).backward(torch.ones(loss.shape).cuda())
            # print("i", i)
            if i % PRINT_STATS_EVERY == 0:
                print(f'Training loss: {loss.mean().item() * GRADIENT_ACCUMULATE_EVERY}')
                print(f'Penalty reward loss: {pr_loss.mean().item()}')
                print(f'Training acc: {acc.mean().item()}')

            train_losses.append(loss.mean().item() * GRADIENT_ACCUMULATE_EVERY)
            train_accs.append(acc.mean().item())

            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            nsteps += 1

            if i % VALIDATE_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    with ctx:
                        val_loss, val_acc = model(next(val_loader))

                        print(f'Validation loss: {val_loss.mean().item()}')
                        print(f'Validation acc: {val_acc.mean().item()}')

                        val_losses.append(val_loss.mean().item())
                        val_accs.append(val_acc.mean().item())

                        print('Plotting training loss graph...')

                        tr_loss_list = train_losses
                        plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                        plt.show()
                        plt.close()
                        print('Done!')

                        print('Plotting training acc graph...')

                        tr_loss_list = train_accs
                        plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                        plt.show()
                        plt.close()
                        print('Done!')

                        print('Plotting validation loss graph...')
                        tr_loss_list = val_losses
                        plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                        plt.show()
                        plt.close()
                        print('Done!')

                        print('Plotting validation acc graph...')
                        tr_loss_list = val_accs
                        plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                        plt.show()
                        plt.close()
                        print('Done!')

            if i % GENERATE_EVERY == 0:
                model.eval()

                inp = random.choice(val_dataset)[:512]

                print(inp)

                # with ctx:
                sample = model.generate(inp[None, ...], GENERATE_LENGTH)

                print(sample)

                data = sample.tolist()[0]

                print('Sample INTs', data[:15])

                out = data[:200000]

                if len(out) != 0:

                    song = out
                    song_f = []

                    time = 0
                    dur = 0
                    vel = 90
                    pitch = 0
                    channel = 0

                    patches = [-1] * 16

                    channels = [0] * 16
                    channels[9] = 1

                    for ss in song:

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

                            song_f.append(['note', time, dur, channel, pitch, vel ])

                patches = [0 if x==-1 else x for x in patches]

                detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                          output_signature = 'Giant Music Transformer',
                                                                          output_file_name = './content/Giant-Music-Transformer-Composition',
                                                                          track_name='Project Los Angeles',
                                                                          list_of_MIDI_patches=patches
                                                                          )

                print('Done!')

            if i % SAVE_EVERY == 0:

                print('Saving model progress. Please wait...')
                print('model_checkpoint_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth')

                fname = './content/model_checkpoint_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth'

                torch.save(model.state_dict(), fname)

                data = [train_losses, train_accs, val_losses, val_accs]

                TMIDIX.Tegridy_Any_Pickle_File_Writer(data, './content/losses_accs')

                print('Done!')

#======================================================================================================

print('Saving model progress. Please wait...')
print('model_checkpoint_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth')

fname = './content/model_checkpoint_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth'

torch.save(model.state_dict(), fname)

print('Done!')

data = [train_losses, train_accs, val_losses, val_accs]

TMIDIX.Tegridy_Any_Pickle_File_Writer(data, './content/losses_accuracies')

# Save training loss graph

plt.plot([i for i in range(len(train_losses))] ,train_losses, 'b')
plt.savefig('./content/training_loss_graph.png')
plt.close()
print('Done!')

# Save training acc graph

plt.plot([i for i in range(len(train_accs))] ,train_accs, 'b')
plt.savefig('./content/training_acc_graph.png')
plt.close()
print('Done!')

# Save validation loss graph

plt.plot([i for i in range(len(val_losses))] ,val_losses, 'b')
plt.savefig('./content/validation_loss_graph.png')
plt.close()
print('Done!')

# Save validation acc graph

plt.plot([i for i in range(len(val_accs))] ,val_accs, 'b')
plt.savefig('./content/validation_acc_graph.png')
plt.close()
print('Done!')