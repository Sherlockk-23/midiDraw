from my_utils import *


def load_model():
    #@title Load Giant Music Transformer Pre-Trained Model
    select_model_to_load = "482M-8L-Ultra-Fast-Medium" 
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