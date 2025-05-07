# midiDraw

## Dataset
- Check what midi is like, design how it should be changed to img. 
- Midi dataset: [https://github.com/bytedance/GiantMIDI-Piano](GiantMIDI-Piano), [https://magenta.tensorflow.org/datasets/maestro](MAESTRO), 
- Img dataset, try QuickDraw or just MNIST.
- Write a `midi2img.py` to convert midi to img, and `img2midi.py` to convert img to midi. **SHOULD HAVE GRADIENTS**.

## LOSS 
- For img, should tolerant scale and shift. We can train another network on augmented data to do this.
- For midi, shouldtolerant chords and some midi shift, also can train another network on augmented data to do this. But better use a pretrained model to do this.

## Model
- Diffusion. No idea but promising.
- VAE. img+midi -> z -> midi. Compute loss.
- GAN. noise -> midi. D1: is midi a music? D2: is midi an img? Most simple to implement!

