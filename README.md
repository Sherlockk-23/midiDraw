# midiDraw

## Dataset
- [Done] Check what midi is like, design how it should be changed to img. 
- [Done] Midi dataset: [https://github.com/bytedance/GiantMIDI-Piano](GiantMIDI-Piano), [https://magenta.tensorflow.org/datasets/maestro](MAESTRO), 
- [Done] Img dataset, try QuickDraw or just MNIST.
- [thought: why not just use img as normalized midi to compute loss? ] Write a `midi2img.py` to convert midi to img, and `img2midi.py` to convert img to midi. **SHOULD HAVE GRADIENTS**.

## LOSS 
- For img, should tolerant scale and shift. Also need Gaussian on midi-generated imgs to fillin strokes. We can train another network on augmented data to do this.
- For midi, should tolerant chords and some midi shift, also can train another network on augmented data to do this. But better use a pretrained model to do this.

- UPD: 5/8 . vae直接取CXEloss感觉不太对，会出来很糊的音乐。最好还是借一个类似于music loss的东西。

## Model
- Diffusion. No idea but promising.
- VAE. img+midi -> z -> midi. Compute loss.
- GAN. noise -> midi. D1: is midi a music? D2: is midi an img? Most simple to implement!
- RNN. Every column as a token. make some sense. 

Thoughts: 人应该是，每一列单独，根据和弦转化成最近的东西。加音或者减音，都是和弦内音。

hardcode怎么写？最小单位，1~n, 把和弦内的音保留下来，其它的丢掉。试一试吧。
或者，先用和弦内音铺满，再丢掉一些

我草，这个绝对是有用的。有数据集了！

以及，旋律可以小噪点。

那么音乐数据集不太行，最好是游戏音乐之类的，并且拍子能卡下来。

- TODO 1 写出一个hardcode再说
- TODO 2 对比学习，改 loss？思路不一定对。