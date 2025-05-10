# midiDraw

灵感来源：[https://www.bilibili.com/video/BV1mr4y1C7Fq](https://www.bilibili.com/video/BV1mr4y1C7Fq)

## Dataset
- [Done] Check what midi is like, design how it should be changed to img. See [json_midi_example](./dataset/midi_dataset/midi_exmple.json) and ![midisample](./figures/midi_sample.png).
- [Done] Midi dataset: [https://github.com/bytedance/GiantMIDI-Piano](GiantMIDI-Piano), [https://magenta.tensorflow.org/datasets/maestro](MAESTRO), [https://github.com/asigalov61/Tegridy-MIDI-Dataset?tab=readme-ov-file](TBD)
- [Done] Img dataset, try QuickDraw or just MNIST. See ![imgsample](./figures/quickdraw_sample.png).
- [Done] Write a `midi2img.py` to convert midi to img, and `img2midi.py` to convert img to midi. 

- [TODO] Find a better dataset instead of Classical music! 古典不好听！我要游戏音乐！or anything else



## LOSS 
- For img, should tolerant scale and shift. Also need Gaussian on midi-generated imgs to fillin strokes. We can train another network on augmented data to do this.
- For midi, should tolerant chords and some midi shift, also can train another network on augmented data to do this. But better use a pretrained model to do this.

- UPD: 5/8 . vae直接取CXEloss感觉不太对，会出来很糊的音乐。最好还是借一个类似于music loss的东西。

## Music Generation Model
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

- [Done] TODO 1 写出一个hardcode再说
- TODO 2 对比学习，改 loss。额外训练一个模型，给两段音乐，判断相似度。思路不一定对。


## Another idea: 
音乐生成不好搞的话，重点做图片生成，用图片匹配midi
1. 最后取midi被图片mask以后剩下的东西
2. 训练图片生成器，加一个loss，在音乐里面采样，能和音乐匹配
3. loss 长得比较像 $\alpha_1*(img - maskedMidi) + \alpha_2(midi - maskedImg) + originalLoss$

灵感来源于图片：![output.png](./figures//output.png)
