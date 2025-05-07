import os
import pretty_midi
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import pygame


def cut_midi(midi_data, start_time, end_time):
    """
    截取 MIDI 文件的指定时间段。
    """
    cut_midi = pretty_midi.PrettyMIDI()
    for instrument in midi_data.instruments:
        new_instrument = pretty_midi.Instrument(program=instrument.program)
        for note in instrument.notes:
            if note.start >= start_time and note.end <= end_time:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start - start_time,
                    end=note.end - start_time
                    # duration=note.end - note.start,
                )
                new_instrument.notes.append(new_note)
        if len(new_instrument.notes) > 0:
            cut_midi.instruments.append(new_instrument)
    return cut_midi


def midi2img(midi_data, min_duration_unit, pad=6, shape=(88, 88)):
    """
    将 MIDI 文件转换为图片。
    """
    # 计算每个音符的持续时间
    notes_vec  = []
    # 创建一个空的图像
    img = np.zeros(shape, dtype=np.float32)

    # 遍历 MIDI 文件中的音符
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            nstart = note.start
            pstart = int(nstart / min_duration_unit)
            nend = note.end
            pend = int(nend / min_duration_unit)
            if note.pitch-21 < 0 or note.pitch-21 > shape[0]-1:
                continue
            if pend > shape[1]:
                pend = shape[1]
            # img[note.pitch-21, pstart:pend] = note.velocity / 127.0
            img[note.pitch-21, pstart:pend] = 1
            
    # padding
    if pad > 0:
        img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
        
    img = img.clip(0,1)

    return img

def img2midi(img, min_duration_unit, pad=12):
    """
    将图片转换为 MIDI 文件。
    """
    # 创建一个新的 MIDI 对象
    midi_data = pretty_midi.PrettyMIDI()
    # 创建一个新的乐器对象
    instrument = pretty_midi.Instrument(program=0)

    # 遍历图像中的每个音符
    for i in range(img.shape[0]):
        lst_vel = 0
        lst_st = 0
        for j in range(img.shape[1]):
            if img[i, j] == lst_vel:
                continue
            if lst_vel > 0 :
                pitch = i + 21 - pad
                if pitch < 21 or pitch > 108:
                    pass
                if lst_st - pad < 0:
                    lst_st = pad
                if j - pad < 0:
                    j = pad
                note = pretty_midi.Note(
                    velocity=int(lst_vel * 127/2),
                    pitch=i + 21 - pad,
                    start=(lst_st - pad) * min_duration_unit,
                    end=(j - pad) * min_duration_unit
                )
                instrument.notes.append(note)
            lst_vel = img[i, j]
            lst_st = j
        if lst_vel > 0:
            note = pretty_midi.Note(
                velocity=int(lst_vel * 127),
                pitch=i + 21 - pad,
                start=(lst_st - pad) * min_duration_unit,
                end=(img.shape[1] - pad) * min_duration_unit
            )
            instrument.notes.append(note)

    # 将乐器添加到 MIDI 对象中
    midi_data.instruments.append(instrument)
    return midi_data

def filter_imgs(imgs, sum_interval=[200, 1000]):
    """
    过滤图片，保留像素和在指定范围内的图片。
    """
    filtered_imgs = []
    for img in imgs:
        if (img.sum()) > sum_interval[0] and (img.sum()) < sum_interval[1]:
            # print("img sum:", img.sum())
            filtered_imgs.append(img)
    return filtered_imgs

def play_midi(file):
   freq = 44100
   bitsize = -16
   channels = 2
   buffer = 1024
   pygame.mixer.init(freq, bitsize, channels, buffer)
   pygame.mixer.music.set_volume(1)
   clock = pygame.time.Clock()
   try:
       pygame.mixer.music.load(file)
   except:
       import traceback
       print(traceback.format_exc())
   pygame.mixer.music.play()
   while pygame.mixer.music.get_busy():
       clock.tick(30)



def test():
    midi_paths = "midis"
    files = os.listdir(midi_paths)
    midi_path = os.path.join(midi_paths, files[2])
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    # 截取 MIDI 文件的某 10 秒
    st_second = 10
    cut_midi_data = cut_midi(midi_data, st_second, st_second+10)

    # 将 MIDI 文件转换为图片
    img = midi2img(cut_midi_data, min_duration_unit=0.1136, pad=6, shape=(88, 88))
    
    print("img shape:", img.shape)

    # 将图片转换为 MIDI 文件
    midi_data2 = img2midi(img, min_duration_unit=0.1136, pad=6)
    
    # ti play the MIDI file
    midi_data2.write("output.mid")
    
    img2 = midi2img(midi_data2, min_duration_unit=0.1136, pad=6, shape=(88, 88))
    # 显示图片
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title("Reconstructed Image")
    # plt.show()
    plt.savefig("midi2img.png")

if __name__ == "__main__":
    path = "midis"
    midi_datas = []
    cnt = 0
    for file in os.listdir(path):
        if file.endswith(".mid"):
            cnt+=1 
            print(cnt, file)
            if cnt > 200:
                break
            midi_path = os.path.join(path, file)
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            midi_datas.append(midi_data)
    print("MIDI files loaded:", len(midi_datas))
    
    midi_datas = midi_datas[:3]
    
    raw_imgs = []
    
    for midi_data in midi_datas:
        for st_seconds in range(0, 10, 5):
            cut_midi_data = cut_midi(midi_data, st_seconds, st_seconds+10)
            img = midi2img(cut_midi_data, min_duration_unit=0.1136, pad=6, shape=(88, 88))
            raw_imgs.append(img)
    print("Raw images:", len(raw_imgs))
    print("Raw images shape:", raw_imgs[0].shape)
    # filter images
    filtered_imgs = filter_imgs(raw_imgs, sum_interval=[200, 1000])
    # filtered_imgs = raw_imgs
    print("Filtered images:", len(filtered_imgs))
    
    # save the filtered images to a file
    print("Saving filtered images to file...")
    np.save('filtered_midi_imgs.npy', filtered_imgs)
    print("Saved.")
    
    # visualize the images
    sample = filtered_imgs[:25]
    sample = np.array(sample)
    sample = sample.reshape(-1, 100, 100)
    print(sample.shape)
    for i in range(sample.shape[0]):
        # print(f"max: {sample[i].max()}, min: {sample[i].min()}, sum: {sample[i].sum()}")
        plt.subplot(5, 5, i+1)
        plt.imshow(sample[i], cmap='gray')
        plt.title(f"Sum : {int(sample[i].sum())}")
        plt.axis('off')
    # plt.tight_layout()
    plt.show()
    plt.savefig("midi_sample.png")
    
    