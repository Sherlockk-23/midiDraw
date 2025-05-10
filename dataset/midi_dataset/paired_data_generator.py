import os
import pretty_midi
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import pygame


import midi2img

def pitch_shift(midi_data):
    """
    将 MIDI 文件中的音符音高移动指定的音高值。
    """
    min_pitch = 120
    max_pitch = 0
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            if note.pitch < min_pitch:
                min_pitch = note.pitch
            if note.pitch > max_pitch:
                max_pitch = note.pitch
    pitch_shift = random.randint(-12, 12)
    pitch_shift = min(pitch_shift, 108 - max_pitch)
    pitch_shift = max(pitch_shift, 21 - min_pitch)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.pitch += pitch_shift
    return midi_data

def duration_shrink(midi_data):
    """
    将 MIDI 文件中的音符st, ed全部乘2
    """
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.start *= 2
            note.end *= 2
    return midi_data

def duration_expand(midi_data):
    """
    将 MIDI 文件中的音符st, ed全部除2, 并复制一份
    """
    time_shift = midi_data.get_end_time() / 2
    new_notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.start /= 2
            note.end /= 2
            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start + time_shift,
                end=note.end + time_shift
            )
            new_notes.append(new_note)
        instrument.notes.extend(new_notes)
            
    return midi_data

def duration_scale(midi_data, scale=None):
    """
    将 MIDI 文件中的音符st, ed全部乘以scale
    """
    if scale is None:
        scale = random.uniform(0.7, 1.5)
    time_shift = midi_data.get_end_time() * scale
    new_notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.start *= scale
            note.end *= scale
            if scale < 1:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start + time_shift,
                    end=note.end + time_shift
                )
                new_notes.append(new_note)
    return midi_data

def musical_augmentation(midi_data):
    """
    对 MIDI 文件进行音乐增强，对每一个note进行dropout, 上下八度变换
    """
    new_notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            choice = random.random()
            if choice < 0.4:  # dropout
                # note.velocity = 0
                drop_prob = (note.pitch - 50)**2
                drop_prob = np.exp(-drop_prob / 500)
                print("drop_prob:", drop_prob)
                if random.random() < drop_prob:
                    note.velocity = 0
            elif choice < 0.6:  # 上下八度变换
                if note.pitch+12 > 108:
                    continue
                if random.random() < 0.3:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end
                    )
                    new_notes.append(new_note)
                note.pitch += 12    
            elif choice < 0.8:  # 上下八度变换
                if note.pitch-12 < 21:
                    continue
                if random.random() < 0.3:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end
                    )
                    new_notes.append(new_note)
                note.pitch -= 12               
        instrument.notes.extend(new_notes)
    return midi_data
    

def gen_new_midi(midi_data, aug_times):
    """
    augment a midi file, that reserves the musical structure, but changes the pitch and duration of the notes
    """
    new_midi_data = midi_data
    
    new_midi_data = pitch_shift(new_midi_data)
    
    if random.random() < 0.5:
        new_midi_data = duration_scale(new_midi_data)
        
    for i in range(aug_times):
        new_midi_data = musical_augmentation(new_midi_data)
        
    
    
    return new_midi_data

def gen_new_img(img):
    midi = midi2img.img2midi(img)
    new_midi = gen_new_midi(midi, 2)
    new_img = midi2img.midi2img(new_midi)
    return new_img
    

def test_augmentation(midi_data, idx=0):
    """
    test the augmentation
    """
    midi_data.write(f"./test/midi_{idx}.mid")
    ori_img = midi2img.midi2img(midi_data)
    aug_midi_data = gen_new_midi(midi_data, 2)
    aug_img = midi2img.midi2img(aug_midi_data)
    
    # midi2img.play_midi(midi_data)
    # midi2img.play_midi(aug_midi_data)
    
    # save_path = "augmented_midi.mid"
    
    aug_midi_data.write(f"./test/augmented_midi_{idx}.mid")
    
    return ori_img, aug_img


if __name__ == "__main__":
    path = "midis"
    midi_datas = []
    cnt = 0
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.midi') or f.endswith('.mid')]
    # print("Total MIDI files:", len(files))
    for file in files:
        cnt+=1 
        # if cnt > 10:
        #     break
        print(cnt, file)
        midi_path = os.path.join(path, file)
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        midi_datas.append(midi_data)
    
    raw_imgs = []
    
    for midi_data in midi_datas:
        end_time = int(midi_data.get_end_time())
        for st_seconds in range(5, end_time-10, 10):
            cut_midi_data = midi2img.cut_midi(midi_data, st_seconds, st_seconds+10)
            img = midi2img.midi2img(cut_midi_data, min_duration_unit=0.1136, pad=6, shape=(88, 88))
            raw_imgs.append(img)
    num_datas = 10_000
    raw_imgs = raw_imgs[:5*num_datas]        
    
    print("Raw images:", len(raw_imgs))
    print("Raw images shape:", raw_imgs[0].shape)
    # filter images
    filtered_imgs = midi2img.filter_imgs(raw_imgs, sum_interval=[200, 1000], min_pitches=15)
    # filtered_imgs = raw_imgs
    print("Filtered images:", len(filtered_imgs))    
    
    # plt.figure(figsize=(10, 5))
    
    
    print("augmenting paired images...")
    
    paired_imgs = []
    
    for i, img in enumerate(filtered_imgs):
        pass
        # TODO
    
    
    
    
    
    
    
    
    