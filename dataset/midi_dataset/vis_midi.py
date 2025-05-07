import os
import pretty_midi
import json
import numpy as np
import matplotlib.pyplot as plt
import random

"""
instrument: Only one instrument in this dataset.
pitch: 21-108, with a range of 88 keys.
velocity: 0-127, Meaning Force of the note.
duration: 0.05-5.0, drop those out of range. 

So, if set to 88*88, 10 sec, min duration unit = 10/88 = 0.1136s

Then expand to 100*100, pad with zero.

"""

def get_object_info(obj, indent=0):
    """
    递归获取对象的所有属性和值，并以树状结构返回。
    """
    obj_info = {}
    for attr in dir(obj):
        if not attr.startswith('__'):
            value = getattr(obj, attr)
            if isinstance(value, list):
                obj_info[attr] = [get_object_info(item, indent + 2) for item in value]
            elif hasattr(value, '__dict__'):
                obj_info[attr] = get_object_info(value, indent + 2)
            else:
                obj_info[attr] = value
    return obj_info

def get_notes(midi_data):
    """
    获取 MIDI 文件中的音符信息。
    """
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append({
                'duration': note.duration,
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'velocity': note.velocity,
                'instrument': instrument.name
            })
    return notes

duration_bins = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

def get_bin_index(value, bins):
    """
    获取值在指定区间内的索引。
    """
    if value < bins[0]:
        return 0
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i+1
    return len(bins) 

# 设置路径和文件
path = "midis"
files = os.listdir(path)
files = [f for f in files if f.endswith('.midi') or f.endswith('.mid')]
num_samples = 300

all_notes = []
random.shuffle(files)
files = files[:num_samples] if len(files) > num_samples else files

cnt = 0
for file in files:
    cnt += 1
    if cnt % 10 == 0:
        print(f"Processing {cnt}/{len(files)}: {file}")
    file_path = os.path.join(path, file)
    midi_data = pretty_midi.PrettyMIDI(file_path)
    notes = get_notes(midi_data)
    all_notes.extend(notes)
    
print("Total notes extracted:", len(all_notes))
    
pitch_count = {}
for note in all_notes:
    pitch = note['pitch']
    if pitch not in pitch_count:
        pitch_count[pitch] = 0
    pitch_count[pitch] += 1
    
velocity_count = {}
for note in all_notes:
    velocity = note['velocity']
    if velocity not in velocity_count:
        velocity_count[velocity] = 0
    velocity_count[velocity] += 1
    
duration_count = {}
for note in all_notes:
    duration = note['duration']
    bin_index = get_bin_index(duration, duration_bins)
    if bin_index not in duration_count:
        duration_count[bin_index] = 0
    duration_count[bin_index] += 1    

# visualize the duration distribution
plt.figure(figsize=(10, 6))
# plt.bar(range(len(duration_count)), list(duration_count.values()), tick_label=[f"{duration_bins[i]}-{duration_bins[i+1]}" for i in range(len(duration_bins)-1)])
# should have two extra bins for the first and last one

plt.bar(range(len(duration_count)), list(duration_count.values()), tick_label=[f"<{duration_bins[0]}"]+[f"{duration_bins[i]}-{duration_bins[i+1]}" for i in range(len(duration_bins)-1)] + [f">{duration_bins[-1]}"])

plt.xlabel('Duration (s)')
plt.ylabel('Count')
plt.title('Duration Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("another_duration_distribution.png")
plt.show()
