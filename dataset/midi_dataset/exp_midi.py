import os
import pretty_midi
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import pygame


path = "midis/example2.mid"
midi_data = pretty_midi.PrettyMIDI(path)

# print(files[0])

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

print(midi_data.get_beats())
print(midi_data.estimate_tempo())
# print(midi_data.estimate_tempi())
print(midi_data.get_end_time())
# print(midi_data.get_chroma())
chroma = midi_data.get_chroma()
plt.imshow(chroma, aspect='auto', origin='lower', cmap='gray_r')
plt.colorbar()
plt.title('Chroma Feature')
plt.xlabel('Time (s)')
plt.ylabel('Pitch Class')
plt.show()
plt.savefig('chroma.png')
print(midi_data.get_tempo_changes())
print(midi_data.estimate_beat_start())
print(midi_data.get_piano_roll())
print(midi_data.get_downbeats())
# print(midi_data.adjust_times(0,2))
# print(get_object_info(midi_data))

