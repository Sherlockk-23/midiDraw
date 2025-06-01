import os
import sys
import numpy as np
import wave

from GiantMusicTransformer.midi_to_colab_audio import midi_to_colab_audio
from GiantMusicTransformer import TMIDIX

    
def midi2wav(midi_path, output_path):
    """
    Convert MIDI file to WAV file.
    
    Args:
        midi_path (str): Path to the input MIDI file.
        output_path (str): Path to save the output WAV file.
    """
    midi_paths = os.listdir(midi_path)
    for midi_file in midi_paths:
        if not midi_file.endswith('.mid'):
            continue
        full_midi_path = os.path.join(midi_path, midi_file)
        try:
            midi_audio = midi_to_colab_audio(full_midi_path)
            r_audio = midi_audio.T
            r_audio = np.int16(r_audio / np.max(np.abs(r_audio)) * 32767)

            output_wav_path = os.path.join(output_path, f"{os.path.splitext(midi_file)[0]}.wav")
            with wave.open(output_wav_path, 'w') as wf:
                wf.setframerate(16000)
                wf.setsampwidth(2)
                wf.setnchannels(r_audio.shape[1])
                wf.writeframes(r_audio)
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gen_wavs.py <midi_path> <output_path>")
        sys.exit(1)

    midi_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    midi2wav(midi_path, output_path)
    print(f"Converted MIDI files from {midi_path} to WAV files in {output_path}.")