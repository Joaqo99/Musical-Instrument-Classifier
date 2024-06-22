import soundfile as sf
from IPython.display import Audio
import numpy as np
from scipy import signal
import torchaudio
import torch
from torchaudio import transforms
import numpy as np

def load_audio(file_name, output_format="numpy"):
    """
    Loads a mono or stereo audio file in audios folder.
    Input:
        - file_name: str type object. The file must be an audio file.
        - output_format: str type object. The desired vector output format ('numpy' or 'torch_tensor'). Defaults to 'numpy'.
    Output:
        - audio: array type object.
        - fs: sample frequency
        - prints if audio is mono or stereo.
    """
    if type(file_name) != str:
        raise Exception("file_name must be a string")

    audio, fs = torchaudio.load(f"./audios/{file_name}")
    audio = audio.squeeze()

    # Converts the resampled signal to the desired output format
    if output_format == 'numpy':
        audio = audio.numpy().astype(np.float32)
    elif output_format == 'torch_tensor':
        audio = audio.type(torch.float32)

    return audio, fs

def play_audio(audio, fs):
    """
    Plays a mono audio
    Inputs:
        - audio: array type object. Audio to play. Must be mono.
        - fs: int type object. Sample rate
    """
    #error handling
    if type(fs) != int:
        raise ValueError("fs must be int")

    return Audio(audio, rate=fs)

def to_mono(audio):
    """
    Converts a stereo audio vector to mono.
    Insert:
        - audio: array type object of 2 rows. Audio to convert.
    Output:
        - audio_mono: audio converted
    """
    #error handling



    if  type(audio) != np.ndarray and type(audio) != torch.Tensor:
        raise ValueError("audio must be a vector")
    if len(audio.shape) == 1:
        raise Exception("Audio is already mono")
    elif audio.shape[0] != 2 and audio.shape[1] != 2: 
        raise Exception("Non valid vector")
    
    #features
    audio_mono = (audio[:,0]/2)+(audio[:,1]/2)
    return audio_mono
