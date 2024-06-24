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

    audio, fs = torchaudio.load(f"./{file_name}")
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
        print("Audio is already mono")
        audio_mono = audio
    elif audio.shape[0] != 2 and audio.shape[1] != 2: 
        raise Exception("Non valid vector")
    else:
        #features
        audio_mono = (audio[:,0]/2)+(audio[:,1]/2)
    return audio_mono

def resample_signal_fs(in_signal, original_sr, target_sr, output_format='numpy'):
    """
    Resamples a signal to a target sampling rate using torchaudio.

    Parameters:
        signal (torch.Tensor or np.ndarray): The input signal.
        original_sr (int): The original sampling rate of the input signal.
        target_sr (int): The target sampling rate for resampling.
        output_format (str, optional): The desired output format ('numpy' or 'torch').Defaults to 'numpy'.
        
    Returns:
        resampled_signal: The resampled signal in the specified format.
    """
    # Convert the input signal to a torch tensor if it's a NumPy array
    if not isinstance(original_sr, int) or not isinstance(target_sr, int):
        raise TypeError("Los parámetros original_sr y target_sr deben ser enteros.")

    if isinstance(in_signal, np.ndarray):
        in_signal = in_signal.astype(np.float32)
        in_signal = torch.from_numpy(in_signal)

    # Resample the signal
    if original_sr == target_sr:
        resampled_signal = in_signal
        print("Las frecuencias de sampleo son iguales, no es necesario resamplear")
    else:
        resampled_signal = torchaudio.transforms.Resample(original_sr, target_sr)(in_signal)
        print(f"Señal resampleada de {original_sr} Hz a {target_sr} Hz")

    # Convert the resampled signal to the desired output format
    if output_format == 'numpy':
        resampled_signal = resampled_signal.numpy().astype(np.float32)
    elif output_format == 'torch':
        resampled_signal = resampled_signal.type(torch.float32)

    return resampled_signal