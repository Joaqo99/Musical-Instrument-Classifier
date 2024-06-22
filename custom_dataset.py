from torch.utils.data import Dataset
import audio_functions as auf
import read_data as rd
import os

"""
Audio metadata consists of 2 columns:
FileName | Class
For this model we create the sample labels.

Testing and training data are splitted and all the audios are in the same folder.
"""


class MusicalInstrumentsDataset(Dataset):

    def __init__(self, annotations_file, audio_dir):
        self.annotations = rd.read_file(annotations_file)
        self.audio_dir = audio_dir
        self.classes = self.annotations["Class"].unique()   #gets audio classes
        self.labels = range(1, len(self.classes) + 2)       #create labels from audio classes

    def __len__(self):
        """Returns number of samples in dataset"""
        return len(self.annotations)

    def __getitem__(self, index):
        """Returns item form x index"""
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        audio_signal, sr = auf.load_audio(audio_sample_path, output_format="torch_tensor")
        return audio_signal, label
    
    def _get_audio_sample_path(self, index):
        """Returns audio file path from x index"""
        sample_filename = self.annotations.iloc[index, "FileName"]
        path = os.path.join(self.audio_dir, sample_filename)
        return path
    
    def get_audio_sample_label(self, index):
        """Returns audio label from x index"""
        sample_class = self.annotations.iloc[index, "Class"]
        sample_label = self.labels[sample_class + 1]
        return sample_label

