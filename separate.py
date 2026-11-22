import torchaudio
import torch
import speechbrain as sb
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy
import os
import numpy as np
import torch
import librosa
from scipy.io.wavfile import write
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.utils.fetching import LocalStrategy


def separate(input_path="NLP_stuff/MiniLibriMix/train/mix_clean/17-362-0014_2999-156967-0015.wav"):
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir="NLP_stuff",
        local_strategy=LocalStrategy.COPY,
    )

    audio, sr = librosa.load(input_path, sr=8000, mono=True)
    audio = torch.from_numpy(audio).float().unsqueeze(0)

    est_sources = model.separate_batch(audio)

    signals = [est_sources[0, :, i].detach().cpu().numpy() for i in range(2)]

   
    for i, sig in enumerate(signals, 1):
        write(f"NLP_stuff/speaker{i}.wav", 8000,
              np.int16(np.clip(sig, -1, 1) * 32767))
        
    print("Both separated audio files saved!")

    return signals

    

