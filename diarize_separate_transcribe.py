#!/usr/bin/env python3
"""
End-to-end: 4-speaker overlap file → diarization → separation → transcripts
CPU only, ~2×RT on a 4-core laptop for a 5-min file.
Usage:
    python diarize_separate_transcribe.py long_overlap.wav

"""
import json, pathlib, sys, typing as tp
import torch, torchaudio
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.utils.signal import Binarize
import os
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

DEVICE = "cpu"
MODEL_FS = 16000  # pyannote & whisper both use 16 kHz



def diarize(wav_path: pathlib.Path):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN")   
    ).to(torch.device("cpu"))
    dz = pipeline(wav_path, num_speakers=4)   
    turns = [(turn.start, turn.end, speaker)
               for turn, _, speaker in dz.itertracks(yield_label=True)]
    return sorted(turns, key=lambda x: x[0])

def separate_and_transcribe(
    wav_path: pathlib.Path,
    turns: tp.List[tp.Tuple[float, float, str]],
    out_dir: pathlib.Path,
):
    waveform, fs = torchaudio.load(wav_path)
    if fs != MODEL_FS:
        waveform = torchaudio.functional.resample(waveform, fs, MODEL_FS)
    model = whisper.load_model("base", device=DEVICE)  
    results = []
    for idx, (start, end, spk) in enumerate(turns):
        seg = waveform[:, int(start*MODEL_FS):int(end*MODEL_FS)]
        track_path = out_dir / f"{wav_path.stem}_{spk}_seg{idx:03d}.wav"
        torchaudio.save(track_path, seg, MODEL_FS)
        txt = model.transcribe(seg.squeeze().numpy(), language="en", fp16=False)["text"].strip()
        results.append({"start": round(start, 2),
                        "end": round(end, 2),
                        "speaker": spk,
                        "text": txt})
    return results

def plot_diarization(turns: list, wav_path: pathlib.Path, save_path: pathlib.Path = None):
    """Simple timeline plot of speaker turns as dots."""
    speakers = sorted({spk for _, _, spk in turns})
    cmap = plt.cm.get_cmap("tab10", len(speakers))  
    spk_color = {spk: cmap(i) for i, spk in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(10, 2))
    for start, end, spk in turns:
        ax.plot([start, end], [spk, spk], 'o-', color=spk_color[spk], markersize=5, linewidth=0.5)
    ax.set_xlim(0, max(end for _, end, _ in turns))
    ax.set_ylim(-1, len(speakers))
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("time (s)")
    ax.set_title(f"Speaker timeline – {wav_path.name}")
    ax.grid(True)
    plt.tight_layout()
    out_png = save_path or wav_path.with_suffix("") / (wav_path.stem + "_timeline.png")
    plt.savefig(out_png, dpi=150)
    plt.show()  
    plt.close()
    print(f"timeline plot saved → {out_png}")

def main():
    if len(sys.argv) != 2:
        print("usage: python diarize_separate_transcribe.py <wav>")
        sys.exit(1)
    wav = pathlib.Path(sys.argv[1]).expanduser()
    out = wav.with_suffix("")
    out.mkdir(exist_ok=True)
    print("1. Diarizing…")
    turns = diarize(wav)
    plot_diarization(turns, wav)  
    with open(out / (wav.stem + ".rttm"), "w") as f:
        for s, e, spk in turns:
            f.write(f"SPEAKER {wav.stem} 1 {s:.3f} {e-s:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    print("2. Separating & transcribing…")
    js = separate_and_transcribe(wav, turns, out)
    with open(out / (wav.stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)
    print("done →", out)

if __name__ == "__main__":
    main()


