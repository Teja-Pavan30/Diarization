
import librosa 
import matplotlib.pyplot as plt
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def NLP(signals, sr_in=8000):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
    )

    results = {}
    for i, sig in enumerate(signals, 1):
        # Sepformer outputs 8 kHz, Whisper wants 16 kHz
        audio = librosa.resample(sig, orig_sr=sr_in, target_sr=16000)
        out = pipe({"array": audio, "sampling_rate": 16000},
                   generate_kwargs={"language": "en"})
        results[f"speaker{i}"] = out["text"]
        print(f"speaker{i}:", out["text"])
    return results


