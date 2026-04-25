import numpy as np
import torch


def convert_audio(wav, sr):
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    if wav.shape[0] > wav.shape[1]:
        wav = wav.transpose(0, 1)

    wav = wav.unsqueeze(0)
    return {"waveform": wav, "sample_rate": sr}


def load_audio_input(audio_input):
    if audio_input is None:
        return None

    waveform = audio_input["waveform"]
    sr = audio_input["sample_rate"]
    print(f"load_audio_input received: waveform.shape={waveform.shape}, waveform.dim={waveform.dim()}")

    if waveform.dim() == 0:
        raise ValueError("Waveform is 0-dimensional (scalar tensor). This causes 'Len() of unsized object' error.")
    if waveform.dim() == 1:
        wav = waveform
        print("Format [samples], returning as-is")
    elif waveform.dim() == 2:
        if waveform.shape[0] < waveform.shape[1]:
            wav = torch.mean(waveform, dim=0) if waveform.shape[0] > 1 else waveform[0]
            print(f"Format [C, T], mixed to mono: {wav.shape}")
        else:
            wav = torch.mean(waveform, dim=1) if waveform.shape[1] > 1 else waveform[:, 0]
            wav = wav.unsqueeze(0)
            print(f"Format [T, C], converted: {wav.shape}")
    elif waveform.dim() == 3:
        wav = waveform[0]
        wav = torch.mean(wav, dim=0) if wav.shape[0] > 1 else wav.squeeze(0)
        print(f"Format [B, C, T], took first batch and mixed: {wav.shape}")
    else:
        raise ValueError(f"Unexpected waveform dimension: {waveform.dim()} (shape: {waveform.shape})")

    if wav.dim() > 1:
        wav = wav.squeeze()
        print(f"Squeezed to 1D: {wav.shape}")

    return (wav.numpy(), sr)
