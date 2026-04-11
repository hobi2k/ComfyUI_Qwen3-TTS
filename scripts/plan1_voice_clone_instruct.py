"""
plan1_voice_clone_instruct.py
Approach A: VoiceClone (Base model) + emotional reference audio — no fine-tuning.
Approach B: CustomVoice model + instruct — no fine-tuning.

Usage
-----
python scripts/plan1_voice_clone_instruct.py \\
    --ref_dir   /path/to/emotional_segments \\
    --output_dir /path/to/outputs \\
    --phrases_file /path/to/phrases.txt \\
    [--instruct "Desperate female voice..."] \\
    [--temperature 1.05] [--top_p 0.95] [--num_seeds 15] \\
    [--leading_silence 5.65] \\
    [--pause_durations 0.32,0.74,0.97,0.91,1.80,1.56] \\
    [--target_duration 36.20] \\
    [--skip_a] [--skip_b]

Each ref segment directory must contain sample_NN.wav + sample_NN.txt pairs.
phrases.txt: one synthesis phrase per line.
If --phrases_file is omitted, a single phrase from --text is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INSTRUCT = (
    "Desperate female voice pleading and begging under extreme duress. "
    "Breathing is heavy and strained throughout, audible between every word. "
    "Voice breaks and trembles from anguish — not crying but suffering. "
    "Each word is forced out with effort, pitch rising desperately. "
    "Anguished moaning quality — raw and strained, not soft. "
    "Speak very slowly, voice cracking under desperation."
)

BASE_REPO         = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CUSTOM_VOICE_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
TRIM_TOP_DB       = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def trim_trailing_silence(wav_path: Path, top_db: float = TRIM_TOP_DB, buffer_s: float = 0.25) -> float:
    audio, sr = librosa.load(str(wav_path), sr=None, mono=True)
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=256)
    if len(intervals) == 0:
        return len(audio) / sr
    last_end = int(intervals[-1][1])
    trim_end = min(last_end + int(buffer_s * sr), len(audio))
    sf.write(str(wav_path), audio[:trim_end], sr)
    return trim_end / sr


def assemble(
    segments: list[np.ndarray],
    sr: int,
    pause_durations: list[float],
    leading_silence: float,
    target_duration: float,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        parts.append(seg)
        pause_s = pause_durations[i] if i < len(pause_durations) else 1.0
        pf = int(round(pause_s * sr))
        ch = seg.shape[1] if seg.ndim > 1 else 1
        sil = np.zeros((pf, ch), dtype=np.float32) if seg.ndim > 1 else np.zeros(pf, dtype=np.float32)
        parts.append(sil)

    speech = np.concatenate(parts, axis=0)

    if leading_silence > 0:
        lf = int(round(leading_silence * sr))
        ch = speech.shape[1] if speech.ndim > 1 else 1
        lead = np.zeros((lf, ch), dtype=np.float32) if speech.ndim > 1 else np.zeros(lf, dtype=np.float32)
        speech = np.concatenate([lead, speech], axis=0)

    tf = int(round(target_duration * sr))
    if len(speech) > tf:
        speech = speech[:tf]
    elif len(speech) < tf:
        gap = tf - len(speech)
        ch = speech.shape[1] if speech.ndim > 1 else 1
        pad = np.zeros((gap, ch), dtype=speech.dtype) if speech.ndim > 1 else np.zeros(gap, dtype=speech.dtype)
        speech = np.concatenate([speech, pad], axis=0)
    return speech


def load_model(repo_id: str) -> object:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(repo_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa")


def best_take(
    generate_fn,
    phrase: str,
    phrase_idx: int,
    num_seeds: int,
    seed_base: int,
    max_new_tokens: int,
    max_safe_dur: float,
    tmp_dir: Path,
    tag: str,
) -> tuple[np.ndarray, int, float, int]:
    best_audio: np.ndarray | None = None
    best_sr    = 24000
    best_dur   = 0.0
    best_seed  = seed_base

    for i in range(num_seeds):
        seed = seed_base + phrase_idx * 100 + i
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        wavs, sr = generate_fn(seed, phrase, max_new_tokens)
        speech = wavs[0]
        if speech.ndim == 1:
            speech = speech[:, np.newaxis]

        tmp = tmp_dir / f"_tmp_{tag}_{phrase_idx}_s{i}.wav"
        sf.write(str(tmp), speech, sr)
        dur = trim_trailing_silence(tmp)
        trimmed, _ = sf.read(str(tmp), always_2d=True)
        tmp.unlink(missing_ok=True)

        flag = " [SKIP: loop]" if dur >= max_safe_dur else ""
        log(f"    seed {seed}: {dur:.2f}s{flag}")

        if dur >= max_safe_dur:
            continue
        if dur > best_dur:
            best_dur  = dur
            best_audio = trimmed
            best_sr   = sr
            best_seed  = seed

    if best_audio is None:
        best_audio = np.zeros((int(0.5 * best_sr), 1), dtype=np.float32)
        best_dur   = 0.5

    log(f"  → seed {best_seed}, {best_dur:.2f}s")
    return best_audio, best_sr, best_dur, best_seed


# ---------------------------------------------------------------------------
# Approach A — VoiceClone
# ---------------------------------------------------------------------------

def run_voice_clone(
    model,
    phrases: list[str],
    ref_dir: Path,
    output_dir: Path,
    args,
) -> Path:
    log("\n" + "=" * 60)
    log("Approach A: VoiceClone (Base model, ICL mode)")
    log("=" * 60)

    seg_files = sorted(ref_dir.glob("sample_*.wav"))
    if not seg_files:
        raise FileNotFoundError(f"No sample_*.wav in {ref_dir}")

    ref_dur = {f: librosa.get_duration(path=str(f)) for f in seg_files}
    sorted_segs = sorted(seg_files, key=lambda f: ref_dur[f], reverse=True)
    max_safe = args.max_new_tokens / 12 * 0.85

    segments: list[np.ndarray] = []
    synth_info: list[dict]     = []
    output_sr = 24000

    for idx, phrase in enumerate(phrases):
        log(f"\n[{idx+1}/{len(phrases)}] '{phrase}'")
        ref_path = sorted_segs[idx % len(sorted_segs)]
        ref_np, ref_sr = librosa.load(str(ref_path), sr=None, mono=True)
        ref_txt_path = ref_path.with_suffix(".txt")
        ref_text = ref_txt_path.read_text(encoding="utf-8").strip() if ref_txt_path.exists() else phrase
        log(f"  ref: {ref_path.name} / '{ref_text}'")

        def gen_fn(seed, text, max_tokens):
            return model.generate_voice_clone(
                text=text,
                language="Korean",
                ref_audio=(ref_np, ref_sr),
                ref_text=ref_text,
                max_new_tokens=max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        audio, sr, dur, seed = best_take(
            gen_fn, phrase, idx, args.num_seeds, 5000, args.max_new_tokens, max_safe, output_dir, "a"
        )
        output_sr = sr
        segments.append(audio)
        synth_info.append({"phrase": phrase, "duration": dur, "seed": seed, "ref": ref_path.name})

    final = assemble(segments, output_sr, args.pause_durations, args.leading_silence, args.target_duration)
    out = output_dir / "plan1a_voice_clone.wav"
    sf.write(str(out), final, output_sr)
    final_dur = len(final) / output_sr
    log(f"\n[A] saved: {out.name} ({final_dur:.2f}s)")

    summary = {
        "approach": "VoiceClone (Base model, no fine-tuning)",
        "output": str(out), "duration": final_dur,
        "temperature": args.temperature, "top_p": args.top_p,
        "num_seeds": args.num_seeds, "synthesis_info": synth_info,
    }
    (output_dir / "summary_plan1a.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


# ---------------------------------------------------------------------------
# Approach B — CustomVoice + instruct
# ---------------------------------------------------------------------------

def run_custom_voice(
    model,
    phrases: list[str],
    output_dir: Path,
    args,
) -> Path:
    log("\n" + "=" * 60)
    log("Approach B: CustomVoice + instruct (Vivian, no fine-tuning)")
    log(f"  instruct: {args.instruct}")
    log("=" * 60)

    max_safe = args.max_new_tokens / 12 * 0.85
    segments: list[np.ndarray] = []
    synth_info: list[dict]     = []
    output_sr = 24000

    for idx, phrase in enumerate(phrases):
        log(f"\n[{idx+1}/{len(phrases)}] '{phrase}'")

        def gen_fn(seed, text, max_tokens):
            return model.generate_custom_voice(
                text=text,
                language="Korean",
                speaker="Vivian",
                instruct=args.instruct,
                max_new_tokens=max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        audio, sr, dur, seed = best_take(
            gen_fn, phrase, idx, args.num_seeds, 6000, args.max_new_tokens, max_safe, output_dir, "b"
        )
        output_sr = sr
        segments.append(audio)
        synth_info.append({"phrase": phrase, "duration": dur, "seed": seed})

    final = assemble(segments, output_sr, args.pause_durations, args.leading_silence, args.target_duration)
    out = output_dir / "plan1b_custom_voice_instruct.wav"
    sf.write(str(out), final, output_sr)
    final_dur = len(final) / output_sr
    log(f"\n[B] saved: {out.name} ({final_dur:.2f}s)")

    summary = {
        "approach": "CustomVoice + instruct, no fine-tuning (Vivian)",
        "output": str(out), "duration": final_dur,
        "instruct": args.instruct,
        "temperature": args.temperature, "top_p": args.top_p,
        "num_seeds": args.num_seeds, "synthesis_info": synth_info,
    }
    (output_dir / "summary_plan1b.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plan 1: VoiceClone / CustomVoice without fine-tuning")
    p.add_argument("--ref_dir",    required=True,  help="Directory with sample_NN.wav + sample_NN.txt")
    p.add_argument("--output_dir", required=True,  help="Output directory for wav and summary JSON")
    p.add_argument("--phrases_file", default=None, help="Text file with one phrase per line")
    p.add_argument("--text",       default=None,   help="Single phrase (if --phrases_file not given)")
    p.add_argument("--instruct",   default=DEFAULT_INSTRUCT)
    p.add_argument("--temperature", type=float, default=1.05)
    p.add_argument("--top_p",       type=float, default=0.95)
    p.add_argument("--num_seeds",   type=int,   default=15)
    p.add_argument("--max_new_tokens", type=int, default=320)
    p.add_argument("--leading_silence", type=float, default=0.0,
                   help="Seconds of silence before first phrase")
    p.add_argument("--pause_durations", default="",
                   help="Comma-separated pause durations between phrases (e.g. 0.32,0.74)")
    p.add_argument("--target_duration", type=float, default=0.0,
                   help="Pad/trim final output to this length in seconds (0=disabled)")
    p.add_argument("--skip_a", action="store_true", help="Skip Approach A (VoiceClone)")
    p.add_argument("--skip_b", action="store_true", help="Skip Approach B (CustomVoice+instruct)")
    return p.parse_args()


def main():
    args = parse_args()

    ref_dir    = Path(args.ref_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse phrases
    if args.phrases_file:
        phrases = [l.strip() for l in Path(args.phrases_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.text:
        phrases = [args.text]
    else:
        print("Error: provide --phrases_file or --text", file=sys.stderr)
        sys.exit(1)

    # Parse pause durations
    args.pause_durations = (
        [float(x) for x in args.pause_durations.split(",") if x.strip()]
        if args.pause_durations else []
    )

    # Approach A
    if not args.skip_a:
        log("\n[Loading] Base model...")
        base_model = load_model(BASE_REPO)
        if torch.cuda.is_available():
            base_model = base_model.cuda()
        run_voice_clone(base_model, phrases, ref_dir, output_dir, args)
        del base_model
        torch.cuda.empty_cache()

    # Approach B
    if not args.skip_b:
        log("\n[Loading] CustomVoice model...")
        cv_model = load_model(CUSTOM_VOICE_REPO)
        if torch.cuda.is_available():
            cv_model = cv_model.cuda()
        run_custom_voice(cv_model, phrases, output_dir, args)
        del cv_model
        torch.cuda.empty_cache()

    log("\nDone.")


if __name__ == "__main__":
    main()
