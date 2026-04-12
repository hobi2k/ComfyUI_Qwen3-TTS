"""
plan2_finetune_customvoice.py
Fine-tune Qwen3-TTS CustomVoice model on real emotional audio segments,
then synthesize with the fine-tuned model + instruct.

Key idea (Plan 2)
-----------------
Unlike fine-tuning on the Base model (which destroys instruct ability),
we fine-tune the CustomVoice model at a very low LR (1e-7).
This adapts the speaker embedding to the target voice while preserving
the model's instruct-following capability.

The fine-tuning JSONL includes an `instruct` field so the model
learns the association: this voice + this instruct -> this audio style.

Usage
-----
python scripts/plan2_finetune_customvoice.py \\
    --ref_dir       /path/to/emotional_segments \\
    --output_dir    /path/to/project \\
    --phrases_file  /path/to/phrases.txt \\
    [--speaker_name my_speaker] \\
    [--epochs 2] [--lr 1e-7] \\
    [--instruct "Desperate female voice..."] \\
    [--temperature 1.05] [--top_p 0.95] [--num_seeds 15] \\
    [--leading_silence 5.65] \\
    [--pause_durations 0.32,0.74,0.97,0.91,1.80,1.56] \\
    [--target_duration 36.20]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


DEFAULT_INSTRUCT = (
    "Desperate female voice pleading and begging under extreme duress. "
    "Breathing is heavy and strained throughout, audible between every word. "
    "Voice breaks and trembles from anguish — not crying but suffering. "
    "Each word is forced out with effort, pitch rising desperately. "
    "Anguished moaning quality — raw and strained, not soft. "
    "Speak very slowly, voice cracking under desperation."
)

CUSTOM_VOICE_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
TOKENIZER_REPO    = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
TRIM_TOP_DB       = 20


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

    if target_duration > 0:
        tf = int(round(target_duration * sr))
        if len(speech) > tf:
            speech = speech[:tf]
        elif len(speech) < tf:
            gap = tf - len(speech)
            ch = speech.shape[1] if speech.ndim > 1 else 1
            pad = np.zeros((gap, ch), dtype=speech.dtype) if speech.ndim > 1 else np.zeros(gap, dtype=speech.dtype)
            speech = np.concatenate([speech, pad], axis=0)
    return speech


def add_instruct_to_jsonl(src: Path, dst: Path, instruct: str) -> None:
    lines = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry["instruct"] = instruct
            lines.append(json.dumps(entry, ensure_ascii=False))
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"  instruct field added: {dst.name} ({len(lines)} entries)")


def parse_args():
    p = argparse.ArgumentParser(description="Plan 2: Fine-tune CustomVoice + synthesize")
    p.add_argument("--ref_dir",      required=True, help="Dir with sample_NN.wav + sample_NN.txt")
    p.add_argument("--output_dir",   required=True, help="Project output directory")
    p.add_argument("--phrases_file", default=None,  help="Text file with one phrase per line")
    p.add_argument("--text",         default=None,  help="Single phrase")
    p.add_argument("--speaker_name", default="custom_speaker_v4")
    p.add_argument("--epochs",  type=int,   default=2)
    p.add_argument("--lr",      type=float, default=1e-7)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum",  type=int, default=4)
    p.add_argument("--instruct",    default=DEFAULT_INSTRUCT)
    p.add_argument("--temperature", type=float, default=1.05)
    p.add_argument("--top_p",       type=float, default=0.95)
    p.add_argument("--num_seeds",   type=int,   default=15)
    p.add_argument("--max_new_tokens", type=int, default=320)
    p.add_argument("--leading_silence",  type=float, default=0.0)
    p.add_argument("--pause_durations",  default="")
    p.add_argument("--target_duration",  type=float, default=0.0)
    p.add_argument("--skip_finetune", action="store_true",
                   help="Skip fine-tuning and use existing checkpoint")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Existing checkpoint dir (used when --skip_finetune)")
    return p.parse_args()


def main():
    args = parse_args()

    ref_dir      = Path(args.ref_dir)
    output_dir   = Path(args.output_dir)
    dataset_dir  = output_dir / "dataset"
    finetune_dir = output_dir / "finetune_v4"
    outputs_dir  = output_dir / "outputs"
    for d in [dataset_dir, finetune_dir, outputs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Locate finetuning scripts relative to this file
    script_dir = Path(__file__).parent
    repo_root  = script_dir.parent
    ft_dir     = repo_root / "finetuning"
    sys.path.insert(0, str(ft_dir))

    if args.phrases_file:
        phrases = [l.strip() for l in Path(args.phrases_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.text:
        phrases = [args.text]
    else:
        print("Error: provide --phrases_file or --text", file=sys.stderr)
        sys.exit(1)

    args.pause_durations = (
        [float(x) for x in args.pause_durations.split(",") if x.strip()]
        if args.pause_durations else []
    )

    # Patch: fix speech_tokenizer loading (HF cache snapshot lacks weights;
    # fall back to ComfyUI models folder or standalone tokenizer repo)
    try:
        import os as _os
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
        from transformers import PreTrainedModel

        if not getattr(Qwen3TTSForConditionalGeneration, "_speech_tok_patched", False):
            # Pre-build candidate list
            _script_dir = Path(__file__).parent
            _comfyui_root = _script_dir.parent.parent.parent  # custom_nodes/.. = Data/Packages/ComfyUI
            _qwen_models = _comfyui_root / "models" / "Qwen3-TTS"
            _st_candidates: list[str] = []
            standalone = _qwen_models / "Qwen3-TTS-Tokenizer-12Hz"
            if (standalone / "model.safetensors").exists():
                _st_candidates.append(str(standalone))
            if _qwen_models.is_dir():
                for _d in _qwen_models.iterdir():
                    st = _d / "speech_tokenizer"
                    if (st / "model.safetensors").exists():
                        _st_candidates.append(str(st))

            @classmethod
            def _smart_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
                import json as _json
                model = PreTrainedModel.from_pretrained.__func__(
                    cls, pretrained_model_name_or_path, *model_args, **kwargs
                )
                cands = list(_st_candidates)
                own_st = Path(str(pretrained_model_name_or_path)) / "speech_tokenizer"
                if (own_st / "model.safetensors").exists():
                    cands.insert(0, str(own_st))
                loaded = False
                for st_dir in cands:
                    try:
                        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
                        model.load_speech_tokenizer(Qwen3TTSTokenizer.from_pretrained(st_dir))
                        loaded = True
                        break
                    except Exception:
                        pass
                if not loaded:
                    log("[plan2] WARNING: speech_tokenizer not loaded")
                try:
                    from transformers.utils import cached_file
                    gcfg = cached_file(pretrained_model_name_or_path, "generation_config.json")
                    if gcfg:
                        with open(gcfg, "r", encoding="utf-8") as f:
                            model.load_generate_config(_json.load(f))
                except Exception:
                    pass
                return model

            Qwen3TTSForConditionalGeneration.from_pretrained = _smart_from_pretrained
            Qwen3TTSForConditionalGeneration._speech_tok_patched = True
    except Exception as e:
        log(f"[plan2] speech_tokenizer patch skipped: {e}")

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    # ── STEP 1: Dataset ────────────────────────────────────────────────────
    if not args.skip_finetune:
        log("=" * 60)
        log("STEP 1: Building dataset")
        log("=" * 60)

        try:
            import importlib
            import importlib.util
            import types
            server = types.ModuleType("server")
            server.PromptServer = type("PromptServer", (), {"instance": None})
            sys.modules["server"] = server
            spec = importlib.util.spec_from_file_location(
                "_qwen_nodes_pkg",
                repo_root / "__init__.py",
                submodule_search_locations=[str(repo_root)],
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_qwen_nodes_pkg"] = mod
            spec.loader.exec_module(mod)
            nodes = importlib.import_module("_qwen_nodes_pkg.nodes")
            dataset_maker = nodes.Qwen3DatasetFromFolder()
            data_prep     = nodes.Qwen3DataPrep()
            use_nodes = True
            log("  Using ComfyUI node pipeline for dataset creation.")
        except Exception as e:
            log(f"  ComfyUI node import failed ({e}), cannot create dataset.")
            raise

        first_wav = sorted(ref_dir.glob("sample_*.wav"))[0]
        raw_jsonl = Path(dataset_maker.create_dataset(
            folder_path=str(ref_dir),
            output_filename="dataset_raw.jsonl",
            ref_audio_path=str(first_wav),
        )[0])
        shutil.copy2(raw_jsonl, dataset_dir / raw_jsonl.name)

        processed_jsonl = Path(data_prep.process(
            jsonl_path=str(raw_jsonl),
            tokenizer_repo=TOKENIZER_REPO,
            source="HuggingFace",
            batch_size=2,
            unique_id=None,
        )[0])
        shutil.copy2(processed_jsonl, dataset_dir / processed_jsonl.name)
        meta = processed_jsonl.with_suffix(".meta.json")
        if meta.exists():
            shutil.copy2(meta, dataset_dir / meta.name)

        instruct_jsonl = dataset_dir / "dataset_instruct.jsonl"
        add_instruct_to_jsonl(processed_jsonl, instruct_jsonl, args.instruct)

        # ── STEP 2: Fine-tune ──────────────────────────────────────────────
        log("\n" + "=" * 60)
        log(f"STEP 2: Fine-tuning CustomVoice ({args.epochs} epochs, lr={args.lr})")
        log("=" * 60)

        import sft_12hz_v4
        final_ckpt = sft_12hz_v4.train(
            init_model_path=CUSTOM_VOICE_REPO,
            output_model_path=str(finetune_dir),
            train_jsonl=str(instruct_jsonl),
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.epochs,
            speaker_name=args.speaker_name,
            gradient_accumulation_steps=args.grad_accum,
            mixed_precision="bf16",
            max_grad_norm=1.0,
            log_every_steps=5,
        )
        log(f"Fine-tune complete: {final_ckpt}")
        torch.cuda.empty_cache()
    else:
        if not args.checkpoint_dir:
            print("Error: --skip_finetune requires --checkpoint_dir", file=sys.stderr)
            sys.exit(1)
        final_ckpt = args.checkpoint_dir
        log(f"Skipping fine-tune, using checkpoint: {final_ckpt}")

    # ── STEP 3: Load fine-tuned model ──────────────────────────────────────
    log("\n" + "=" * 60)
    log("STEP 3: Loading fine-tuned model")
    log("=" * 60)
    model = Qwen3TTSModel.from_pretrained(
        CUSTOM_VOICE_REPO,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    from safetensors.torch import load_file
    ckpt_weights = load_file(str(Path(final_ckpt) / "model.safetensors"))
    model.model.load_state_dict(ckpt_weights, strict=False)
    if torch.cuda.is_available():
        model = model.cuda()
    log("Model loaded.")

    # ── STEP 4: Synthesize ─────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log(f"STEP 4: Synthesis ({args.num_seeds} seeds/phrase)")
    log(f"  instruct: {args.instruct}")
    log("=" * 60)

    max_safe = args.max_new_tokens / 12 * 0.85
    segments: list[np.ndarray] = []
    synth_info: list[dict]     = []
    output_sr = 24000

    for idx, phrase in enumerate(phrases):
        log(f"\n[{idx+1}/{len(phrases)}] '{phrase}'")
        best_audio: np.ndarray | None = None
        best_dur   = 0.0
        best_seed  = 0
        best_sr    = 24000

        for i in range(args.num_seeds):
            seed = 7000 + idx * 100 + i
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            wavs, sr = model.generate_custom_voice(
                text=phrase,
                language="Korean",
                speaker=args.speaker_name,
                instruct=args.instruct,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            speech = wavs[0]
            if speech.ndim == 1:
                speech = speech[:, np.newaxis]

            tmp = outputs_dir / f"_tmp_p2_{idx}_s{i}.wav"
            sf.write(str(tmp), speech, sr)
            dur = trim_trailing_silence(tmp)
            trimmed, _ = sf.read(str(tmp), always_2d=True)
            tmp.unlink(missing_ok=True)

            flag = " [SKIP: loop]" if dur >= max_safe else ""
            log(f"    seed {seed}: {dur:.2f}s{flag}")

            if dur >= max_safe:
                continue
            if dur > best_dur:
                best_dur  = dur
                best_audio = trimmed
                best_sr   = sr
                best_seed  = seed

        if best_audio is None:
            best_audio = np.zeros((int(0.5 * best_sr), 1), dtype=np.float32)
            best_dur   = 0.5

        log(f"  -> seed {best_seed}, {best_dur:.2f}s")
        output_sr = best_sr
        segments.append(best_audio)
        synth_info.append({"phrase": phrase, "duration": best_dur, "seed": best_seed})

    final = assemble(segments, output_sr, args.pause_durations, args.leading_silence, args.target_duration)
    out = outputs_dir / "plan2_finetune_customvoice.wav"
    sf.write(str(out), final, output_sr)
    final_dur = len(final) / output_sr

    summary = {
        "approach": "CustomVoice fine-tune + instruct in training",
        "output": str(out), "duration": final_dur,
        "speaker_name": args.speaker_name,
        "instruct": args.instruct,
        "epochs": args.epochs, "lr": args.lr,
        "temperature": args.temperature, "top_p": args.top_p,
        "num_seeds": args.num_seeds,
        "synthesis_info": synth_info,
    }
    (outputs_dir / "summary_plan2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"\nDone. Output: {out.name} ({final_dur:.2f}s)")


if __name__ == "__main__":
    main()
