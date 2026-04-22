# ComfyUI_Qwen3-TTS

Extended ComfyUI custom nodes for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), with:

- **Temperature/top_p control** on VoiceClone and CustomVoice nodes (upstream nodes lacked this)
- **Instruct-aware fine-tuning pipeline** (`finetuning/sft_12hz_v4.py`) — fine-tune on CustomVoice base instead of Base model, preserving instruct-following ability
- **CLI scripts** for running VoiceClone+instruct and CustomVoice fine-tuning experiments outside ComfyUI

## Why this repo exists

The upstream ComfyUI nodes for Qwen3-TTS have a critical omission: temperature and top_p are not forwarded to `generate_custom_voice()` or `generate_voice_clone()`. This means emotional variability is capped at the default temperature=0.9, making it nearly impossible to generate expressive, breathy, or emotionally intense speech.

Fine-tuning on the Base model destroys instruct-following ability. This repo adds `sft_12hz_v4.py`, which fine-tunes the CustomVoice model (which already has instruct support) at a very low LR, preserving emotional expressiveness while adapting the speaker timbre.

## Installation

Drop this folder into `ComfyUI/custom_nodes/`:

```
ComfyUI/custom_nodes/ComfyUI_Qwen3-TTS/
```

Install dependencies:

```bash
pip install qwen-tts soundfile librosa safetensors accelerate
```

## ComfyUI Nodes

| Node | Description |
|------|-------------|
| Qwen3-TTS Loader | Load Base, CustomVoice, or fine-tuned model |
| Qwen3-TTS Voice Clone | Clone voice from reference audio (Base model). Now supports **temperature** and **top_p**. |
| Qwen3-TTS Custom Voice | Generate with CustomVoice model + instruct. Now supports **temperature** and **top_p**. |
| Qwen3-TTS Voice Design | Voice design via natural language instruct |
| Qwen3-TTS Clone Prompt From Audio | Create a clone prompt from any audio source, including VoiceDesign output |
| Qwen3-TTS Custom Voice From Prompt | Generate consistent CustomVoice speech from clone prompt + instruct |
| Qwen3-TTS Directed Clone From Voice Design | Hybrid node: VoiceDesign audio -> Base clone prompt -> CustomVoice clone + instruct |
| Qwen3-TTS Prompt Maker | Create reusable voice clone prompt from reference audio |
| Qwen3-TTS Save/Load Prompt | Persist voice clone prompts to disk |
| Qwen3-TTS Dataset Maker | Build fine-tuning dataset from an audio folder |
| Qwen3-TTS Data Prep | Tokenize dataset for fine-tuning |
| Qwen3-TTS Finetune | Fine-tune node (use sft_12hz_v4 backend for instruct-aware training) |
| Qwen3-TTS Audio Compare | Compare two audio outputs |

## CLI Scripts

### Plan 1 — No fine-tuning

Two approaches: VoiceClone (Base model, ref audio ICL) and CustomVoice+instruct (no training).

```bash
# Approach A: VoiceClone with emotional reference audio
python scripts/plan1_voice_clone_instruct.py \
    --ref_dir   /path/to/emotional_segments \
    --output_dir /path/to/output \
    --phrases_file /path/to/phrases.txt \
    --temperature 1.05 --top_p 0.95 --num_seeds 15 \
    --skip_b

# Approach B: CustomVoice + instruct (Vivian speaker)
python scripts/plan1_voice_clone_instruct.py \
    --ref_dir   /path/to/emotional_segments \
    --output_dir /path/to/output \
    --phrases_file /path/to/phrases.txt \
    --instruct "Desperate female voice pleading under extreme duress. Heavy strained breathing throughout. Voice breaks from anguish." \
    --temperature 1.05 --top_p 0.95 --num_seeds 15 \
    --skip_a
```

### Plan 2 — Fine-tune CustomVoice + instruct

Fine-tune the CustomVoice model on real emotional audio segments while preserving instruct ability.

```bash
python scripts/plan2_finetune_customvoice.py \
    --ref_dir       /path/to/emotional_segments \
    --output_dir    /path/to/project \
    --phrases_file  /path/to/phrases.txt \
    --speaker_name  my_emotional_speaker \
    --epochs 2 --lr 1e-7 \
    --instruct "Desperate female voice pleading under extreme duress. Heavy strained breathing throughout. Voice breaks from anguish." \
    --temperature 1.05 --top_p 0.95 --num_seeds 15 \
    --leading_silence 5.65 \
    --pause_durations 0.32,0.74,0.97,0.91,1.80,1.56 \
    --target_duration 36.20
```

### Re-synthesize with existing checkpoint (skip fine-tuning)

```bash
python scripts/plan2_finetune_customvoice.py \
    --ref_dir      /path/to/emotional_segments \
    --output_dir   /path/to/project \
    --phrases_file /path/to/phrases.txt \
    --skip_finetune \
    --checkpoint_dir /path/to/project/finetune_v4/epoch_2 \
    --speaker_name my_emotional_speaker \
    --temperature 1.05 --top_p 0.95
```

## Fine-tuning Pipeline Details

### `finetuning/sft_12hz_v4.py`

Key differences from the original `sft_12hz.py`:

1. **CustomVoice base** — accepts any model (Base or CustomVoice), not just Base
2. **Instruct embedding** — reads `instruct` field from training JSONL; prepends instruct token embeddings before the talker forward pass; labels for instruct positions are masked (-100) so no loss is computed on them
3. **SDPA attention** — uses PyTorch SDPA instead of flash_attention_2 for Windows compatibility

### `finetuning/dataset.py`

Extended from upstream to support the `instruct` field in JSONL entries. If `instruct` is present, it is tokenized separately and passed as `instruct_ids`/`instruct_mask` to `sft_12hz_v4`.

## Credits

- Nodes base: [DarioFT/ComfyUI-Qwen3-TTS](https://github.com/DarioFT/ComfyUI-Qwen3-TTS)
- Fine-tuning base: [flybirdxx/ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS)
- Model: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba/Qwen team

## License

Apache-2.0
