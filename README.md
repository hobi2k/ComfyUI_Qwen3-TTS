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

## Package Layout

The repo is now split so the standalone vendor code is easier to maintain when
`Qwen3-TTS-Demo` changes:

- `core/`
  shared ComfyUI paths, cache helpers, audio conversion, progress utilities
- `inference/voicebox/`
  vendor mirror of `Qwen3-TTS/inference/voicebox/` for runtime helpers
- `finetuning/voicebox_training_common.py`
  vendor mirror of `Qwen3-TTS/finetuning/voicebox_training_common.py`
- `fusion/make_voicebox_checkpoint.py`
  vendor mirror of `Qwen3-TTS/fusion/make_voicebox_checkpoint.py`
- `nodes.py`
  ComfyUI node classes only, importing shared helpers from `core/`, `inference/`, `finetuning/`, and `fusion/`
- `finetuning/`
  upstream-style training backends kept for the older fine-tune node
- `workflows/`
  example ComfyUI graphs for clone, prompt reuse, CustomVoice, and VoiceBox flows

The older `voicebox/` package is kept only as a compatibility shim.
If you need to re-vendor from the demo repo again, follow `Qwen3-TTS/inference`,
`Qwen3-TTS/finetuning`, and `Qwen3-TTS/fusion`.

## ComfyUI Nodes

| Node | Description |
|------|-------------|
| Qwen3-TTS Loader | Load Base, CustomVoice, or fine-tuned model |
| Qwen3-TTS Voice Clone | Clone voice from reference audio (Base model). Now supports **temperature** and **top_p**. |
| Qwen3-TTS Custom Voice | Generate with CustomVoice model + instruct. Now supports **temperature** and **top_p**. |
| Qwen3-TTS Voice Design | Voice design via natural language instruct |
| Qwen3-TTS Base+CustomVoice Clone+Instruct | Demo-style hybrid inference: Base clone prompt + CustomVoice instruct |
| Qwen3-TTS Clone Prompt From Audio | Create a clone prompt from any audio source, including VoiceDesign output |
| Qwen3-TTS Custom Voice From Prompt | Generate consistent CustomVoice speech from clone prompt + instruct |
| Qwen3-TTS Directed Clone From Voice Design | Hybrid node: VoiceDesign audio -> Base clone prompt -> CustomVoice clone + instruct |
| Qwen3-TTS Hybrid Clone+Instruct Preset | Hybrid preset path with CustomVoice speaker anchor, matching the demo extension flow |
| Qwen3-TTS Prompt Maker | Create reusable voice clone prompt from reference audio |
| Qwen3-TTS Save/Load Prompt | Persist voice clone prompts to disk |
| Qwen3-TTS VoiceBox Instruct | Run regular VoiceBox `speaker + instruct` inference |
| Qwen3-TTS VoiceBox Clone | Dedicated VoiceBox clone node mapped to the upstream clone entrypoint |
| Qwen3-TTS VoiceBox Clone+Instruct | Dedicated VoiceBox clone+instruct node mapped to the upstream clone_instruct entrypoint |
| Qwen3-TTS VoiceBox Clone Experiment | Run low-level VoiceBox clone / clone+instruct strategies |
| Qwen3-TTS VoiceBox Morph Speaker | Create a persistent morphed speaker row inside a VoiceBox checkpoint |
| Qwen3-TTS Dataset Maker | Build fine-tuning dataset from an audio folder |
| Qwen3-TTS Data Prep | Tokenize dataset for fine-tuning |
| Qwen3-TTS Finetune | Fine-tune node (use sft_12hz_v4 backend for instruct-aware training) |
| Qwen3-TTS SFT Base 12Hz | Upstream-style Base SFT node alias |
| Qwen3-TTS Plain CustomVoice Finetune | Standalone plain CustomVoice fine-tuning copied from the demo flow |
| Qwen3-TTS SFT CustomVoice 12Hz | Upstream-style CustomVoice SFT node alias |
| Qwen3-TTS VoiceBox Create | Convert a plain CustomVoice checkpoint into a self-contained VoiceBox checkpoint |
| Qwen3-TTS Upload VoiceBox To Hub | Upload a VoiceBox checkpoint folder to Hugging Face Hub |
| Qwen3-TTS VoiceBox Bootstrap Finetune | Train a VoiceBox checkpoint directly from CustomVoice + Base speaker encoder |
| Qwen3-TTS SFT VoiceBox Bootstrap 12Hz | Upstream-style VoiceBox bootstrap SFT node alias |
| Qwen3-TTS VoiceBox Finetune | Run `VoiceBox -> VoiceBox` fine-tuning on an existing VoiceBox checkpoint |
| Qwen3-TTS SFT VoiceBox 12Hz | Upstream-style VoiceBox SFT node alias |
| Qwen3-TTS Audio Compare | Compare two audio outputs |

## Example Workflows

- `voice_design_to_prompt_to_customvoice.json`
  VoiceDesign -> clone prompt extraction -> CustomVoice generate
- `customvoice_from_saved_prompt.json`
  Reuse a previously saved prompt with `Qwen3LoadPrompt`
- `compare_x_vector_only_mode.json`
  Compare `x_vector_only_mode=true/false` with the same reference
- `base_customvoice_clone_instruct.json`
  Demo-style Base + CustomVoice clone+instruct chain
- `plain_customvoice_finetune_and_voicebox_create.json`
  Plain CustomVoice fine-tune followed by VoiceBox checkpoint fusion
- `voicebox_bootstrap_finetune.json`
  Direct VoiceBox bootstrap training from CustomVoice + Base speaker encoder
- `voicebox_retrain.json`
  Continue training an existing VoiceBox checkpoint
- `hybrid_clone_instruct_preset.json`
  Preset-driven hybrid clone+instruct inference with speaker anchors
- `voicebox_instruct_inference.json`
  Basic `speaker + instruct` VoiceBox inference
- `voicebox_clone.json`
  Dedicated VoiceBox clone node example
- `voicebox_clone_instruct.json`
  Dedicated VoiceBox clone+instruct node example
- `voicebox_clone_experiment.json`
  Low-level VoiceBox clone / clone+instruct strategy playground
- `voicebox_morph_and_infer.json`
  Morph a new persistent speaker into a VoiceBox checkpoint, then infer
- `voicebox_morph_from_prompt_and_infer.json`
  Morph a persistent speaker from a generated prompt, then infer
- `sft_base_12hz.json`
  Upstream-style Base SFT training node example
- `sft_custom_voice_12hz.json`
  Upstream-style CustomVoice SFT training node example
- `sft_voicebox_12hz.json`
  Upstream-style VoiceBox SFT training node example
- `sft_voicebox_bootstrap_12hz.json`
  Upstream-style VoiceBox bootstrap SFT training node example
- `upload_voicebox_to_hub.json`
  Hugging Face Hub upload node example

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
