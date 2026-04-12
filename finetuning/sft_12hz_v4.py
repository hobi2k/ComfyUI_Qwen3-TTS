# coding=utf-8
"""
sft_12hz_v4.py — Fine-tuning with instruct support + CustomVoice base

Changes from sft_12hz.py:
  1. Accepts any model as init (Base OR CustomVoice)
  2. Reads instruct_ids / instruct_mask from batch (requires updated dataset.py)
  3. Prepends instruct embeddings to input_embeddings before talker forward
  4. Uses extended codec_mask for hidden_states extraction
"""
import argparse
import json
import os
import shutil
import sys

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

# Ensure relative imports work regardless of calling directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import TTSDataset  # updated dataset with instruct support


def _patch_speech_tokenizer_fallback():
    """
    Monkey-patch Qwen3TTSForConditionalGeneration.from_pretrained to handle
    missing speech_tokenizer weights in the HF cache snapshot.

    The HF snapshot for CustomVoice often ships without speech_tokenizer
    model weights (only the config). This patch:
      1. Calls PreTrainedModel.from_pretrained to load the main model weights.
      2. Tries to load the speech tokenizer from multiple candidate directories
         (the model's own speech_tokenizer/ subdir, then the ComfyUI models
         folder, then the standalone Qwen3-TTS-Tokenizer-12Hz repo).
      3. Loads generate_config if available.
    """
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
        from transformers import PreTrainedModel

        if getattr(Qwen3TTSForConditionalGeneration, "_speech_tok_patched", False):
            return  # already patched

        @classmethod
        def _smart_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            """Load main model then find speech_tokenizer from best available location."""
            import json as _json

            model = PreTrainedModel.from_pretrained.__func__(
                cls, pretrained_model_name_or_path, *model_args, **kwargs
            )

            # ── Find speech_tokenizer with actual weights ──────────────────
            candidates = []

            # 1. Standard subdir of the model path
            if os.path.isdir(pretrained_model_name_or_path):
                st = os.path.join(pretrained_model_name_or_path, "speech_tokenizer")
                if os.path.isfile(os.path.join(st, "model.safetensors")):
                    candidates.append(st)

            # 2. HF cache snapshot subdir
            try:
                from transformers.utils import cached_file
                st_cfg = cached_file(pretrained_model_name_or_path, "speech_tokenizer/config.json")
                if st_cfg:
                    st_dir = os.path.dirname(st_cfg)
                    if os.path.isfile(os.path.join(st_dir, "model.safetensors")):
                        candidates.append(st_dir)
            except Exception:
                pass

            # 3. ComfyUI models folder: scan for any model with speech_tokenizer weights
            _comfyui_qwen = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "..", "models", "Qwen3-TTS"
            )
            _comfyui_qwen = os.path.normpath(_comfyui_qwen)
            if os.path.isdir(_comfyui_qwen):
                # Prefer the standalone tokenizer repo first
                standalone = os.path.join(_comfyui_qwen, "Qwen3-TTS-Tokenizer-12Hz")
                if os.path.isfile(os.path.join(standalone, "model.safetensors")):
                    candidates.append(standalone)
                # Also scan model subdirs for speech_tokenizer/
                for d in os.listdir(_comfyui_qwen):
                    st = os.path.join(_comfyui_qwen, d, "speech_tokenizer")
                    if os.path.isfile(os.path.join(st, "model.safetensors")):
                        candidates.append(st)

            # Try each candidate
            speech_tokenizer_loaded = False
            for st_dir in candidates:
                try:
                    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
                    speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(st_dir)
                    model.load_speech_tokenizer(speech_tokenizer)
                    speech_tokenizer_loaded = True
                    print(f"[sft_12hz_v4] speech_tokenizer loaded from: {st_dir}", flush=True)
                    break
                except Exception as e:
                    print(f"[sft_12hz_v4] speech_tokenizer candidate {st_dir} failed: {e}", flush=True)

            if not speech_tokenizer_loaded:
                print("[sft_12hz_v4] WARNING: speech_tokenizer not loaded — synthesis will fail", flush=True)

            # ── Load generate_config ───────────────────────────────────────
            try:
                from transformers.utils import cached_file
                gcfg_path = cached_file(pretrained_model_name_or_path, "generation_config.json")
                if gcfg_path:
                    with open(gcfg_path, "r", encoding="utf-8") as f:
                        model.load_generate_config(_json.load(f))
            except Exception:
                pass

            return model

        Qwen3TTSForConditionalGeneration.from_pretrained = _smart_from_pretrained
        Qwen3TTSForConditionalGeneration._speech_tok_patched = True
    except Exception as e:
        print(f"[sft_12hz_v4] speech_tokenizer patch skipped: {e}", flush=True)


def train(
    init_model_path: str,
    output_model_path: str,
    train_jsonl: str,
    batch_size: int = 1,
    lr: float = 1e-7,
    num_epochs: int = 2,
    speaker_name: str = "speaker_v4",
    gradient_accumulation_steps: int = 4,
    mixed_precision: str = "bf16",
    max_grad_norm: float = 1.0,
    log_every_steps: int = 5,
    seed: int = 42,
) -> str:
    """
    Fine-tune Qwen3-TTS with instruct support.
    Returns path to the final epoch checkpoint directory.
    """
    _patch_speech_tokenizer_fallback()
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    torch.manual_seed(seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    accelerator.print(f"Loading model from: {init_model_path}")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    config = AutoConfig.from_pretrained(init_model_path)

    train_data = [json.loads(line) for line in open(train_jsonl, encoding="utf-8")]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=lr, weight_decay=0.01)
    model, optimizer, dataloader = accelerator.prepare(qwen3tts.model, optimizer, dataloader)
    model.train()

    target_speaker_embedding = None
    final_ckpt = output_model_path

    # Resolve init_model_path to a local directory for checkpoint copying.
    # If it's a HuggingFace repo ID (not an existing local dir), download/resolve snapshot.
    if os.path.isdir(init_model_path):
        local_init_path = init_model_path
    else:
        try:
            from huggingface_hub import snapshot_download
            local_init_path = snapshot_download(init_model_path, local_files_only=True)
            accelerator.print(f"Resolved init model path: {local_init_path}")
        except Exception as e:
            accelerator.print(f"Warning: could not resolve local snapshot ({e}). "
                              "Checkpoint copy may fail.")
            local_init_path = init_model_path

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):

                input_ids           = batch['input_ids']
                codec_ids           = batch['codec_ids']
                ref_mels            = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask= batch['codec_embedding_mask']
                attention_mask      = batch['attention_mask']
                codec_0_labels      = batch['codec_0_labels']
                codec_mask          = batch['codec_mask']
                instruct_ids_batch  = batch.get('instruct_ids')   # [B, T_inst] or None
                instruct_mask_batch = batch.get('instruct_mask')  # [B, T_inst] or None

                # Speaker embedding
                if model.speaker_encoder is not None:
                    # Base model: extract speaker embedding from reference mel
                    speaker_embedding = model.speaker_encoder(
                        ref_mels.to(model.device).to(model.dtype)
                    ).detach()
                    if target_speaker_embedding is None:
                        target_speaker_embedding = speaker_embedding
                else:
                    # CustomVoice model has no speaker_encoder.
                    # Initialize new speaker from "sohee" (index 2864, Korean female)
                    # and reuse it every step.
                    if target_speaker_embedding is None:
                        INIT_IDX = 2864  # sohee
                        target_speaker_embedding = (
                            model.talker.model.codec_embedding.weight[INIT_IDX]
                            .detach().clone().unsqueeze(0)
                        )
                    speaker_embedding = target_speaker_embedding

                input_text_ids  = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # For 1.7B: text_hidden_size == hidden_size → no text_projection
                input_text_embedding  = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_emb = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
                    codec_i_emb = codec_i_emb * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_emb

                # ── V4: prepend instruct embeddings ──────────────────────────
                if instruct_ids_batch is not None and instruct_ids_batch.shape[1] > 0:
                    T_inst = instruct_ids_batch.shape[1]
                    B      = input_ids.shape[0]

                    # Embed instruct tokens (1.7B: same dim, no projection)
                    inst_emb = model.talker.model.text_embedding(instruct_ids_batch)
                    inst_mask_f = instruct_mask_batch.unsqueeze(-1).to(inst_emb.dtype)
                    inst_emb = inst_emb * inst_mask_f  # zero out padding

                    # Prepend to sequence
                    input_embeddings = torch.cat([inst_emb, input_embeddings], dim=1)
                    attention_mask   = torch.cat(
                        [instruct_mask_batch.long(), attention_mask], dim=1
                    )

                    # Labels: -100 for instruct positions (no loss on instruct output)
                    inst_labels = torch.full(
                        (B, T_inst), -100, dtype=torch.long, device=codec_0_labels.device
                    )
                    codec_0_labels = torch.cat([inst_labels, codec_0_labels], dim=1)

                    # Extend codec_mask with False for instruct positions
                    false_inst = torch.zeros((B, T_inst), dtype=torch.bool, device=codec_mask.device)
                    codec_mask_ext = torch.cat([false_inst, codec_mask], dim=1)
                else:
                    codec_mask_ext = codec_mask
                # ─────────────────────────────────────────────────────────────

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask_ext[:, 1:]]
                talker_codec_ids     = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + sub_talker_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

            if step % log_every_steps == 0:
                accelerator.print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

        # ── Save checkpoint ──────────────────────────────────────────────────
        if accelerator.is_main_process:
            epoch_dir = os.path.join(output_model_path, f"epoch_{epoch + 1}")
            shutil.copytree(local_init_path, epoch_dir, dirs_exist_ok=True)

            # Update config: mark as custom_voice, register speaker
            cfg_path = os.path.join(epoch_dir, "config.json")
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["tts_model_type"] = "custom_voice"
            talker_cfg = cfg.get("talker_config", {})
            talker_cfg["spk_id"]         = {speaker_name: 3000}
            talker_cfg["spk_is_dialect"] = {speaker_name: False}
            cfg["talker_config"] = talker_cfg
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)

            # Save weights (drop speaker_encoder, inject speaker embedding)
            unwrapped = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().cpu() for k, v in unwrapped.state_dict().items()}
            drop_keys = [k for k in state_dict if k.startswith("speaker_encoder")]
            for k in drop_keys:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = (
                target_speaker_embedding[0].detach().cpu().to(weight.dtype)
            )
            save_file(state_dict, os.path.join(epoch_dir, "model.safetensors"))
            accelerator.print(f"  → saved: {epoch_dir}")
            final_ckpt = epoch_dir

    return final_ckpt


# ── Standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path",   type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--train_jsonl",       type=str, required=True)
    parser.add_argument("--batch_size",        type=int,   default=1)
    parser.add_argument("--lr",                type=float, default=1e-7)
    parser.add_argument("--num_epochs",        type=int,   default=2)
    parser.add_argument("--speaker_name",      type=str,   default="speaker_v4")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mixed_precision",   type=str,   default="bf16")
    parser.add_argument("--max_grad_norm",     type=float, default=1.0)
    parser.add_argument("--log_every_steps",   type=int,   default=5)
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    result = train(
        init_model_path=args.init_model_path,
        output_model_path=args.output_model_path,
        train_jsonl=args.train_jsonl,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        speaker_name=args.speaker_name,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        max_grad_norm=args.max_grad_norm,
        log_every_steps=args.log_every_steps,
        seed=args.seed,
    )
    print(f"Training complete. Final checkpoint: {result}")
