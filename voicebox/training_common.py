import json
import os
import shutil
from pathlib import Path

import comfy.model_management as mm
import torch
from accelerate import Accelerator
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Adafactor, AutoConfig

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

from ..core.progress import send_progress_text
from ..dataset import TTSDataset
from .runtime import load_qwen_or_voicebox_model


def resolve_training_output_dir(output_dir: str) -> str:
    output_path = Path(output_dir).resolve()
    final_dir = output_path / "final"
    if final_dir.exists():
        return str(final_dir)
    return str(output_path)


def resolve_training_attention() -> str:
    configured = (os.getenv("QWEN_DEMO_ATTN_IMPL") or "").strip()
    if configured:
        return configured
    device = mm.get_torch_device()
    if device.type == "mps":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        import importlib.metadata

        importlib.metadata.version("flash_attn")
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def resolve_training_runtime() -> dict:
    device = mm.get_torch_device()
    if device.type == "cuda":
        return {
            "device": device,
            "dtype": torch.bfloat16,
            "mixed_precision": "bf16",
            "attention": resolve_training_attention(),
        }
    return {
        "device": device,
        "dtype": torch.float32,
        "mixed_precision": "no",
        "attention": "sdpa",
    }


def resolve_jsonl_audio_path(raw_path: str, jsonl_path: Path) -> str:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate)
    from_jsonl_dir = (jsonl_path.resolve().parent / candidate).resolve()
    if from_jsonl_dir.exists():
        return str(from_jsonl_dir)
    return str(candidate.resolve())


def load_jsonl_records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sanitize_speaker_encoder_config(config_dict: dict | None) -> dict | None:
    if config_dict is None:
        return None
    allowed_keys = {
        "mel_dim",
        "enc_dim",
        "enc_channels",
        "enc_kernel_sizes",
        "enc_dilations",
        "enc_attention_channels",
        "enc_res2net_scale",
        "enc_se_channels",
        "sample_rate",
    }
    return {key: value for key, value in dict(config_dict).items() if key in allowed_keys}


def checkpoint_has_speaker_encoder(model_path: Path) -> bool:
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        return False
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        return any(key.startswith("speaker_encoder.") for key in handle.keys())


def load_speaker_encoder(model_path: Path, runtime: dict) -> Qwen3TTSSpeakerEncoder:
    config = Qwen3TTSConfig.from_pretrained(str(model_path))
    speaker_encoder_config = config.speaker_encoder_config
    if speaker_encoder_config is None:
        raw_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        source_model_path = raw_config.get("speaker_encoder_source_model_path")
        if source_model_path:
            source_config = Qwen3TTSConfig.from_pretrained(str(Path(source_model_path)))
            speaker_encoder_config = source_config.speaker_encoder_config
    if speaker_encoder_config is None:
        raise ValueError(f"No speaker_encoder_config available in {model_path}")

    speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_encoder_config)
    speaker_state = {}
    with safe_open(str(model_path / "model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("speaker_encoder."):
                speaker_state[key.removeprefix("speaker_encoder.")] = handle.get_tensor(key)
    if not speaker_state:
        raise ValueError(f"No speaker_encoder weights found in {model_path}")
    speaker_encoder.load_state_dict(speaker_state)
    speaker_encoder = speaker_encoder.to(device=runtime["device"], dtype=runtime["dtype"])
    speaker_encoder.eval()
    return speaker_encoder


def resolve_speaker_encoder_source(init_model_path: Path, speaker_encoder_model_path: Path | None) -> Path | None:
    if speaker_encoder_model_path is not None and speaker_encoder_model_path.exists():
        return speaker_encoder_model_path
    guessed = Path(str(init_model_path).replace("CustomVoice", "Base"))
    if guessed != init_model_path and guessed.exists():
        return guessed
    return init_model_path if init_model_path.exists() else None


def resolve_voicebox_speaker_encoder(qwen3tts: Qwen3TTSModel, init_model_path: Path, runtime: dict, speaker_encoder_model_path: Path | None):
    embedded = getattr(qwen3tts.model, "speaker_encoder", None)
    if embedded is not None:
        embedded = embedded.to(device=runtime["device"], dtype=runtime["dtype"])
        embedded.eval()
        return embedded
    if checkpoint_has_speaker_encoder(init_model_path):
        return load_speaker_encoder(init_model_path, runtime)
    speaker_source = resolve_speaker_encoder_source(init_model_path, speaker_encoder_model_path)
    if speaker_source is None:
        raise ValueError("A speaker encoder source model is required.")
    return load_speaker_encoder(speaker_source, runtime)


def resolve_output_speaker_id(talker_config: dict, speaker_name: str) -> int:
    spk_id_map = dict(talker_config.get("spk_id", {}) or {})
    if speaker_name in spk_id_map:
        return int(spk_id_map[speaker_name])
    if not spk_id_map:
        return 3000
    return max(int(value) for value in spk_id_map.values()) + 1


def voicebox_metadata(*, source_checkpoint: Path, speaker_encoder_included: bool, speaker_encoder_source_path: str | None) -> dict:
    speaker_encoder_config = None
    if speaker_encoder_source_path:
        source_config = Qwen3TTSConfig.from_pretrained(str(Path(speaker_encoder_source_path)))
        speaker_encoder_config = source_config.speaker_encoder_config
        if speaker_encoder_config is not None and hasattr(speaker_encoder_config, "to_dict"):
            speaker_encoder_config = speaker_encoder_config.to_dict()
        speaker_encoder_config = sanitize_speaker_encoder_config(speaker_encoder_config)
    return {
        "demo_model_family": "voicebox",
        "speaker_encoder_included": bool(speaker_encoder_included),
        "speaker_encoder_source_model_path": speaker_encoder_source_path,
        "speaker_encoder_config": speaker_encoder_config,
        "voicebox_source_checkpoint": str(source_checkpoint),
    }


def checkpoint_epoch(path: Path) -> int:
    name = path.name
    if name.startswith("checkpoint-epoch-"):
        try:
            return int(name.split("-")[-1])
        except ValueError:
            return -1
    return -1


def finalize_checkpoint_layout(output_model_path: Path) -> Path:
    checkpoints = [path for path in output_model_path.glob("checkpoint-epoch-*") if path.is_dir()]
    if not checkpoints:
        final_dir = output_model_path / "final"
        return final_dir if final_dir.exists() else output_model_path
    latest = max(checkpoints, key=checkpoint_epoch)
    final_dir = output_model_path / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(latest, final_dir)
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)
    return final_dir


def train_customvoice_family_model(
    *,
    train_jsonl_path: str,
    init_model_path: str,
    output_dir: str,
    speaker_name: str,
    batch_size: int,
    lr: float,
    num_epochs: int,
    speaker_encoder_model_path: str = "",
    embed_speaker_encoder: bool = False,
    unique_id=None,
):
    runtime = resolve_training_runtime()
    init_model_dir = Path(init_model_path).resolve()
    output_model_path = Path(output_dir).resolve()
    train_jsonl = Path(train_jsonl_path).resolve()
    speaker_encoder_path = Path(speaker_encoder_model_path).resolve() if speaker_encoder_model_path.strip() else None

    if not init_model_dir.exists():
        raise ValueError(f"Init model not found: {init_model_dir}")
    if not train_jsonl.exists():
        raise ValueError(f"Training JSONL not found: {train_jsonl}")
    if speaker_encoder_path is not None and not speaker_encoder_path.exists():
        raise ValueError(f"Speaker encoder source not found: {speaker_encoder_path}")

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    send_progress_text(unique_id, "Loading training model...")

    qwen3tts = load_qwen_or_voicebox_model(
        str(init_model_dir),
        device_map=runtime["device"],
        dtype=runtime["dtype"],
        attn_implementation=runtime["attention"],
    )
    config = AutoConfig.from_pretrained(str(init_model_dir))
    auxiliary_speaker_encoder = resolve_voicebox_speaker_encoder(
        qwen3tts=qwen3tts,
        init_model_path=init_model_dir,
        runtime=runtime,
        speaker_encoder_model_path=speaker_encoder_path,
    )

    train_data = load_jsonl_records(train_jsonl)
    for row in train_data:
        if "audio" in row:
            row["audio"] = resolve_jsonl_audio_path(str(row["audio"]), train_jsonl)
        if "ref_audio" in row:
            row["ref_audio"] = resolve_jsonl_audio_path(str(row["ref_audio"]), train_jsonl)

    train_data.sort(key=lambda row: len(row.get("audio_codes", [])))
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    optimizer_name = (os.getenv("QWEN_DEMO_OPTIMIZER") or "adamw").strip().lower()
    if optimizer_name == "adafactor":
        optimizer = Adafactor(
            qwen3tts.model.parameters(),
            lr=lr,
            weight_decay=0.01,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        optimizer_kwargs = {"lr": lr, "weight_decay": 0.01}
        if runtime["device"].type == "cuda":
            optimizer_kwargs["fused"] = True
        optimizer = AdamW(qwen3tts.model.parameters(), **optimizer_kwargs)

    gradient_accumulation_steps = max(1, int(os.getenv("QWEN_DEMO_GRAD_ACCUM_STEPS", "1")))
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=runtime["mixed_precision"],
    )
    model, optimizer, train_dataloader = accelerator.prepare(qwen3tts.model, optimizer, train_dataloader)
    model.train()

    target_speaker_embedding = None
    checkpoint_embeds_encoder = checkpoint_has_speaker_encoder(init_model_dir)
    speaker_source = resolve_speaker_encoder_source(init_model_dir, speaker_encoder_path)
    log_every = max(1, int(os.getenv("QWEN_DEMO_LOG_EVERY", "10")))

    for epoch in range(num_epochs):
        send_progress_text(unique_id, f"Epoch {epoch + 1}/{num_epochs} training...")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                current_model = accelerator.unwrap_model(model)
                speaker_encoder = getattr(current_model, "speaker_encoder", None) or auxiliary_speaker_encoder
                model_device = next(current_model.parameters()).device
                model_dtype = next(current_model.parameters()).dtype
                speaker_embedding = speaker_encoder(ref_mels.to(model_device).to(model_dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]
                input_text_embedding = current_model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = current_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding
                input_embeddings = input_text_embedding + input_codec_embedding

                for index in range(1, 16):
                    codec_i_embedding = current_model.talker.code_predictor.get_input_embeddings()[index - 1](codec_ids[:, :, index])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = current_model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]
                _, sub_talker_loss = current_model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % log_every == 0:
                status = f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                print(status)
                send_progress_text(unique_id, status)

        if accelerator.is_main_process:
            checkpoint_dir = output_model_path / f"checkpoint-epoch-{epoch}"
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            shutil.copytree(str(init_model_dir), str(checkpoint_dir), dirs_exist_ok=True)

            config_dict = json.loads((init_model_dir / "config.json").read_text(encoding="utf-8"))
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = dict(config_dict.get("talker_config", {}) or {})
            spk_id_map = dict(talker_config.get("spk_id", {}) or {})
            spk_is_dialect = dict(talker_config.get("spk_is_dialect", {}) or {})
            speaker_id = resolve_output_speaker_id(talker_config, speaker_name)
            spk_id_map[speaker_name] = speaker_id
            spk_is_dialect[speaker_name] = False
            talker_config["spk_id"] = spk_id_map
            talker_config["spk_is_dialect"] = spk_is_dialect
            config_dict["talker_config"] = talker_config

            if embed_speaker_encoder:
                config_dict.update(
                    voicebox_metadata(
                        source_checkpoint=init_model_dir,
                        speaker_encoder_included=True,
                        speaker_encoder_source_path=str(init_model_dir if checkpoint_embeds_encoder else speaker_source) if speaker_source else None,
                    )
                )
            else:
                config_dict.pop("demo_model_family", None)
                config_dict.pop("speaker_encoder_included", None)
                config_dict.pop("speaker_encoder_source_model_path", None)
                config_dict.pop("speaker_encoder_config", None)
                config_dict.pop("voicebox_source_checkpoint", None)

            (checkpoint_dir / "config.json").write_text(
                json.dumps(config_dict, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {key: value.detach().to("cpu") for key, value in unwrapped_model.state_dict().items()}

            if embed_speaker_encoder:
                speaker_encoder_state = {
                    f"speaker_encoder.{key}": value.detach().to("cpu")
                    for key, value in auxiliary_speaker_encoder.state_dict().items()
                }
                state_dict.update(speaker_encoder_state)
            else:
                state_dict = {key: value for key, value in state_dict.items() if not key.startswith("speaker_encoder.")}

            codec_weight = state_dict["talker.model.codec_embedding.weight"]
            state_dict["talker.model.codec_embedding.weight"][speaker_id] = target_speaker_embedding[0].detach().to(codec_weight.dtype)
            save_file(state_dict, str(checkpoint_dir / "model.safetensors"))

    final_path = finalize_checkpoint_layout(output_model_path)
    send_progress_text(unique_id, f"Training complete: {final_path}")
    return str(final_path), speaker_name


def create_voicebox_checkpoint_internal(*, input_checkpoint: str, speaker_encoder_source: str, output_checkpoint: str) -> str:
    input_checkpoint_path = Path(input_checkpoint).resolve()
    speaker_source_path = Path(speaker_encoder_source).resolve()
    output_checkpoint_path = Path(output_checkpoint).resolve()

    if not input_checkpoint_path.exists():
        raise ValueError(f"Input checkpoint not found: {input_checkpoint_path}")
    if not speaker_source_path.exists():
        raise ValueError(f"Speaker encoder source not found: {speaker_source_path}")

    if output_checkpoint_path.exists():
        shutil.rmtree(output_checkpoint_path)
    shutil.copytree(input_checkpoint_path, output_checkpoint_path)

    state_dict = load_file(str(output_checkpoint_path / "model.safetensors"))
    speaker_source_state = load_file(str(speaker_source_path / "model.safetensors"))
    speaker_encoder_state = {
        key: value for key, value in speaker_source_state.items() if key.startswith("speaker_encoder.")
    }
    if not speaker_encoder_state:
        raise ValueError(f"No speaker_encoder weights found in {speaker_source_path}")
    state_dict.update(speaker_encoder_state)
    save_file(state_dict, str(output_checkpoint_path / "model.safetensors"))

    config = json.loads((output_checkpoint_path / "config.json").read_text(encoding="utf-8"))
    source_config = json.loads((speaker_source_path / "config.json").read_text(encoding="utf-8"))
    config["tts_model_type"] = "custom_voice"
    config["demo_model_family"] = "voicebox"
    config["speaker_encoder_included"] = True
    config["speaker_encoder_source_model_path"] = str(speaker_source_path)
    config["speaker_encoder_config"] = sanitize_speaker_encoder_config(source_config.get("speaker_encoder_config"))
    (output_checkpoint_path / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(output_checkpoint_path)
