import json
from pathlib import Path

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
from transformers import AutoConfig, AutoModel, AutoProcessor


def is_voicebox_checkpoint_dir(model_path: str | Path) -> bool:
    try:
        config_path = Path(model_path) / "config.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(raw.get("demo_model_family") == "voicebox" and raw.get("speaker_encoder_included"))


def load_voicebox_model(pretrained_model_name_or_path: str, **kwargs):
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    runtime_tts_model_type = config.tts_model_type
    config.tts_model_type = "base"

    model = AutoModel.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
    if not isinstance(model, Qwen3TTSForConditionalGeneration):
        raise TypeError(f"Expected Qwen3TTSForConditionalGeneration, got {type(model)}")

    model.tts_model_type = runtime_tts_model_type
    model.config.tts_model_type = runtime_tts_model_type
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True)
    return Qwen3TTSModel(model=model, processor=processor, generate_defaults=model.generate_config)


def load_qwen_or_voicebox_model(pretrained_model_name_or_path: str, **kwargs):
    checkpoint_dir = Path(pretrained_model_name_or_path)
    if checkpoint_dir.is_dir() and is_voicebox_checkpoint_dir(checkpoint_dir):
        return load_voicebox_model(str(checkpoint_dir), **kwargs)
    return Qwen3TTSModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
