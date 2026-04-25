import os

import folder_paths


QWEN3_TTS_MODELS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-TTS")
os.makedirs(QWEN3_TTS_MODELS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-TTS", QWEN3_TTS_MODELS_DIR)

QWEN3_TTS_PROMPTS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-TTS", "prompts")
os.makedirs(QWEN3_TTS_PROMPTS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-TTS-Prompts", QWEN3_TTS_PROMPTS_DIR)

QWEN3_TTS_MODELS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "Qwen3-TTS-12Hz-0.6B-Base",
}

QWEN3_TTS_TOKENIZERS = {
    "Qwen/Qwen3-TTS-Tokenizer-12Hz": "Qwen3-TTS-Tokenizer-12Hz",
}


def get_local_model_path(repo_id: str) -> str:
    folder_name = QWEN3_TTS_MODELS.get(repo_id) or QWEN3_TTS_TOKENIZERS.get(repo_id) or repo_id.replace("/", "_")
    return os.path.join(QWEN3_TTS_MODELS_DIR, folder_name)


def get_available_models() -> list:
    available = []
    for repo_id, folder_name in QWEN3_TTS_MODELS.items():
        local_path = os.path.join(QWEN3_TTS_MODELS_DIR, folder_name)
        if os.path.exists(local_path) and os.listdir(local_path):
            available.append(f"✓ {repo_id}")
        else:
            available.append(repo_id)
    return available
