from .nodes import (
    Qwen3Loader,
    Qwen3CustomVoice,
    Qwen3VoiceDesign,
    Qwen3VoiceClone,
    Qwen3DirectedCloneFromVoiceDesign,
    Qwen3PromptMaker,
    Qwen3ClonePromptFromAudio,
    Qwen3SavePrompt,
    Qwen3LoadPrompt,
    Qwen3CustomVoiceFromPrompt,
    Qwen3DatasetFromFolder,
    Qwen3DataPrep,
    Qwen3FineTune,
    Qwen3AudioCompare
)

NODE_CLASS_MAPPINGS = {
    "Qwen3Loader": Qwen3Loader,
    "Qwen3CustomVoice": Qwen3CustomVoice,
    "Qwen3VoiceDesign": Qwen3VoiceDesign,
    "Qwen3VoiceClone": Qwen3VoiceClone,
    "Qwen3DirectedCloneFromVoiceDesign": Qwen3DirectedCloneFromVoiceDesign,
    "Qwen3PromptMaker": Qwen3PromptMaker,
    "Qwen3ClonePromptFromAudio": Qwen3ClonePromptFromAudio,
    "Qwen3SavePrompt": Qwen3SavePrompt,
    "Qwen3LoadPrompt": Qwen3LoadPrompt,
    "Qwen3CustomVoiceFromPrompt": Qwen3CustomVoiceFromPrompt,
    "Qwen3DatasetFromFolder": Qwen3DatasetFromFolder,
    "Qwen3DataPrep": Qwen3DataPrep,
    "Qwen3FineTune": Qwen3FineTune,
    "Qwen3AudioCompare": Qwen3AudioCompare
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3Loader": "Qwen3-TTS Loader",
    "Qwen3CustomVoice": "Qwen3-TTS Custom Voice",
    "Qwen3VoiceDesign": "Qwen3-TTS Voice Design",
    "Qwen3VoiceClone": "Qwen3-TTS Voice Clone",
    "Qwen3DirectedCloneFromVoiceDesign": "Qwen3-TTS Directed Clone From Voice Design",
    "Qwen3PromptMaker": "Qwen3-TTS Prompt Maker",
    "Qwen3ClonePromptFromAudio": "Qwen3-TTS Clone Prompt From Audio",
    "Qwen3SavePrompt": "Qwen3-TTS Save Prompt",
    "Qwen3LoadPrompt": "Qwen3-TTS Load Prompt",
    "Qwen3CustomVoiceFromPrompt": "Qwen3-TTS Custom Voice From Prompt",
    "Qwen3DatasetFromFolder": "Qwen3-TTS Dataset Maker",
    "Qwen3DataPrep": "Qwen3-TTS Data Prep",
    "Qwen3FineTune": "Qwen3-TTS Finetune",
    "Qwen3AudioCompare": "Qwen3-TTS Audio Compare"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
