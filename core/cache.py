import hashlib
import json
import os
import shutil

from .paths import get_local_model_path


def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def count_jsonl_lines(file_path: str) -> int:
    with open(file_path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_cache_metadata(meta_path: str) -> dict | None:
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("version") != 1:
            return None
        return metadata
    except (json.JSONDecodeError, IOError):
        return None


def save_cache_metadata(meta_path: str, metadata: dict) -> None:
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def migrate_cached_model(repo_id: str, target_path: str) -> bool:
    if os.path.exists(target_path) and os.listdir(target_path):
        return True

    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    hf_model_dir = os.path.join(hf_cache, f"models--{repo_id.replace('/', '--')}")
    if os.path.exists(hf_model_dir):
        snapshots_dir = os.path.join(hf_model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                source = os.path.join(snapshots_dir, snapshots[0])
                print(f"Migrating model from HuggingFace cache: {source} -> {target_path}")
                shutil.copytree(source, target_path, dirs_exist_ok=True)
                return True

    ms_cache = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
    ms_model_dir = os.path.join(ms_cache, repo_id.replace("/", os.sep))
    if os.path.exists(ms_model_dir):
        print(f"Migrating model from ModelScope cache: {ms_model_dir} -> {target_path}")
        shutil.copytree(ms_model_dir, target_path, dirs_exist_ok=True)
        return True

    return False


def download_model_to_comfyui(repo_id: str, source: str) -> str:
    target_path = get_local_model_path(repo_id)
    if migrate_cached_model(repo_id, target_path):
        print(f"Model available at: {target_path}")
        return target_path

    os.makedirs(target_path, exist_ok=True)

    if source == "ModelScope":
        from modelscope import snapshot_download

        print(f"Downloading {repo_id} from ModelScope to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    else:
        from huggingface_hub import snapshot_download

        print(f"Downloading {repo_id} from HuggingFace to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)

    return target_path
