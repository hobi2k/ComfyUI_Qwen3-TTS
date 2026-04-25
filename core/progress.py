from server import PromptServer


def send_progress_text(unique_id, text: str) -> None:
    if unique_id:
        PromptServer.instance.send_progress_text(text, unique_id)
