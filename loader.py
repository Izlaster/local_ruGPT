import os
from huggingface_hub import snapshot_download

# Параметры для скачивания
REPO_ID = "ai-forever/ruGPT-3.5-13B"
REVISION = None  # Например, "main" или хеш коммита
DOWNLOAD_DIR = os.path.join("models", REPO_ID.replace("/", "--"))


def download_model(repo_id: str, download_dir: str, revision: str = None) -> str:
    os.makedirs(download_dir, exist_ok=True)
    print(f"Скачиваем модель {repo_id} в {download_dir}...")
    local_path = snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=download_dir,
        local_files_only=False
    )
    print(f"Модель сохранена в: {local_path}")
    return local_path


if __name__ == "__main__":
    # Запуск скачивания
    model_path = download_model(REPO_ID, DOWNLOAD_DIR, REVISION)
    print("Готово. Укажите MODEL_PATH в основном скрипте:")
    print(f"MODEL_PATH = \"{model_path}\"")
