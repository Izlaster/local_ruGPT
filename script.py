from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch
import os

# Параметры
REPO_ID = "ai-forever/ruGPT-3.5-13B"
MODEL_PATH = "models/ruGPT-3.5-13B/models--ai-forever--ruGPT-3.5-13B/snapshots/64b115374b8f086ef13ccb8ba4f49d8076a53324/" # после скачивания нужно указать путь до чекпоинта 
INPUT_FILE = "input_txts/__doc2_.txt"
OUTPUT_FILE = "sources.txt"
MAX_TOKENS_PER_CHUNK = 1500
MAX_NEW_TOKENS = 200

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",
        offload_folder="offload",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

def chunk_text(full_text, tokenizer, max_tokens):
    token_ids = tokenizer.encode(full_text)
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, clean_up_tokenization_spaces=True))
    return chunks

def extract_sources_from_chunk(chunk, tokenizer, model):
    prompt = (
        "Извлеки из следующего текста список источников (автор и название публикации):\n"
        f"{chunk}\n\nСписок:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

def main():
    tokenizer, model = load_model(MODEL_PATH)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Не найден входной файл: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        article = f.read()

    chunks = chunk_text(article, tokenizer, MAX_TOKENS_PER_CHUNK)

    all_sources = set()
    for idx, chunk in enumerate(chunks, 1):
        print(f"Обрабатываю чанк {idx}/{len(chunks)}...")
        result = extract_sources_from_chunk(chunk, tokenizer, model)
        for line in result.splitlines():
            line = line.strip(" ‒–-—. ")
            if line:
                all_sources.add(line)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for src in sorted(all_sources):
            out.write(src + "\n")

    print(f"Готово! Найдено {len(all_sources)} уникальных источников. Сохранено в {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
