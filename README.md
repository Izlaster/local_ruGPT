## Краткое описание  
Этот проект загружает модель ruGPT-3.5-13B, разбивает входной текст на чанки, извлекает из них список источников и сохраняет результат в файл. В Windows-среде мы рекомендуем ​Poetry​ для управления зависимостями и виртуальным окружением.  

---

## Требования  
- **Windows 10/11** (или новее) с правами на установку ПО.  
- **Python ≥ 3.10** (рекомендуется 3.9–3.11).  
- **Git** (для клонирования репозитория) или ZIP-архив проекта.  
- **NVIDIA-драйверы + CUDA** (если планируете inference на GPU).  

---

## 1. Установка Poetry  
1. Откройте PowerShell (от имени администратора не требуется).  
2. Запустите официальный скрипт установки:  
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```  
3. Перезапустите PowerShell и проверьте:  
   ```powershell
   poetry --version
   ```  
   Poetry кроссплатформен и одинаково поддерживает Windows, macOS и Linux  ([Introduction | Documentation | Poetry - Python dependency ...](https://python-poetry.org/docs/?utm_source=chatgpt.com)).  

---

## 2. Установка зависимостей через Poetry  
Инициализируйте виртуальное окружение и установите все пакеты:  
```powershell
poetry install
```  
Это автоматически создаст из pyproject.toml изолированное окружение и поставит:  
- transformers  
- torch (unified wheel, GPU-ready при наличии CUDA)
- bitsandbytes, accelerate, huggingface-hub, safetensors  

---

## 3. Проверка доступности GPU (опционально)  
Внутри активированного Poetry-shell запустите Python:
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
```
- Если `CUDA available: True` — inference пойдёт на GPU.  
- Иначе будет использоваться CPU.  

---

## 4. Подготовка входных данных  
- Поместите ваш текст в файл `input_txts/__doc2_.txt` (или измените константу `INPUT_FILE` в коде).  
- Создайте папку `models` рядом с `script.py`, если её нет.  

---

## 5. Запуск скрипта  
Находясь в активном Poetry-окружении:
```powershell
poetry run python script.py
```
- Скрипт автоматически скачает модель в `models/ruGPT-3.5-13B`, если её там нет.  
- Разобьёт текст на чанки, извлечёт источники и сохранит их в `sources.txt`.  

