VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
# Файл-метка для отслеживания загрузки
DOWNLOAD_FLAG = $(VENV)/.downloaded

.PHONY: all run deps clean test

all: run

# 1. run зависит от флага загрузки
run: $(DOWNLOAD_FLAG)
	@echo "🚀 Запуск программы..."
	$(PYTHON) main.py --video videos/video_2.mp4 --live

# 2. download теперь создает файл-метку после успеха
# Если файл $(DOWNLOAD_FLAG) уже есть, эта цель не будет выполняться
$(DOWNLOAD_FLAG): $(VENV)/.installed
	@echo "📥 Загрузка файлов (это произойдет один раз)..."
	$(PYTHON) scripts/download_videos.py
	@touch $(DOWNLOAD_FLAG)

# 3. Установка зависимостей
$(VENV)/.installed: requirements.txt | $(VENV)
	@if [ -f requirements.txt ]; then \
		echo "📦 Установка библиотек..."; \
		$(PIP) install -r requirements.txt; \
	else \
		echo "⚠️ requirements.txt не найден"; \
	fi
	@touch $(VENV)/.installed

$(VENV):
	@echo "🛠 Создание venv..."
	python3 -m venv $(VENV)

# Если нужно принудительно перекачать — просто удаляем флаг
redownload:
	rm -f $(DOWNLOAD_FLAG)
	$(MAKE) run

clean:
	rm -rf $(VENV)