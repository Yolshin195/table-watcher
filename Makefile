VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Главная команда
run: $(VENV)/.installed
	$(PYTHON) main.py

# Создание venv (если нет)
$(VENV):
	python3 -m venv $(VENV)

# Установка зависимостей (если есть requirements.txt)
$(VENV)/.installed: | $(VENV)
	@if [ -f requirements.txt ]; then \
		echo "📦 Установка зависимостей..."; \
		$(PIP) install -r requirements.txt; \
	else \
		echo "⚠️ requirements.txt не найден, пропускаю"; \
	fi
	touch $(VENV)/.installed

# Принудительная переустановка зависимостей
deps:
	rm -f $(VENV)/.installed
	$(MAKE) run


# Удалить окружение
clean:
	rm -rf $(VENV)


test:
	$(PYTHON) -m pytest tests -v