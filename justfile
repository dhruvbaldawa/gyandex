test:
    pytest --cov=gyandex --cov-report html --cov-report term:skip-covered gyandex/

lint:
    ruff check
    ruff check --select I --fix
    ruff format
