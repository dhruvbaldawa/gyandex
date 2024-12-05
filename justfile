test:
    pytest --cov=gyandex --cov-report html --cov-report term:skip-covered gyandex/

check:
    pyright && \
    ruff check && \
    ruff format
