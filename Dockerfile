FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASHARE_DATA_ROOT=/data/market/daily \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

COPY pyproject.toml README.md ./
COPY ashare_cross_section_similarity ./ashare_cross_section_similarity
RUN python -m pip install --upgrade pip && python -m pip install .

COPY streamlit_app.py ./streamlit_app.py

RUN mkdir -p /data/market/daily/qfq /app/outputs

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
