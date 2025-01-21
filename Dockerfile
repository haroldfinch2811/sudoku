FROM python:slim

RUN pip install uv

WORKDIR /app
COPY requirements.lock ./
RUN uv pip install --no-cache --system -r requirements.lock

COPY src .

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]