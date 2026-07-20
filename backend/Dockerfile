FROM python:3.14-slim

WORKDIR /backend

RUN apt-get update && apt-get install -y \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024"]
