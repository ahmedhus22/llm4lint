# syntax=docker/dockerfile:1
FROM python:3-bookworm

# Environment setup
COPY . /app
WORKDIR /app
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN pip install ollama

# ollama model creation
WORKDIR /app/llm4lint
RUN ollama create llm4lint7b -f ../models/q4_gguf/Modelfile

# What the container should run when it is started.
ENTRYPOINT [ "python", "llm4lint.py" ]
