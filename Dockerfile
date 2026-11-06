# /warthadev-agent/Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent_config.py agent_runtime.py agent_main.py ./
EXPOSE 8788
CMD ["python", "agent_main.py"]
