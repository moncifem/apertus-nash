# Reward service, apart from the trainer. Stdlib plus sympy (the exact structured-answer grader). Code, prompt and config are bind-mounted
# (compose: ./reward_service:/app:ro), so editing reward.py / reward_prompt.md / reward_config.json on the host updates the running reward.
FROM python:3.12-slim
RUN pip install --no-cache-dir sympy && useradd -m -u 10002 agent
COPY . /app
USER agent
WORKDIR /app
EXPOSE 8090
CMD ["python", "/app/server.py"]
