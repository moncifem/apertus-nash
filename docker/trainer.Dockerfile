# Policy trainer image: Swiss AI Apertus 1.5 vLLM/Transformers forks + PEFT + Math-Verify.
# Pin BASE_IMAGE to a digest (ghcr.io/...@sha256:...) once you have a working build.
ARG BASE_IMAGE=ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64
FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src PIP_DISABLE_PIP_VERSION_CHECK=1

# Record the forks shipped by the base image, install our deps, and fail the build if pip
# replaced torch / transformers / vllm (it must never overwrite the Swiss forks).
COPY docker/fork_fingerprint.py /opt/rlaif/fork_fingerprint.py
RUN python3 /opt/rlaif/fork_fingerprint.py > /opt/rlaif/forks.before
COPY requirements.txt /opt/rlaif/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /opt/rlaif/requirements.txt \
 && python3 /opt/rlaif/fork_fingerprint.py > /opt/rlaif/forks.after \
 && diff /opt/rlaif/forks.before /opt/rlaif/forks.after \
 && python3 -m pip freeze > /opt/rlaif/pip-freeze.txt

COPY src /app/src
COPY prompts /app/prompts
COPY configs /app/configs
COPY tests /app/tests
COPY reward_service /app/reward_service
