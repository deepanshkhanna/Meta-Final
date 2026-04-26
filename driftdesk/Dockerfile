# Use PyTorch base image — torch/cuda pre-installed
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Pin EXACT versions; use --no-deps for ML stack to bypass pip's resolver
# backtracking storm (which hung the previous build for 1h+).
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    pydantic==2.9.2 \
    pydantic-core==2.23.4 \
    websocket-client==1.8.0 \
    websockets==15.0.1 \
    requests==2.32.3

RUN pip install --no-cache-dir --no-deps \
    transformers==4.46.3 \
    trl==0.12.2 \
    peft==0.13.2 \
    accelerate==1.0.1 \
    bitsandbytes==0.44.1 \
    datasets==3.0.2 \
    huggingface_hub==0.26.5 \
    safetensors==0.4.5 \
    sentencepiece==0.2.0 \
    tokenizers==0.20.3 \
    regex==2024.11.6 \
    tqdm==4.67.0 \
    pyarrow==17.0.0 \
    pandas==2.2.3 \
    python-dateutil==2.9.0.post0 \
    fsspec==2024.6.1 \
    filelock==3.16.1 \
    multiprocess==0.70.16 \
    dill==0.3.8 \
    xxhash==3.5.0 \
    aiohttp==3.10.10 \
    psutil==6.0.0 \
    packaging==24.1 \
    pyyaml==6.0.2 \
    rich==13.9.4

RUN pip install --no-cache-dir --no-deps openenv-core==0.2.2

COPY . .

# Install the driftdesk package so absolute imports (driftdesk.models, etc.) resolve.
RUN pip install --no-cache-dir --no-build-isolation -e .

EXPOSE 7860
ENV ENABLE_WEB_INTERFACE=false

COPY run_all.sh /app/run_all.sh
RUN chmod +x /app/run_all.sh
CMD ["/app/run_all.sh"]
