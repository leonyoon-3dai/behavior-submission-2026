# The base digest is replaced with the actual pulled digest by build_submission_docker.py.
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libglib2.0-0 libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY runtime/python /opt/python
COPY runtime/site-packages /opt/site-packages
COPY runtime/packages.json /opt/runtime-packages.json
COPY app /app
COPY licenses /licenses
COPY context-manifest.json /opt/context-manifest.json
ENV PYTHONPATH=/app:/app/adapters:/opt/site-packages:/app/external/1st_behavior-1k-solution/src:/app/external/1st_behavior-1k-solution/openpi/src:/app/external/1st_behavior-1k-solution/openpi/packages/openpi-client/src:/app/external/expo-ft-rt \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD /opt/python/bin/python3.11 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=3)"
ENTRYPOINT ["/opt/python/bin/python3.11", "-m", "submission.server"]
CMD ["--manifest", "/models/model-manifest.json"]
