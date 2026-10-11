# Independent evaluator container: one physical GPU is exposed to both CUDA and Vulkan.
FROM ubuntu:22.04
ENV NVIDIA_DRIVER_CAPABILITIES=all NVIDIA_VISIBLE_DEVICES=all
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates build-essential libatomic1 libegl1 libgl1 libglu1-mesa libglx0 libgomp1 libsm6 libxi6 libxrandr2 libxt6 libglib2.0-0 libnghttp2-14 libnss3 libxkbcommon0 libxcb-cursor0 libvulkan1 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}' > /etc/vulkan/icd.d/nvidia_icd.json && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ADD behavior-env.tar.gz /opt/behavior-env/
RUN /opt/behavior-env/bin/python /opt/behavior-env/bin/conda-unpack
COPY BEHAVIOR-1K /behavior-src
ENV PATH=/opt/behavior-env/bin:$PATH PYTHONPATH=/behavior-src/OmniGibson:/behavior-src/bddl3:/behavior-src/joylo OMNIGIBSON_DATA_PATH=/data OMNIGIBSON_APPDATA_PATH=/cache/appdata OMNIGIBSON_GPU_ID=0 OMNIGIBSON_HEADLESS=1 VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json PYTHONUNBUFFERED=1
RUN /opt/behavior-env/bin/python -m pip install --no-cache-dir --no-deps --ignore-installed packaging==25.0 && /opt/behavior-env/bin/python -c "import setuptools; import torch.utils.cpp_extension; print('BUILD_IMPORTS_PASS')"
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /behavior-src/OmniGibson
ENTRYPOINT ["/opt/behavior-env/bin/python", "-m", "omnigibson.eval.eval"]
