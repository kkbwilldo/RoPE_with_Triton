# NVIDIA PyTorch container를 베이스 이미지로 사용합니다.
FROM nvcr.io/nvidia/pytorch:24.02-py3

# 워킹 디렉토리를 설정합니다.
WORKDIR /app

# vimrc를 복사하여 세팅합니다.
COPY vimrc /root/.vimrc

# Ninja의 job 세팅을 1로 설정하여 cpu oom을 방지합니다.
ENV MAX_JOBS=1

# Transformer Engine을 설치합니다.
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

