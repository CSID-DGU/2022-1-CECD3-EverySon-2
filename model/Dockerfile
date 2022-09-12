FROM pytorch/pytorch

# 한글 설치
RUN apt-get update && apt-get install -y locales git
RUN localedef -f UTF-8 -i ko_KR ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8
ENV PYTHONIOENCODING=utf-8

RUN mkdir opt/code
WORKDIR opt/code

# 코드 연동
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Switch to bash shell
SHELL ["/bin/bash", "-c"]

# Set ${CONDA_ENV_NAME} to default virutal environment
RUN  ~/.bashrc