FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends python2.7

RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py && \
    python2.7 get-pip.py && \
    rm get-pip.py

# Copy requirements.txt to the container
COPY requirements.txt /tmp/requirements.txt

# Install requirements
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements.txt

COPY utils/requirements.txt /tmp/requirements.txt

RUN pip2.7 --disable-pip-version-check --no-cache-dir install -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements.txt
