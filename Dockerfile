FROM mcr.microsoft.com/vscode/devcontainers/miniconda:3

# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

COPY environment.yml* /tmp/conda-tmp/
RUN conda env update -n base -f /tmp/conda-tmp/environment.yml \
    && rm -rf /tmp/conda-tmp
