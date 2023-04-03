FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
# COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
# RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
#     && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# Copy requirements.txt to the container
COPY requirements.txt /tmp/requirements.txt

# Install requirements
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements.txt

#ENV PYTHONPATH "${PYTHONPATH}:/workspaces/bha2/src"

# #Section extracted from neurodocker
# ENV CONDA_DIR="/opt/miniconda-latest" \
#     PATH="/opt/miniconda-latest/bin:$PATH"
# RUN export PATH="/opt/miniconda-latest/bin:$PATH" \
#     && echo "Downloading Miniconda installer ..." \
#     && conda_installer="/tmp/miniconda.sh" \
#     && curl -fsSL --retry 5 -o "$conda_installer" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && bash "$conda_installer" -b -p /opt/miniconda-latest \
#     && rm -f "$conda_installer" \
#     && conda update -yq -nbase conda \
#     && conda config --system --prepend channels conda-forge \
#     && conda config --system --set auto_update_conda false \
#     && conda config --system --set show_channel_urls true \
#     && sync && conda clean -y --all && sync \
#     && conda create -y -q --name craddock \
#     && conda install -y -q --name craddock \
#            "python=2.7" \
#     && sync && conda clean -y --all && sync \
#     && bash -c "source activate craddock \
#     &&   pip install --no-cache-dir  \
#              "scipy" \
#              "numpy" \
#              "pandas" \
#              "nibabel"" \
#     && rm -rf ~/.cache/pip/* \
#     && sync
