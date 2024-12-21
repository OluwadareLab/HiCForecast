# Base image with Ubuntu 20.04 and CUDA 11.4 support
FROM nvidia/cuda:11.4.3-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    software-properties-common \
    dirmngr \
    gpg-agent \
    python3.8 \
    python3.8-venv \
    python3.8-distutils \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --config python3

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Add CRAN repository for R and install R 4.3.1
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 'E298A3A825C0D65DFD57CBB651716619E084DAB9' && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/' && \
    apt-get update && \
    apt-get install -y r-base=4.3.1-* && \
    rm -rf /var/lib/apt/lists/*

# Install necessary R packages (optional, customize as needed)
RUN Rscript -e "install.packages('remotes', repos='https://cloud.r-project.org')"

# Clone the Git repository
ARG REPO_URL=https://github.com/OluwadareLab/HiCForecast.git
RUN git clone $REPO_URL /workspace

# Set the working directory
WORKDIR /workspace

# Install Python dependencies from requirements.txt
RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi

# Set the directory as home
ENV HOME=/workspace

# Default command
CMD ["/bin/bash"]
