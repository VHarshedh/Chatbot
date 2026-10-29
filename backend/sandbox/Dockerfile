FROM python:3.12-slim

# Install system dependencies, LAMMPS, and Quantum ESPRESSO
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    lammps \
    quantum-espresso \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m sandboxuser
USER sandboxuser
WORKDIR /home/sandboxuser

# Install python dependencies
RUN pip install --no-cache-dir \
    ase \
    numpy \
    pandas \
    matplotlib \
    scipy \
    pydantic-settings

# Temporary workspace for the sandbox
RUN mkdir -p /tmp/workspace

CMD ["/bin/bash"]
