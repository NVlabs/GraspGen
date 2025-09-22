# Build base image
FROM nvcr.io/nvidia/pytorch:23.07-py3 AS base

# tmux is for debugging, osmesa is for rendering. Put all apt-get installs in this line
RUN apt update && apt-get install -y tmux libosmesa6-dev

RUN pip install --upgrade pip

# Install dependencies
RUN pip install h5py hydra-core matplotlib meshcat scikit-learn scipy tensorboard trimesh==4.5.3

# Install scene_synthesizer
RUN pip install scene-synthesizer[recommend]

RUN pip install torch==2.1.0 torchvision

# Install torch_cluster
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

RUN pip install imageio pickle5 opencv-python python-fcl

# Install pyrender
RUN pip install pyrender==0.1.45 pyglet==2.1.6 && pip install PyOpenGL==3.1.5

# Install pointnet2 modules
COPY pointnet2_ops pointnet2_ops
RUN pip install ./pointnet2_ops

# Diffusion dependencies
RUN pip install diffusers==0.11.1 timm==1.0.15
RUN pip install huggingface-hub==0.25.2

# PointTransformerV3 dependencies
RUN pip install addict yapf==0.40.1 tensorboardx sharedarray torch-geometric
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
RUN pip install spconv-cu120

# For the analytic model
RUN pip install qpsolvers[clarabel]

# This was not installed before
RUN pip install yourdfpy==0.0.56

# Not sure why this is needed again
RUN pip install trimesh==4.5.3 timm==1.0.15

# For dataset management and manifold
RUN pip install webdataset
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN pip install objaverse==0.1.7
RUN mkdir -p /install/
RUN cd /install/ && git clone --recursive -j8 https://github.com/hjwdzh/Manifold.git
RUN mkdir -p /install/Manifold/build
RUN cd /install/Manifold/build
RUN cd /install/Manifold/build && cmake .. -DCMAKE_BUILD_TYPE=Release
RUN cd /install/Manifold/build && make
ENV PATH="${PATH}:/install/Manifold/build/"

WORKDIR /code/