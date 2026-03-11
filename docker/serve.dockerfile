# GraspGen ZMQ Inference Server
#
# This Dockerfile builds on top of the graspgen:latest base image
# (built via `bash docker/build.sh`) and adds ZMQ serving dependencies.
#
# Build standalone:
#   docker build -f docker/serve.dockerfile -t graspgen-server:latest .
#
# Or via compose:
#   docker compose -f docker/compose.serve.yml up --build

FROM graspgen:latest

RUN pip install pyzmq msgpack msgpack-numpy

COPY . /code
WORKDIR /code
RUN pip install -e .

EXPOSE 5556

ENTRYPOINT ["python", "tools/graspgen_server.py"]
CMD ["--gripper_config", "/models/checkpoints/graspgen_robotiq_2f_140.yml", "--port", "5556"]
