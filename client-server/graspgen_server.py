#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Start a GraspGen ZMQ inference server.

Usage:
    # Serve with a specific gripper checkpoint config:
    python client-server/graspgen_server.py --gripper_config /models/checkpoints/graspgen_franka_panda.yml

    # Custom port:
    python client-server/graspgen_server.py --gripper_config /models/checkpoints/graspgen_robotiq_2f_140.yml --port 5557

    # Bind to localhost only:
    python client-server/graspgen_server.py --gripper_config /models/checkpoints/graspgen_franka_panda.yml --host 127.0.0.1
"""

import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start a GraspGen ZMQ inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        required=True,
        help="Path to gripper configuration YAML file (e.g. checkpoints/graspgen_franka_panda.yml)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Address to bind the ZMQ socket (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="Port to bind the ZMQ socket (default: 5556)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    from grasp_gen.serving.zmq_server import GraspGenZMQServer

    server = GraspGenZMQServer(
        gripper_config=args.gripper_config,
        host=args.host,
        port=args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
