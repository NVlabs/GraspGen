# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .server import serve


def main():
    """MCP GraspGen Server — 6-DOF grasp generation for LLM tool-calling."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="MCP server that bridges LLM tool-calling to a GraspGen ZMQ inference server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="GraspGen ZMQ server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="GraspGen ZMQ server port (default: 5556)",
    )
    args = parser.parse_args()
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
