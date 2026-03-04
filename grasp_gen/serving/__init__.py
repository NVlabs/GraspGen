def __getattr__(name):
    if name == "GraspGenZMQServer":
        from grasp_gen.serving.zmq_server import GraspGenZMQServer
        return GraspGenZMQServer
    if name == "GraspGenClient":
        from grasp_gen.serving.zmq_client import GraspGenClient
        return GraspGenClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["GraspGenZMQServer", "GraspGenClient"]
