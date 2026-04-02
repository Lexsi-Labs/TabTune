"""No-op telemetry shim — replaces tabpfn_common_utils.telemetry."""

def track_model_call(*args, **kwargs):
    """No-op decorator."""
    def decorator(func):
        return func
    if args and callable(args[0]):
        return args[0]
    return decorator

def set_model_config(*args, **kwargs):
    pass

def set_init_params(*args, **kwargs):
    pass

class interactive:
    @staticmethod
    def capture_session(*args, **kwargs):
        pass
    @staticmethod
    def ping(*args, **kwargs):
        pass
