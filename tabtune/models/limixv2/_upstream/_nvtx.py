try:
    import nvtx
except ImportError:
    class _NoopRange:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __call__(self, fn):
            return fn

    class _NvtxShim:
        @staticmethod
        def annotate(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return _NoopRange()

        @staticmethod
        def push_range(*args, **kwargs):
            return None

        @staticmethod
        def pop_range(*args, **kwargs):
            return None

    nvtx = _NvtxShim()
