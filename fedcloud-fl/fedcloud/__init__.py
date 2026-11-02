"""fedcloud: federated fraud detection across cloud nodes (sync FedAvg/FedProx via Flower, async FedAsync)."""
import os
import sys

# Python >= 3.14 compatibility fix.
# Flower 1.18 requires protobuf<5, whose compiled "upb" extension crashes on Python 3.14
# with "TypeError: Metaclasses with custom tp_new are not supported". protobuf only falls
# back to its pure-Python backend if importing that extension raises ImportError, so we
# block that one import (before Flower/protobuf are loaded) and select the Python backend.
if sys.version_info >= (3, 14):
    import importlib.abc

    class _BlockUpb(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "google._upb._message":
                raise ImportError("protobuf upb backend disabled on Python 3.14 (fedcloud)")
            return None

    sys.meta_path.insert(0, _BlockUpb())
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

__version__ = "1.0.2"
