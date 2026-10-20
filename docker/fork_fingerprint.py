"""Print versions + install locations of the packages that must come from the base image."""
import importlib
import importlib.util

for name in ("torch", "transformers", "vllm", "triton"):
    try:
        m = importlib.import_module(name)
        print(name, getattr(m, "__version__", "?"), m.__file__)
    except Exception as e:  # noqa: BLE001
        print(name, "MISSING", type(e).__name__)
print("apertus1p5_in_transformers", importlib.util.find_spec("transformers.models.apertus1p5") is not None)
