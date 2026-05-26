"""Concrete VLM backends. The package-level factory `build_vlm` is in
`sci_fi_parser.vlm`; this subpackage just holds the implementations.
"""

from sci_fi_parser.vlm.backends.api import APIVLM

__all__ = ["APIVLM", "HFTransformersVLM"]


def __getattr__(name: str):
    if name == "HFTransformersVLM":
        from sci_fi_parser.vlm.backends.hf_transformers import HFTransformersVLM
        return HFTransformersVLM
    raise AttributeError(name)
