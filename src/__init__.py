# Package marker for vectro.src. The real Mojo-native module should be installed
# as `vectro.src.quantizer` when built; this file exists to make imports work
# during development and testing when the native module is not present.

__all__ = ["quantizer"]
