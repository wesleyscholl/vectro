import onnx

_HAVE_ONNX: bool
_OPSET_VERSION: int
_OPSET_DOMAIN: str

def to_onnx_model(result: object) -> onnx.ModelProto: ...
def export_onnx(result: object, path: str) -> None: ...
