import argparse
import sys
import traceback
from multiprocessing.connection import Connection, Listener
from types import ModuleType
from typing import Any

from docling.models.stages.ocr.paddleocr_vl_ipc import (
    ErrorResponse,
    InitializeRequest,
    PredictionResponse,
    PredictRequest,
    ReadyResponse,
    ShutdownRequest,
    StoppedResponse,
)


class _ModelScopeStub(ModuleType):
    """Keep ModelScope from importing PyTorch into the Paddle CUDA process."""

    __path__: list[str]

    def __init__(self) -> None:
        super().__init__("modelscope")
        self.__path__ = []

    def snapshot_download(self, *_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError(
            "ModelScope downloads are disabled in the isolated PaddleOCR worker. "
            "Use the default Hugging Face model source or local model directories."
        )


def _install_modelscope_stub() -> None:
    sys.modules["modelscope"] = _ModelScopeStub()


def _create_pipeline(pipeline_kwargs: dict[str, Any]) -> Any:
    # PaddleOCR imports ModelScope eagerly, and ModelScope imports PyTorch. Loading
    # both CUDA runtimes in one Windows process can bind incompatible CUDA DLLs.
    _install_modelscope_stub()

    # These optional, heavyweight imports must happen only inside the isolated
    # worker and Paddle must be loaded before any accidental PyTorch import.
    import paddle  # type: ignore[import-untyped]
    from paddleocr import PaddleOCRVL  # type: ignore[import-untyped]

    return PaddleOCRVL(**pipeline_kwargs)


def _send_error(connection: Connection, *, operation: str, error: Exception) -> None:
    connection.send(
        ErrorResponse(
            operation=operation,
            message=str(error),
            traceback=traceback.format_exc(),
        )
    )


def _predict(pipeline: Any, request: PredictRequest) -> PredictionResponse:
    raw_results = pipeline.predict(input=request.image)
    if raw_results is None:
        return PredictionResponse(results=[])

    results: list[dict[str, Any]] = []
    for result in raw_results:
        result_json = result.json
        if not isinstance(result_json, dict):
            raise TypeError(
                "PaddleOCR-VL returned a result whose json property is not a dict."
            )
        results.append(result_json)
    return PredictionResponse(results=results)


def _serve(connection: Connection) -> int:
    request = connection.recv()
    if not isinstance(request, InitializeRequest):
        connection.send(
            ErrorResponse(
                operation="initialize",
                message="The first worker request must initialize the pipeline.",
                traceback="",
            )
        )
        return 1

    try:
        pipeline = _create_pipeline(request.pipeline_kwargs)
    except Exception as error:
        _send_error(connection, operation="initialize", error=error)
        return 1

    connection.send(ReadyResponse())
    while True:
        try:
            request = connection.recv()
        except EOFError:
            return 0

        if isinstance(request, ShutdownRequest):
            connection.send(StoppedResponse())
            return 0
        if not isinstance(request, PredictRequest):
            connection.send(
                ErrorResponse(
                    operation="request",
                    message=f"Unsupported worker request: {type(request).__name__}.",
                    traceback="",
                )
            )
            continue

        try:
            connection.send(_predict(pipeline, request))
        except Exception as error:
            _send_error(connection, operation="predict", error=error)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--auth-key", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    with Listener(
        (args.host, args.port),
        family="AF_INET",
        authkey=bytes.fromhex(args.auth_key),
    ) as listener:
        with listener.accept() as connection:
            return _serve(connection)


if __name__ == "__main__":
    raise SystemExit(main())
