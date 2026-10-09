import logging
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Connection
from typing import Any

import numpy

from docling.models.stages.ocr.paddleocr_vl_ipc import (
    ErrorResponse,
    InitializeRequest,
    PredictionResponse,
    PredictRequest,
    ReadyResponse,
    ShutdownRequest,
    StoppedResponse,
)

_log = logging.getLogger(__name__)

_LOOPBACK_HOST = "127.0.0.1"
_CONNECT_TIMEOUT_SECONDS = 30.0
_INITIALIZE_TIMEOUT_SECONDS = 1800.0
_PREDICT_TIMEOUT_SECONDS = 600.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_WORKER_MODULE = "docling.models.stages.ocr.paddleocr_vl_worker"


class PaddleOcrVlWorkerError(RuntimeError):
    """Failure reported by the isolated PaddleOCR-VL process."""


@dataclass(frozen=True)
class PaddleOcrVlResult:
    """PaddleOCR-VL result returned from the isolated worker."""

    json: dict[str, Any]


def _allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((_LOOPBACK_HOST, 0))
        return int(probe.getsockname()[1])


def _creation_flags() -> int:
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return 0


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    if (
        environment.get("PADDLE_PDX_MODEL_SOURCE", "huggingface").lower()
        == "modelscope"
    ):
        _log.warning(
            "ModelScope requires PyTorch, so the isolated PaddleOCR worker will "
            "use Hugging Face as its model source instead."
        )
        environment["PADDLE_PDX_MODEL_SOURCE"] = "huggingface"
    return environment


def _start_worker(*, port: int, auth_key: bytes) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            _WORKER_MODULE,
            "--host",
            _LOOPBACK_HOST,
            "--port",
            str(port),
            "--auth-key",
            auth_key.hex(),
        ],
        stdin=subprocess.DEVNULL,
        close_fds=True,
        creationflags=_creation_flags(),
        env=_worker_environment(),
    )


def _connect_to_worker(
    process: subprocess.Popen[bytes],
    *,
    port: int,
    auth_key: bytes,
) -> Connection:
    deadline = time.monotonic() + _CONNECT_TIMEOUT_SECONDS
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise PaddleOcrVlWorkerError(
                f"PaddleOCR worker exited before accepting requests (code {return_code})."
            )
        try:
            return Client((_LOOPBACK_HOST, port), family="AF_INET", authkey=auth_key)
        except AuthenticationError as error:
            raise PaddleOcrVlWorkerError(
                "Could not authenticate the PaddleOCR worker connection."
            ) from error
        except OSError as error:
            last_error = error
            time.sleep(0.05)

    raise PaddleOcrVlWorkerError(
        "Timed out while connecting to the PaddleOCR worker."
    ) from last_error


def _terminate_worker(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)


def _finalize_worker(connection: Connection, process: subprocess.Popen[bytes]) -> None:
    connection.close()
    if process.poll() is None:
        process.terminate()


class PaddleOcrVlClient:
    """Proxy PaddleOCR-VL calls to a process with an isolated CUDA runtime."""

    def __init__(self, *, pipeline_kwargs: dict[str, Any]) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False

        auth_key = secrets.token_bytes(32)
        port = _allocate_loopback_port()
        self._process: subprocess.Popen[bytes] = _start_worker(
            port=port, auth_key=auth_key
        )
        connection: Connection | None = None
        try:
            connection = _connect_to_worker(self._process, port=port, auth_key=auth_key)
            self._connection: Connection = connection
            self._connection.send(InitializeRequest(pipeline_kwargs=pipeline_kwargs))
            response = self._receive(
                operation="initialize", timeout=_INITIALIZE_TIMEOUT_SECONDS
            )
            if not isinstance(response, ReadyResponse):
                raise PaddleOcrVlWorkerError(
                    "PaddleOCR worker returned an invalid initialization response."
                )
        except Exception:
            if connection is not None:
                connection.close()
            _terminate_worker(self._process)
            raise

        self._finalizer: weakref.finalize = weakref.finalize(
            self, _finalize_worker, self._connection, self._process
        )

    def _receive(self, *, operation: str, timeout: float) -> object:
        if not self._connection.poll(timeout):
            raise PaddleOcrVlWorkerError(
                f"PaddleOCR worker timed out during {operation}."
            )
        try:
            response = self._connection.recv()
        except EOFError as error:
            raise PaddleOcrVlWorkerError(
                f"PaddleOCR worker disconnected during {operation}."
            ) from error
        if isinstance(response, ErrorResponse):
            detail = (
                f"\nRemote traceback:\n{response.traceback}"
                if response.traceback
                else ""
            )
            raise PaddleOcrVlWorkerError(
                f"PaddleOCR worker failed during {response.operation}: "
                f"{response.message}{detail}"
            )
        return response

    def predict(self, *, inputs: list[numpy.ndarray]) -> list[PaddleOcrVlResult]:
        if not inputs:
            return []

        with self._lock:
            if self._closed:
                raise PaddleOcrVlWorkerError("PaddleOCR worker is already closed.")
            if self._process.poll() is not None:
                raise PaddleOcrVlWorkerError(
                    f"PaddleOCR worker exited with code {self._process.returncode}."
                )

            try:
                self._connection.send(PredictRequest(images=inputs))
                response = self._receive(
                    operation="predict", timeout=_PREDICT_TIMEOUT_SECONDS
                )
            except OSError as error:
                raise PaddleOcrVlWorkerError(
                    "Could not communicate with the PaddleOCR worker."
                ) from error

            if not isinstance(response, PredictionResponse):
                raise PaddleOcrVlWorkerError(
                    "PaddleOCR worker returned an invalid prediction response."
                )
            return [PaddleOcrVlResult(json=result) for result in response.results]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

            if self._process.poll() is None:
                try:
                    self._connection.send(ShutdownRequest())
                    response = self._receive(
                        operation="shutdown", timeout=_SHUTDOWN_TIMEOUT_SECONDS
                    )
                    if not isinstance(response, StoppedResponse):
                        _log.warning(
                            "PaddleOCR worker returned an invalid shutdown response."
                        )
                except (OSError, PaddleOcrVlWorkerError) as error:
                    _log.warning(
                        "Could not stop PaddleOCR worker gracefully: %s", error
                    )

            self._connection.close()
            _terminate_worker(self._process)
            self._finalizer.detach()
