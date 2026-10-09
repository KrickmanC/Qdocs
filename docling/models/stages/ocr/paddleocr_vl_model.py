import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions, PaddleOcrVlOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.stages.ocr.paddleocr_vl_client import PaddleOcrVlClient
from docling.utils.profiling import TimeIntervalRecorder

_log = logging.getLogger(__name__)

_PIPELINE_VERSION = "v1.6"
_LAYOUT_MODEL_NAME = "PP-DocLayoutV3"


def _paddle_device(device: str | AcceleratorDevice) -> str | None:
    """Translate Docling accelerator names to PaddleOCR device names."""
    value = device.value if isinstance(device, AcceleratorDevice) else device
    if value == AcceleratorDevice.AUTO.value:
        return None
    if value == AcceleratorDevice.CUDA.value:
        return "gpu:0"
    if value.startswith(f"{AcceleratorDevice.CUDA.value}:"):
        return f"gpu:{value.split(':', maxsplit=1)[1]}"
    if value == AcceleratorDevice.MPS.value:
        _log.warning("PaddleOCR-VL does not accept the Docling MPS device; using CPU.")
        return AcceleratorDevice.CPU.value
    return value


def _parse_bbox(value: object) -> tuple[float, float, float, float] | None:
    # The documented result.json interface converts NumPy boxes to JSON lists.
    if not isinstance(value, list | tuple):
        return None

    coordinates: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, int | float):
            return None
        coordinates.append(float(coordinate))

    if len(coordinates) != 4:
        return None

    left, top, right, bottom = coordinates
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _result_payload(result: Any) -> Mapping[str, Any] | None:
    result_json = result.json
    if not isinstance(result_json, Mapping):
        return None

    payload = result_json.get("res", result_json)
    return payload if isinstance(payload, Mapping) else None


def _text_cells_from_results(
    results: Iterable[Any],
    *,
    scale: float,
    offset_x: float,
    offset_y: float,
    start_index: int,
) -> list[TextCell]:
    cells: list[TextCell] = []
    for result in results:
        payload = _result_payload(result)
        if payload is None:
            _log.warning("PaddleOCR-VL returned a result with an invalid JSON payload.")
            continue

        blocks = payload.get("parsing_res_list")
        if not isinstance(blocks, list):
            _log.warning("PaddleOCR-VL returned a result without a parsing_res_list.")
            continue

        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            content = block.get("block_content")
            if not isinstance(content, str) or not content.strip():
                continue
            bbox = _parse_bbox(block.get("block_bbox"))
            if bbox is None:
                _log.warning("PaddleOCR-VL returned a text block with an invalid bbox.")
                continue

            left, top, right, bottom = bbox
            text = content.strip()
            cells.append(
                TextCell(
                    index=start_index + len(cells),
                    text=text,
                    orig=text,
                    from_ocr=True,
                    rect=BoundingRectangle.from_bounding_box(
                        BoundingBox.from_tuple(
                            coord=(
                                left / scale + offset_x,
                                top / scale + offset_y,
                                right / scale + offset_x,
                                bottom / scale + offset_y,
                            ),
                            origin=CoordOrigin.TOPLEFT,
                        )
                    ),
                )
            )
    return cells


@dataclass
class _PageOcrState:
    page: Page
    ocr_rects: list[BoundingBox]
    cells: list[TextCell]
    recorder: TimeIntervalRecorder | None


@dataclass
class _BufferedOcrInput:
    state: _PageOcrState
    ocr_rect: BoundingBox
    image: numpy.ndarray


class PaddleOcrVlModel(BaseOcrModel):
    """OCR adapter for the complete PaddleOCR-VL-1.6 document pipeline."""

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: PaddleOcrVlOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PaddleOcrVlOptions
        self.scale = options.scale
        self.page_batch_size = options.page_batch_size if self.enabled else 1
        self.reader: PaddleOcrVlClient | None = None

        if not self.enabled:
            return

        if self.options.lang:
            _log.debug(
                "Ignoring OCR languages %s because PaddleOCR-VL detects "
                "languages automatically.",
                self.options.lang,
            )

        init_kwargs: dict[str, Any] = {
            "pipeline_version": _PIPELINE_VERSION,
            "layout_detection_model_name": _LAYOUT_MODEL_NAME,
            "use_layout_detection": True,
            "use_doc_orientation_classify": self.options.use_doc_orientation_classify,
            "use_doc_unwarping": self.options.use_doc_unwarping,
            "use_chart_recognition": self.options.use_chart_recognition,
            "use_seal_recognition": self.options.use_seal_recognition,
            "use_ocr_for_image_block": self.options.use_ocr_for_image_block,
            "format_block_content": self.options.format_block_content,
            "use_queues": self.options.use_queues,
            "cpu_threads": accelerator_options.num_threads,
        }
        optional_kwargs = {
            "engine": self.options.engine,
            "device": _paddle_device(accelerator_options.device),
            "layout_detection_model_dir": self.options.layout_detection_model_dir,
            "vl_rec_model_dir": self.options.vl_rec_model_dir,
            "vl_rec_backend": self.options.vl_rec_backend,
            "vl_rec_server_url": self.options.vl_rec_server_url,
            "vl_rec_max_concurrency": self.options.vl_rec_max_concurrency,
            "vl_rec_api_model_name": self.options.vl_rec_api_model_name,
            "layout_threshold": self.options.layout_threshold,
        }
        init_kwargs.update(
            {key: value for key, value in optional_kwargs.items() if value is not None}
        )
        self.reader = PaddleOcrVlClient(pipeline_kwargs=init_kwargs)

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        states: list[_PageOcrState] = []
        buffered_inputs: list[_BufferedOcrInput] = []
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                states.append(
                    _PageOcrState(
                        page=page,
                        ocr_rects=[],
                        cells=[],
                        recorder=None,
                    )
                )
                continue

            recorder = TimeIntervalRecorder(conv_res, "ocr")
            recorder.resume()
            ocr_rects = self.get_ocr_rects(page)
            state = _PageOcrState(
                page=page,
                ocr_rects=ocr_rects,
                cells=[],
                recorder=recorder,
            )
            states.append(state)

            for ocr_rect in ocr_rects:
                if ocr_rect.area() == 0:
                    continue

                high_res_image = page._backend.get_page_image(
                    scale=self.scale, cropbox=ocr_rect
                ).convert("RGB")
                # PaddleX reads NumPy image inputs in BGR order.
                buffered_inputs.append(
                    _BufferedOcrInput(
                        state=state,
                        ocr_rect=ocr_rect,
                        image=numpy.ascontiguousarray(
                            numpy.asarray(high_res_image)[:, :, ::-1]
                        ),
                    )
                )
                del high_res_image
            recorder.pause()

        if buffered_inputs:
            images = [entry.image for entry in buffered_inputs]
            inference_started = time.monotonic()
            if self.reader is None:
                raise RuntimeError("The PaddleOCR-VL worker is closed.")
            results = self.reader.predict(inputs=images)
            inference_seconds = time.monotonic() - inference_started
            del images

            if len(results) != len(buffered_inputs):
                raise RuntimeError(
                    "PaddleOCR-VL returned "
                    f"{len(results)} results for {len(buffered_inputs)} input images."
                )

            seconds_per_input = inference_seconds / len(buffered_inputs)
            for entry, result in zip(buffered_inputs, results):
                assert entry.state.recorder is not None
                entry.state.recorder.add(seconds_per_input)
                entry.state.cells.extend(
                    _text_cells_from_results(
                        (result,),
                        scale=self.scale,
                        offset_x=entry.ocr_rect.l,
                        offset_y=entry.ocr_rect.t,
                        start_index=len(entry.state.cells),
                    )
                )
            buffered_inputs.clear()

        for state in states:
            if state.recorder is not None:
                state.recorder.resume()
                self.post_process_cells(state.cells, state.page)
                state.recorder.pause()
                state.recorder.close()

                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, state.page, state.ocr_rects)

            yield state.page

    def close(self) -> None:
        """Stop the isolated PaddleOCR-VL worker process."""
        if self.reader is None:
            return
        reader = self.reader
        self.reader = None
        reader.close()

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        return PaddleOcrVlOptions
