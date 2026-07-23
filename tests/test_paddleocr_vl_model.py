from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.pipeline_options import PaddleOcrVlOptions
from docling.models.factories import get_ocr_factory
from docling.models.stages.ocr import paddleocr_vl_model
from docling.models.stages.ocr.paddleocr_vl_model import PaddleOcrVlModel


def test_paddleocr_vl_backend_is_registered() -> None:
    factory = get_ocr_factory(allow_external_plugins=False)

    model = factory.create_instance(
        options=PaddleOcrVlOptions(),
        enabled=False,
        artifacts_path=None,
        accelerator_options=AcceleratorOptions(),
    )

    assert "paddleocr-vl-1.6" in factory.registered_kind
    assert isinstance(model, PaddleOcrVlModel)


def test_paddleocr_vl_initializes_fixed_pipeline_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakePaddleOcrVlClient:
        def __init__(self, *, pipeline_kwargs: dict[str, object]) -> None:
            captured.update(pipeline_kwargs)

    monkeypatch.setattr(paddleocr_vl_model, "PaddleOcrVlClient", FakePaddleOcrVlClient)

    model = PaddleOcrVlModel(
        enabled=True,
        artifacts_path=None,
        options=PaddleOcrVlOptions(
            engine="paddle",
            layout_detection_model_dir="layout-model",
            vl_rec_model_dir="vl-model",
            layout_threshold=0.65,
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_chart_recognition=True,
            use_seal_recognition=True,
            use_ocr_for_image_block=True,
            format_block_content=True,
        ),
        accelerator_options=AcceleratorOptions(device="cuda:2", num_threads=6),
    )

    assert isinstance(model.reader, FakePaddleOcrVlClient)
    assert captured == {
        "pipeline_version": "v1.6",
        "layout_detection_model_name": "PP-DocLayoutV3",
        "use_layout_detection": True,
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": True,
        "use_chart_recognition": True,
        "use_seal_recognition": True,
        "use_ocr_for_image_block": True,
        "format_block_content": True,
        "use_queues": False,
        "cpu_threads": 6,
        "engine": "paddle",
        "device": "gpu:2",
        "layout_detection_model_dir": "layout-model",
        "vl_rec_model_dir": "vl-model",
        "layout_threshold": 0.65,
    }


class _RecordingPaddleReader:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []

    def predict(self, *, input: np.ndarray) -> list[object]:
        self.images.append(input)
        return [
            SimpleNamespace(
                json={
                    "res": {
                        "parsing_res_list": [
                            {
                                "block_bbox": [2, 4, 10, 12],
                                "block_content": "  Recognized text  ",
                            },
                            {
                                "block_bbox": [0, 0, 2, 2],
                                "block_content": "",
                            },
                        ]
                    }
                }
            )
        ]


class _ImageBackend:
    def is_valid(self) -> bool:
        return True

    def get_page_image(self, *, scale: float, cropbox: BoundingBox) -> Image.Image:
        return Image.new(
            "RGB",
            (round(cropbox.width * scale), round(cropbox.height * scale)),
            (10, 20, 30),
        )


def test_paddleocr_vl_maps_ordered_blocks_to_ocr_cells() -> None:
    options = PaddleOcrVlOptions(scale=2)
    model = PaddleOcrVlModel.__new__(PaddleOcrVlModel)
    model.enabled = True
    model.options = options
    model.scale = options.scale
    reader = _RecordingPaddleReader()
    model.reader = reader

    ocr_rect = BoundingBox(
        l=5,
        t=7,
        r=15,
        b=17,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    model.get_ocr_rects = Mock(return_value=[ocr_rect])
    model.post_process_cells = Mock()

    page = Page(page_no=1, size=Size(width=20, height=20))
    page._backend = _ImageBackend()  # type: ignore[assignment]

    assert list(model(Mock(), [page])) == [page]
    assert len(reader.images) == 1
    assert reader.images[0].flags.c_contiguous
    assert reader.images[0][0, 0].tolist() == [30, 20, 10]

    cells = model.post_process_cells.call_args.args[0]
    assert len(cells) == 1
    assert cells[0].text == "Recognized text"
    assert cells[0].orig == "Recognized text"
    assert cells[0].from_ocr is True
    assert cells[0].rect.to_bounding_box().as_tuple() == (6.0, 9.0, 10.0, 13.0)
    assert model.post_process_cells.call_args.args[1] is page
