from unittest.mock import Mock

import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from PIL import Image

from docling.datamodel.base_models import Page
from docling.datamodel.pipeline_options import EasyOcrOptions
from docling.models.stages.ocr.easyocr_model import EasyOcrModel


class _RecordingReader:
    def __init__(self) -> None:
        self.input_owns_data: bool | None = None
        self.kwargs: dict[str, object] = {}

    def readtext(self, image: np.ndarray, **kwargs: object) -> list[object]:
        self.input_owns_data = image.flags.owndata
        self.kwargs = kwargs
        return []


class _ImageBackend:
    def is_valid(self) -> bool:
        return True

    def get_page_image(self, *, scale: float, cropbox: BoundingBox) -> Image.Image:
        return Image.new(
            "RGB",
            (round(cropbox.width * scale), round(cropbox.height * scale)),
            "white",
        )


def test_easyocr_recognition_uses_bounded_memory_settings() -> None:
    model = EasyOcrModel.__new__(EasyOcrModel)
    model.enabled = True
    model.options = EasyOcrOptions()
    model.scale = 3
    reader = _RecordingReader()
    model.reader = reader

    ocr_rect = BoundingBox(
        l=0,
        t=0,
        r=10,
        b=10,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    model.get_ocr_rects = Mock(return_value=[ocr_rect])
    model.post_process_cells = Mock()

    page = Page(page_no=1, size=Size(width=10, height=10))
    page._backend = _ImageBackend()  # type: ignore[assignment]

    assert list(model(Mock(), [page])) == [page]
    assert reader.input_owns_data is False
    assert reader.kwargs == {
        "detail": 1,
        "paragraph": False,
        "batch_size": 1,
        "workers": 0,
    }
