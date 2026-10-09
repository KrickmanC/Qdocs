import gc
import sys
import weakref
from unittest.mock import Mock, patch

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter
from docling.pipeline.base_pipeline import BasePipeline
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.utils.accelerator_utils import release_accelerator_memory


def _cache_pipeline(converter: DocumentConverter, pipeline: BasePipeline) -> None:
    converter.initialized_pipelines[(SimplePipeline, "test-options")] = pipeline


def test_release_resources_closes_pipelines_and_accelerator_cache() -> None:
    converter = DocumentConverter(allowed_formats=[])
    pipeline = Mock(spec=BasePipeline)
    _cache_pipeline(converter, pipeline)

    with (
        patch("docling.document_converter.gc.collect") as collect,
        patch(
            "docling.document_converter.release_accelerator_memory"
        ) as release_accelerator,
    ):
        converter.release_resources()

    pipeline.close.assert_called_once_with()
    collect.assert_called_once_with()
    release_accelerator.assert_called_once_with()
    assert converter.initialized_pipelines == {}


def test_converter_finalizer_releases_uncached_server_converter() -> None:
    converter = DocumentConverter(allowed_formats=[])
    pipeline = Mock(spec=BasePipeline)
    _cache_pipeline(converter, pipeline)
    converter_ref = weakref.ref(converter)

    with patch("docling.document_converter.release_accelerator_memory"):
        del converter
        gc.collect()

    assert converter_ref() is None
    pipeline.close.assert_called_once_with()


def test_automatic_resource_release_runs_after_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF])
    pipeline = Mock(spec=BasePipeline)
    _cache_pipeline(converter, pipeline)
    input_document = Mock(format=InputFormat.PDF)
    monkeypatch.setattr(settings.resources, "release_after_conversion", True)

    with (
        patch.object(
            converter,
            "_execute_pipeline",
            side_effect=RuntimeError("conversion failed"),
        ),
        patch("docling.document_converter.gc.collect"),
        patch("docling.document_converter.release_accelerator_memory"),
        pytest.raises(RuntimeError, match="conversion failed"),
    ):
        converter._process_document(input_document, raises_on_error=True)

    pipeline.close.assert_called_once_with()
    assert converter.initialized_pipelines == {}


def test_release_accelerator_memory_uses_loaded_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in (
        "torch.cuda",
        "torch.mps",
        "torch.xpu",
        "paddle",
        "mlx.core",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    cuda = Mock()
    cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda)

    release_accelerator_memory()

    cuda.empty_cache.assert_called_once_with()
