from concurrent.futures import ThreadPoolExecutor
from typing import cast

from docling.pipeline.standard_pdf_pipeline import ThreadedItem, ThreadedQueue


def test_threaded_queue_accumulates_a_batch_larger_than_queue_capacity() -> None:
    queue = ThreadedQueue(max_size=2)
    items = [cast(ThreadedItem, object()) for _ in range(3)]
    assert queue.put(items[0])
    assert queue.put(items[1])

    with ThreadPoolExecutor(max_workers=1) as executor:
        batch_future = executor.submit(queue.get_batch, 3, 1.0)
        assert queue.put(items[2], timeout=1.0)
        batch = batch_future.result(timeout=1.0)

    assert batch == items
