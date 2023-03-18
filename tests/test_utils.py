import asyncio

from core.utils import convert_base64_to_bytes, get_grid_dimension, run_in_thread_async


def test_convert_base64_to_bytes():
    data = "aGVsbG8gd29ybGQ="
    assert convert_base64_to_bytes(data).read() == b"hello world"


def test_get_grid_dimension():
    assert get_grid_dimension(1) == (1, 1)
    assert get_grid_dimension(2) == (2, 1)
    assert get_grid_dimension(3) == (2, 2)
    assert get_grid_dimension(4) == (2, 2)
    assert get_grid_dimension(12) == (4, 3)
    assert get_grid_dimension(50) == (8, 7)


def test_run_in_thread_async():
    # async def coroutine():
    #     return 1

    def func():
        return 1

    assert asyncio.run(run_in_thread_async(func)) == 1
    # assert run_in_thread_async(coroutine) == 1
