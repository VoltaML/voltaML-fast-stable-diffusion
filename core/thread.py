from asyncio import ensure_future
from threading import Thread
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple, TypeVar, Union

from core.shared import asyncio_loop

T = TypeVar("T")


class ThreadWithReturnValue(Thread):
    "A thread class that supports returning a value from the target function"

    def __init__(
        self,
        target: Union[Callable[..., T], Coroutine[Any, Any, T]],
        group=None,
        name: Optional[str] = None,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict] = None,
    ):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        super().__init__(group, target, name, args, kwargs, daemon=True)  # type: ignore
        self._return: Any = None
        self._err: Optional[Exception] = None

    def run(self):
        target: Union[Callable, Coroutine] = self._target  # type: ignore
        if target is not None:  # type: ignore
            if isinstance(target, Callable):
                try:
                    self._return = target(*self._args, **self._kwargs)  # type: ignore
                except Exception as err:  # pylint: disable=broad-except
                    self._err = err
            else:
                try:
                    if asyncio_loop:
                        self._return = ensure_future(target, loop=asyncio_loop).result()
                    else:
                        raise Exception("Asyncio loop not found")
                except Exception as err:  # pylint: disable=broad-except
                    self._err = err

    def join(self, *args) -> Tuple[Any | None, Exception | None]:
        Thread.join(self, *args)
        return (self._return, self._err)
