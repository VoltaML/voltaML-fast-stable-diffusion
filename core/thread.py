from threading import Thread
from typing import Dict, Optional, Tuple


class ThreadWithReturnValue(Thread):
    "A thread class that supports returning a value from the target function"

    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict] = None,
    ):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        super().__init__(group, target, name, args, kwargs)
        self._return = None
        self._err: Optional[Exception] = None

    def run(self):
        if self._target is not None:  # type: ignore
            try:
                self._return = self._target(*self._args, **self._kwargs)  # type: ignore
            except Exception as err:  # pylint: disable=broad-except
                self._err = err

    def join(self, *args):
        Thread.join(self, *args)
        return (self._return, self._err)
