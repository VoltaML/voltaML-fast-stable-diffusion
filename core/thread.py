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

    def run(self):
        if self._target is not None:  # type: ignore
            self._return = self._target(*self._args, **self._kwargs)  # type: ignore

    def join(self, *args):
        Thread.join(self, *args)
        return self._return
