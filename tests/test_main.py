import os

os.environ["TESTING"] = "1"

from main import checks, main  # noqa: E402


def test_checks():
    checks()


def test_main():
    main(exit_after_init=True)
