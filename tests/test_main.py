import os

os.environ["TESTING"] = "1"

from main import checks, main  # pylint: disable=wrong-import-position


def test_checks():
    checks()


def test_main():
    main(exit_after_init=True)
