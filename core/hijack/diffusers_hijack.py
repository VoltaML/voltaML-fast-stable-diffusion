import importlib


def get_old_diffusers():
    "Returns the old diffusers module that can compile the TRT engine"

    return importlib.import_module("core.hijack.diffusers")


def get_new_diffusers():
    "Returns the new diffusers module that can do inference"

    return importlib.import_module("diffusers")
