import torch


def torch_dtype_from_str(dtype: str):
    return torch.__dict__.get(dtype, None)


def get_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"] for d in y[i]._attrs["shape"]]
        print("AIT output_{} shape: {}".format(i, y_shape))
