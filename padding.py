from icecream import ic


def cal_padding(
    in_hw,
    out_hw,
    kernel_size,
    stride,
):
    return ((out_hw - 1) * stride - in_hw + kernel_size) / 2


in_hw = 7
out_hw = 7
kernel_size = 3
stride = 1
padding = cal_padding(in_hw, out_hw, kernel_size, stride)
ic(padding)
