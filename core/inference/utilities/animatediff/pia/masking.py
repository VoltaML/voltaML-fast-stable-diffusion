from typing import List

# fmt: off
RANGE_LIST = [
        [1.0, 0.9, 0.85, 0.85, 0.85, 0.8], # 0 Small Motion
        [1.0, 0.8, 0.8, 0.8, 0.79, 0.78, 0.75], # Moderate Motion
        [1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.5], # Large Motion
        [1.0, 0.7, 0.65, 0.65, 0.6, 0.6, 0.6, 0.55, 0.5, 0.5, 0.45, 0.45, 0.4], # ULTRA Large Motion
        [1.0 , 0.9 , 0.85, 0.85, 0.85, 0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.8 , 0.85, 0.85, 0.9 , 1.0 ], # Loop
        [1.0 , 0.8 , 0.8 , 0.8 , 0.79, 0.78, 0.75, 0.75, 0.75, 0.75, 0.75, 0.78, 0.79, 0.8 , 0.8 , 1.0 ], # Loop
        [1.0 , 0.8 , 0.7 , 0.7 , 0.7 , 0.7 , 0.6 , 0.5 , 0.5 , 0.6 , 0.7 , 0.7 , 0.7 , 0.7 , 0.8 , 1.0 ], # Loop
        [1.0 , 0.7 , 0.6 , 0.6 , 0.6 , 0.6 , 0.5 , 0.4 , 0.4 , 0.5 , 0.6 , 0.6 , 0.6 , 0.6 , 0.7 , 1.0 ], # Loop
        [0.4, 0.1], # Style Transfer ULTRA Large Motion
        [0.5, 0.2], # Style Transfer Large Motion
        [0.5, 0.4, 0.4, 0.4, 0.35, 0.35, 0.3, 0.25, 0.2], # Style Transfer Moderate Motion
        [0.5, 0.4, 0.4, 0.4, 0.35, 0.3], # Style Transfer Candidate Small Motion
]
# fmt: on


def prepare_mask_coef(
    video_length: int, cond_frame: int, sim_range: List[float] = [0.2, 1.0]
):
    assert (
        len(sim_range) == 2
    ), "sim_range should has the length of 2, including the min and max similarity"

    assert video_length > 1, "video_length should be greater than 1"

    assert video_length > cond_frame, "video_length should be greater than cond_frame"

    diff = abs(sim_range[0] - sim_range[1]) / (video_length - 1)
    coef = [1.0] * video_length
    for f in range(video_length):
        f_diff = diff * abs(cond_frame - f)
        f_diff = 1 - f_diff
        coef[f] *= f_diff

    return coef


def prepare_mask_coef_by_statistics(video_length: int, cond_frame: int, sim_range: int):
    assert video_length > 0, "video_length should be greater than 0"

    assert video_length > cond_frame, "video_length should be greater than cond_frame"

    range_list = RANGE_LIST

    assert sim_range < len(range_list), f"sim_range type{sim_range} not implemented"

    coef = range_list[sim_range]
    coef = coef + ([coef[-1]] * (video_length - len(coef)))

    order = [abs(i - cond_frame) for i in range(video_length)]
    coef = [coef[order[i]] for i in range(video_length)]

    return coef
