


def get_inference(args):
    if args.sliding_window:
        from .inference3d import inference_sliding_window
        return inference_sliding_window

    else:
        from .inference3d import inference_whole_image
        return inference_whole_image
        



def split_idx(half_win, size, i):
    '''
    half_win: The size of half window
    size: img size along one axis
    i: the patch index
    '''

    start_idx = half_win * i
    end_idx = start_idx + half_win*2

    if end_idx > size:
        start_idx = size - half_win*2
        end_idx = size

    return start_idx, end_idx

