import random
def random_crop_slice(input_shape, target_shape,rand=random):
    dim_pairs = list(zip(input_shape,target_shape))
    assert not any([s-c < 0 for s, c in dim_pairs])
    starts = [0 if s-c == 0 else rand.randrange(s-c) for s,c in dim_pairs]
    slices = [slice(s, s+c) for s, c in zip(starts, target_shape)]
    return slices
