# datasets
from vae.datasets.panda import PandaPushVideo, PandaPushImage


def get_video_dataset(ds, root, seq_len=1, mode='train', image_size=128):
    # load data
    if ds == "panda":
        dataset = PandaPushVideo(root=root, image_size=image_size, mode=mode, sample_length=seq_len)
    else:
        raise NotImplementedError
    return dataset


def get_image_dataset(ds, root, mode='train', image_size=128, seq_len=1):
    # load data
    if ds == "panda":
        dataset = PandaPushImage(root=root, image_size=image_size, mode=mode, sample_length=seq_len)
    else:
        raise NotImplementedError
    return dataset
