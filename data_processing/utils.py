import h5py
import numpy as np
from collections.abc import Iterable

SMPL_JOINT_MAPPER = lambda joints: joints[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]


def write_to_h5py(filename, data, img_chunk_size=64,
                compression='gzip'):

    imgs = data['imgs']
    H, W = imgs.shape[1:3]

    # TODO: any smarter way for this?
    redundants = ['index', 'img_path']
    img_to_chunk = ['imgs', 'bkgds', 'masks']
    img_to_keep_whole = ['sampling_masks']

    for r in redundants:
        if r in data:
            data.pop(r)

    chunk = (1, int(img_chunk_size**2),)
    whole = (1, H * W,)

    h5_file = h5py.File(filename, 'w')

    # store meta
    ds = h5_file.create_dataset('img_shape', (4,), np.int32)
    ds[:] = np.array([*imgs.shape])

    for k in data.keys():
        if not isinstance(data[k], Iterable):
            print(f'{k}: non-iterable')
            ds = h5_file.create_dataset(k, (), type(data[k]))
            ds[()] = data[k]
            continue

        d_shape = data[k].shape
        C = d_shape[-1]
        N = d_shape[0]
        if k in img_to_chunk or k in img_to_keep_whole:
            data_chunk = chunk + (C,) if k in img_to_chunk else whole + (C,)
            flatten_shape = (N, H * W, C)
            print(f'{k}: img to chunk in size {data_chunk}, flatten as {flatten_shape}')
            # flatten the image for faster indexing
            ds = h5_file.create_dataset(k, flatten_shape, data[k].dtype,
                                        chunks=data_chunk, compression=compression)
            for idx in range(N):
                ds[idx] = data[k][idx].reshape(*flatten_shape[1:])
            #ds[:] = data[k].reshape(*flatten_shape)
        elif k == 'img_paths':
            img_paths = data[k].astype('S')
            ds = h5_file.create_dataset(k, (len(img_paths),), img_paths.dtype)
            ds[:] = img_paths
        else:
            if np.issubdtype(data[k].dtype, np.floating):
                dtype = np.float32
            elif np.issubdtype(data[k].dtype, np.integer):
                dtype = np.int64
            else:
                raise NotImplementedError('Unknown datatype for key {k}: {data[k].dtype}')

            ds = h5_file.create_dataset(k, data[k].shape, dtype,
                                        compression=compression)
            ds[:] = data[k][:]
            print(f'{k}: data to store as {dtype}')
        pass

    h5_file.close()

