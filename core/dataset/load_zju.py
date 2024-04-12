import h5py
from core.dataset.dataset import BaseH5Dataset
from core.utils.skeleton_utils import *

h36m_zju_eval_frames = {
    'S1': np.arange(49),
    'S5': np.arange(127),
    'S6': np.arange(83),
    'S7': np.arange(200),
    'S8': np.arange(87),
    'S9': np.arange(2), #np.arange(133),
    'S11': np.arange(82),
}

num_train_frames = {
    'S1': [150, 0],
    'S5': [250, 0],
    'S6': [150, 0],
    'S7': [300, 0],
    'S8': [250, 0],
    'S9': [260, 0],
    'S11': [200, 0],
}
num_novel_view_frames = {
    'S1': [25, 0],
    'S5': [42, 0],
    'S6': [25, 0],
    'S7': [50, 0],
    'S8': [42, 0],
    'S9': [44, 0],
    'S11': [34, 0],
}
num_novel_pose_frames = {
    'S1': [9, 25],
    'S5': [22, 42],
    'S6': [14, 25],
    'S7': [34, 50],
    'S8': [15, 42],
    'S9': [23, 44],
    'S11': [14, 34],
}

subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

h36m_novel_view_frames = {k:[0] for k in subjects}
h36m_novel_pose_frames = {k:[0] for k in subjects}
h36m_training_frames = {k:[0] for k in subjects}

for subject in subjects:
    n_idxs = num_novel_view_frames[subject][0]
    n_start = num_novel_view_frames[subject][1]
    idxs = np.arange(n_start, n_start + n_idxs)
    h36m_novel_view_frames[subject] = idxs

    n_idxs = num_novel_pose_frames[subject][0]
    n_start = num_novel_pose_frames[subject][1]
    idxs = np.arange(n_start, n_start + n_idxs)
    h36m_novel_pose_frames[subject] = idxs

    n_select = 5
    n_cam = 3
    n_idxs = num_train_frames[subject][0]
    n_start = num_train_frames[subject][1]
    inv = int(n_idxs / n_select) * n_cam
    idxs_cam_0 = np.arange(n_idxs * n_cam)[::inv]
    idxs = idxs_cam_0
    for i in range(n_cam - 1):
        idxs = np.concatenate([idxs, idxs_cam_0 + i + 1])
    h36m_training_frames[subject] = idxs

class ZJUH36MDataset(BaseH5Dataset):

    N_render = 8
    render_skip = 20

    def __init__(self, *args, pose_skip=None, color_bg=False, **kwargs):
        self.pose_skip = pose_skip
        self.color_bg = color_bg
        super(ZJUH36MDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, q_idx):
        '''
        q_idx: index queried by sampler, should be in range [0, len(dataset)].
        Note - self._idx_map maps q_idx to indices of the sub-dataset that we want to use.
               therefore, self._idx_map[q_idx] may not lie within [0, len(dataset)]
        '''

        if self._idx_map is not None:
            idx = self._idx_map[q_idx]
        else:
            idx = q_idx

        # TODO: map idx to something else (e.g., create a seq of img idx?)
        # or implement a different sampler
        # as we may not actually use the idx here

        if self.dataset is None:
            self.init_dataset()

        # get camera information
        c2w, focal, center, cam_idxs = self.get_camera_data(idx, q_idx, self.N_samples)

        # get kp index and kp, skt, bone, cyl
        kp_idxs, kps, bones, skts, cyls = self.get_pose_data(idx, q_idx, self.N_samples)

        # sample pixels
        pixel_idxs, fg, sampling_mask = self.sample_pixels(idx, q_idx)

        # maybe get a version that computes only sampled points?
        rays_o, rays_d = self.get_rays(c2w, focal, pixel_idxs, center)

        # load the image, foreground and background,
        # and get values from sampled pixels
        rays_rgb, fg, bg, rgb_not_masked = self.get_img_data(idx, pixel_idxs, fg=fg)

        return_dict = {'rays_o': rays_o,
                       'rays_d': rays_d,
                       'target_s': rays_rgb,
                       'target_s_not_masked': rgb_not_masked,
                       'kp_idx': kp_idxs,
                       'kp3d': kps,
                       'bones': bones,
                       'skts': skts,
                       'cyls': cyls,
                       'cam_idxs': cam_idxs,
                       'fgs': fg,
                       'bgs': bg,
                       }

        return return_dict

    def init_meta(self):
        if self.split == 'test':
            self.h5_path = self.h5_path.replace('train', 'test')
        super(ZJUH36MDataset, self).init_meta()

        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]

        if self.split == 'test':
            n_unique_cam = len(np.unique(self.cam_idxs))
            self.kp_idxs = self.kp_idxs // n_unique_cam

        # forcefully make the background dark
        if not self.color_bg:
            self.bgs = self.bgs.copy() * 0

        dataset.close()

    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return self.kp_idxs[idx], q_idx

    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return self.cam_idxs[idx], q_idx

    def _get_subset_idxs(self, render=False):
        '''
        get the part of data that you want to train on
        '''
        if self._idx_map is not None:
            i_idxs = self._idx_map
            _k_idxs = self._idx_map
            _c_idxs = self._idx_map
            _kq_idxs = np.arange(len(self._idx_map))
            _cq_idxs = np.arange(len(self._idx_map))
        else:
            i_idxs = np.arange(self._N_total_img)
            _k_idxs = _kq_idxs = np.arange(self._N_total_img)
            _c_idxs = _cq_idxs = np.arange(self._N_total_img)

        # call the dataset-dependent fns to get the true kp/cam idx
        k_idxs, kq_idxs = self.get_kp_idx(_k_idxs, _kq_idxs)
        c_idxs, cq_idxs = self.get_cam_idx(_c_idxs, _cq_idxs)

        return k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs

    def get_img_data(self, idx, pixel_idxs, *args, **kwargs):
        '''
        get image data (in np.uint8), convert to float
        '''

        fg = self.dataset['masks'][idx, pixel_idxs].astype(np.float32)
        img = self.dataset['imgs'][idx, pixel_idxs].astype(np.float32) / 255.
        bnd = (self.dataset['sampling_masks'][idx, pixel_idxs] > 1).astype(np.float32)

        bg = None
        img_not_masked = img.copy()
        if self.has_bg:
            bg_idx = self.bg_idxs[idx]
            bg = self.bgs[bg_idx, pixel_idxs].astype(np.float32) / 255.

            bnd_rgb = img[bnd[:, 0] > 0].copy()
            if self.perturb_bg:
                noise = np.random.random(bg.shape).astype(np.float32)
                bg = (1 - fg) * noise + fg * bg# do not perturb foreground area!
                # provide boundary color
                #bg[bnd[:, 0] > 0] = bnd_rgb

            if self.mask_img:
                img = img * fg + (1. - fg) * bg
                # provide boundary color
                #img[bnd[:, 0] > 0] = bnd_rgb

        return img, fg, bg, img_not_masked

    def get_render_novel_pose_data(self):
        eval_idxs = h36m_novel_pose_frames[self.subject]
        h5_path = self.h5_path.replace('train', 'anim')

        H, W = self.HW
        dataset = h5py.File(h5_path, 'r')
        render_imgs = dataset['imgs'][:][eval_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][:][eval_idxs].reshape(-1, H, W, 1).astype(np.float32)

        bgs = dataset['bkgds'][:]
        bkgd_idxs = dataset['bkgd_idxs'][:][eval_idxs]
        render_bgs = bgs[bkgd_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_bgs = np.zeros(render_bgs.shape, dtype=np.float32)

        render_imgs = render_imgs * render_fgs + (1. - render_fgs) * render_bgs

        render_bg_idxs = np.zeros(len(eval_idxs)).astype(np.int64)
        pose_idx = dataset['kp_idxs'][:][eval_idxs]
        kp3d = dataset['kp3d'][:][pose_idx]
        skts = dataset['skts'][:][pose_idx]
        bones = dataset['bones'][:][pose_idx]

        c_idxs = dataset['img_pose_indices'][:][eval_idxs]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, self.focals[c_idxs])
        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()
        dataset.close()

        render_data = {
            'imgs': render_imgs,
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(self.bgs),
            # camera data
            'cam_idxs': np.array([-1]), #c_idxs,
            'cam_idxs_len': len(self.c2ws),
            'c2ws': self.c2ws[c_idxs],
            'hwf': hwf,
            'center': center,
            # keypoint data
            'kp_idxs': np.arange(len(eval_idxs)),
            'kp_idxs_len': len(kp3d),
            'kp3d': kp3d,
            'skts': skts,
            'bones': bones,
        }
        return render_data

    def get_render_novel_view_data(self):
        eval_idxs = h36m_novel_view_frames[self.subject]
        h5_path = self.h5_path.replace('train', 'anim')

        H, W = self.HW
        dataset = h5py.File(h5_path, 'r')
        render_imgs = dataset['imgs'][:][eval_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][:][eval_idxs].reshape(-1, H, W, 1).astype(np.float32)
        bgs = dataset['bkgds'][:]
        bkgd_idxs = dataset['bkgd_idxs'][:][eval_idxs]
        render_bgs = bgs[bkgd_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_bgs = np.zeros(render_bgs.shape, dtype=np.float32)

        render_imgs = render_imgs * render_fgs + (1. - render_fgs) * render_bgs

        render_bg_idxs = np.zeros(len(eval_idxs)).astype(np.int64)
        pose_idx = dataset['kp_idxs'][:][eval_idxs]
        kp3d = dataset['kp3d'][:][pose_idx]
        skts = dataset['skts'][:][pose_idx]
        bones = dataset['bones'][:][pose_idx]

        c_idxs = dataset['img_pose_indices'][:][eval_idxs]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, self.focals[c_idxs])
        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()
        dataset.close()

        render_data = {
            'imgs': render_imgs,
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(self.bgs),
            # camera data
            'cam_idxs': np.array([-1]), #c_idxs,
            'cam_idxs_len': len(self.c2ws),
            'c2ws': self.c2ws[c_idxs],
            'hwf': hwf,
            'center': center,
            # keypoint data
            'kp_idxs': np.arange(len(eval_idxs)),
            'kp_idxs_len': len(kp3d),
            'kp3d': kp3d,
            'skts': skts,
            'bones': bones,
        }
        return render_data

    def get_render_training_data(self):
        eval_idxs = h36m_training_frames[self.subject]
        # h5_path = self.h5_path.replace('train', 'test')
        h5_path = self.h5_path

        H, W = self.HW
        dataset = h5py.File(h5_path, 'r')
        render_imgs = dataset['imgs'][:][eval_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][:][eval_idxs].reshape(-1, H, W, 1).astype(np.float32)
        #render_bgs = np.ones_like(render_imgs[:1]).astype(np.float32)
        bgs = dataset['bkgds'][:]
        bkgd_idxs = dataset['bkgd_idxs'][:][eval_idxs]
        render_bgs = bgs[bkgd_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_bgs = np.zeros(render_bgs.shape, dtype=np.float32)

        render_imgs = render_imgs * render_fgs + (1. - render_fgs) * render_bgs

        render_bg_idxs = np.zeros(len(eval_idxs)).astype(np.int64)
        pose_idx = dataset['kp_idxs'][:][eval_idxs]
        kp3d = dataset['kp3d'][:][pose_idx]
        skts = dataset['skts'][:][pose_idx]
        bones = dataset['bones'][:][pose_idx]

        c_idxs = dataset['img_pose_indices'][:][eval_idxs]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, self.focals[c_idxs])
        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()
        dataset.close()


        render_data = {
            'imgs': render_imgs,
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(self.bgs),
            # camera data
            'cam_idxs': c_idxs,
            'cam_idxs_len': len(self.c2ws),
            'c2ws': self.c2ws[c_idxs],
            'hwf': hwf,
            'center': center,
            # keypoint data
            'kp_idxs': np.arange(len(eval_idxs)),
            'kp_idxs_len': len(kp3d),
            'kp3d': kp3d,
            'skts': skts,
            'bones': bones,
        }
        return render_data

    def get_render_data(self):
        eval_idxs = h36m_zju_eval_frames[self.subject][::self.render_skip][:self.N_render]
        h5_path = self.h5_path.replace('train', 'test')

        H, W = self.HW
        dataset = h5py.File(h5_path, 'r')
        render_imgs = dataset['imgs'][eval_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][eval_idxs].reshape(-1, H, W, 1).astype(np.float32)
        #render_bgs = np.ones_like(render_imgs[:1]).astype(np.float32)
        bgs = dataset['bkgds'][:]
        bkgd_idxs = dataset['bkgd_idxs'][eval_idxs]
        render_bgs = bgs[bkgd_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_bg_idxs = np.zeros(len(eval_idxs)).astype(np.int64)

        kp3d = dataset['kp3d'][eval_idxs]
        skts = dataset['skts'][eval_idxs]
        bones = dataset['bones'][eval_idxs]

        c_idxs = dataset['img_pose_indices'][eval_idxs]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, self.focals[c_idxs])
        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()
        dataset.close()


        render_data = {
            'imgs': render_imgs,
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(self.bgs),
            # camera data
            'cam_idxs': c_idxs,
            'cam_idxs_len': len(self.c2ws),
            'c2ws': self.c2ws[c_idxs],
            'hwf': hwf,
            'center': center,
            # keypoint data
            'kp_idxs': np.arange(len(eval_idxs)),
            'kp_idxs_len': len(kp3d),
            'kp3d': kp3d,
            'skts': skts,
            'bones': bones,
        }
        return render_data


