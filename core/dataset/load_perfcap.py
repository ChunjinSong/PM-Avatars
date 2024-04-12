import h5py
from core.dataset.dataset import PoseRefinedDataset
from core.utils.skeleton_utils import *

class MonoPerfCapDataset(PoseRefinedDataset):
    n_vals = {'weipeng': 230, 'nadia': 327}

    # define the attribute for rendering data
    render_skip = 10
    N_render = 15

    def __init__(self, *args, undo_scale=True, **kwargs):
        self.undo_scale = undo_scale

        if undo_scale and 'load_refined' in kwargs:
            assert kwargs['load_refined'] is False, 'Cannot load refined data when undoing scale.'
        super(MonoPerfCapDataset, self).__init__(*args, **kwargs)

    def init_meta(self):
        dataset = h5py.File(self.h5_path, 'r', swmr=True)
        self.pose_scale = dataset['pose_scale'][()]
        self.rest_pose = dataset['rest_pose'][:]

        train_idxs = np.arange(len(dataset['imgs']))

        self._idx_map = None
        if self.split != 'full':
            n_val = self.n_vals[self.subject]
            val_idxs = train_idxs[-n_val:]
            train_idxs = train_idxs[:-n_val]

            if self.split == 'train':
                self._idx_map = train_idxs
            elif self.split == 'val':
                # skip redundant frames
                self._idx_map = val_idxs[::5]
            else:
                raise NotImplementedError(f'Split {self.split} is not implemented.')

        self.temp_validity = np.ones(len(train_idxs))
        self.temp_validity[0] = 0
        dataset.close()
        super(MonoPerfCapDataset, self).init_meta()
        # the estimation for MonoPerfCap is somehow off by a small scale (possibly due do the none 1:1 aspect ratio)
        if self.undo_scale:
            self.undo_pose_scale()
        self.c2ws[..., :3, -1] /= 1.05

    def undo_pose_scale(self):
        print(f'Undoing MonoPerfCap pose scale')
        pose_scale = self.pose_scale
        self.kp3d = self.kp3d / pose_scale
        l2ws = np.linalg.inv(self.skts)
        l2ws[..., :3, 3] /= pose_scale
        self.skts = np.linalg.inv(l2ws)
        self.cyls = get_kp_bounding_cylinder(self.kp3d, skel_type=SMPLSkeleton, head='-y')
        self.c2ws[..., :3, -1] /= pose_scale

        self.rest_pose = self.rest_pose.copy() / pose_scale
        # assertions to check if everything is alright
        assert np.allclose(self.kp3d, np.linalg.inv(self.skts)[:, :, :3, 3], atol=1e-5)
        l2ws = np.linalg.inv(self.skts)

        l2ws_from_rest = np.array([get_smpl_l2ws(b, self.rest_pose) for b in self.bones]).astype(np.float32)
        l2ws_from_rest[..., :3, -1] += self.kp3d[:, :1]

        assert np.allclose(l2ws, l2ws_from_rest, atol=1e-5)

        print(f'Done undoing MonoPerfCap pose scale.')

    def init_temporal_validity(self):
        return self.temp_validity

    def get_render_data(self):
        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        n_val = self.n_vals[self.subject]
        train_idxs = np.arange(len(dataset['imgs']))
        val_idxs = train_idxs[-n_val:]
        eval_idxs = val_idxs[::5]

        # # get the subset idxs to collect the right data
        # k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs = self._get_subset_idxs(render=True)
        #
        # # grab only a subset (15 images) for rendering
        # kq_idxs = kq_idxs[::self.render_skip][:self.N_render]
        # cq_idxs = cq_idxs[::self.render_skip][:self.N_render]
        # i_idxs = i_idxs[::self.render_skip][:self.N_render]
        # k_idxs = k_idxs[::self.render_skip][:self.N_render]
        # c_idxs = c_idxs[::self.render_skip][:self.N_render]

        # get images if split == 'render'
        # note: needs to have self._idx_map
        H, W = self.HW
        render_imgs = dataset['imgs'][eval_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][eval_idxs].reshape(-1, H, W, 1)
        render_bgs = self.bgs.reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_bg_idxs = self.bg_idxs[eval_idxs]

        render_bgs = np.zeros(render_bgs.shape, dtype=np.float32)
        render_imgs = render_imgs * render_fgs + (1. - render_fgs) * render_bgs
        render_bg_idxs = np.zeros(len(eval_idxs)).astype(np.int64)

        H = np.repeat([H], len(eval_idxs), 0)
        W = np.repeat([W], len(eval_idxs), 0)
        hwf = (H, W, self.focals[eval_idxs])

        center = None
        if self.centers is not None:
            center = self.centers[eval_idxs].copy()
        else:
            center = (self.HW[::-1] // 2)[None].repeat(len(self.c2ws), 0).astype(np.float32)

        # TODO: c_idxs, k_idxs ... confusion

        render_data = {
            'imgs': render_imgs,
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(self.bgs),
            # camera data
            'cam_idxs': np.array([-1]),
            'cam_idxs_len': len(self.c2ws),
            'c2ws': self.c2ws[eval_idxs],
            'hwf': hwf,
            'center': center,
            # keypoint data
            'kp_idxs': eval_idxs,
            'kp_idxs_len': len(self.kp3d),
            'kp3d': self.kp3d[eval_idxs],
            'skts': self.skts[eval_idxs],
            'bones': self.bones[eval_idxs],
        }

        '''
        from core.utils.skeleton_utils import draw_skeletons_3d
        import imageio
        skel_imgs = draw_skeletons_3d((render_data['imgs']*255).astype(np.uint8), render_data['kp3d'], render_data['c2ws'], *hwf, center)
        import pdb; pdb.set_trace()
        print
        '''
        '''
        '''

        dataset.close()

        return render_data
