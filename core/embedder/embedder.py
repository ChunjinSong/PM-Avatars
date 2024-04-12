import torch
import torch.nn as nn
import numpy as np

from core.utils.skeleton_utils import (
    axisang_to_rot6d,
    get_bone_align_transforms,
)


def transform_batch_pts(pts, skt):
    '''
    Transform points/vectors from world space to local space

    Parameters
    ----------
    pts: Tensor (..., 3) in world space
    skt: Tensor (..., N_joints, 4, 4) world-to-local transformation
    '''

    N_rays, N_samples = pts.shape[:2]
    NJ = skt.shape[-3]

    if skt.shape[0] < pts.shape[0]:
        skt = skt.expand(pts.shape[0], *skt.shape[1:])

    # make it from (N_rays, N_samples, 4) to (N_rays, NJ, 4, N_samples)
    pts = torch.cat([pts, torch.ones(*pts.shape[:-1], 1)], dim=-1)
    pts = pts.view(N_rays, -1, N_samples, 4).expand(-1, NJ, -1, -1).transpose(3, 2).contiguous()
    # MM: (N_rays, NJ, 4, 4) x (N_rays, NJ, 4, N_samples) -> (N_rays, NJ, 4, N_samples)
    # permute back to (N_rays, N_samples, NJ, 4)
    mm = (skt @ pts).permute(0, 3, 1, 2).contiguous()

    return mm[..., :-1] # don't need the homogeneous part


def transform_batch_rays(rays_d, skt):
    '''
    Transform direction vectors from world space to local space

    Parameters
    ----------
    rays_d: Tensor (N_rays, 3) direction in world space
    skt: Tensor (..., N_joints, 4, 4) world-to-local transformation
    '''

    # apply only the rotational part
    assert rays_d.dim() == 2
    N_rays = len(rays_d)
    N_samples = 1
    NJ = skt.shape[-3]
    rot = skt[..., :3, :3]

    if rot.shape[0] < rays_d.shape[0]:
        rot = rot.expand(rays_d.shape[0], *rot.shape[1:])
    rays_d = rays_d.view(N_rays, -1, N_samples, 3).expand(-1, NJ, -1, -1).transpose(3, 2).contiguous()
    mm = (rot @ rays_d).permute(0, 3, 1, 2).contiguous()

    return mm


class BaseEmbedder(nn.Module):

    def __init__(self, N_joints=24, N_dims=3, skel_type=None):
        super().__init__()
        self.N_joints = N_joints
        self.N_dims = N_dims
        self.skel_type = skel_type

    @property
    def dims(self):
        return self.N_joints * self.N_dims

    @property
    def encoder_name(self):
        return self.__class__.__name__

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.encoder_name}(N_joints={self.N_joints}, N_dims={self.N_dims})'


class WorldToLocalEmbedder(BaseEmbedder):

    def forward(self, pts, skts, **kwargs):
        pts_t = transform_batch_pts(pts, skts)
        return {'pts_t': pts_t}


class BoneAlignEmbedder(WorldToLocalEmbedder):

    def __init__(self, rest_pose, *args, **kwargs):
        super(BoneAlignEmbedder, self).__init__(*args, **kwargs)
        self.rest_pose = rest_pose

        transforms, child_idxs = get_bone_align_transforms(rest_pose, self.skel_type)
        self.child_idxs = np.array(child_idxs)
        self.register_buffer('transforms', transforms)

    def forward(self, pts, skts, rigid_idxs=None, **kwargs):
        if rigid_idxs is not None:
            skts = skts[..., rigid_idxs, :, :]
        pts_jt = transform_batch_pts(pts, skts)
        pts_t = self.align_pts(pts_jt, self.transforms, rigid_idxs=rigid_idxs)
        return {'pts_t': pts_t, 'pts_jt': pts_jt}
    
    def align_pts(self, pts, align_transforms=None, rigid_idxs=None):
        if align_transforms is None:
            align_transforms = self.transforms
        if rigid_idxs is not None:
            align_transforms = align_transforms[rigid_idxs]
        pts_t = (align_transforms[..., :3, :3] @ pts[..., None]).squeeze(-1) \
                    + align_transforms[..., :3, -1]
        return pts_t
    
    def unalign_pts(self, pts_t, align_transforms=None, rigid_idxs=None):
        if align_transforms is None:
            align_transforms = self.transforms
        if rigid_idxs is not None:
            align_transforms = align_transforms[rigid_idxs]
        pts_t = pts_t - align_transforms[..., :3, -1]
        pts = align_transforms[..., :3, :3].transpose(-1, -2) @ pts_t[..., None]
        return pts.squeeze(-1)


class Pose6DEmbedder(WorldToLocalEmbedder):

    @property
    def dims(self):
        return 6

    def forward(self, pts, skts, bones, N_unique=1, **kwargs):
        skip = bones.shape[0] // N_unique
        unique_bones = bones[::skip]
        return {
            'pose': axisang_to_rot6d(unique_bones),
        }


class WorldToRootViewEmbedder(BaseEmbedder):

    @property
    def dims(self):
        return self.N_dims

    def forward(self, rays_o, rays_d, skts, refs, **kwargs):
        root = self.skel_type.root_id
        # Assume root index is at 0
        rays_dt = transform_batch_rays(rays_d, skts[:, root:root+1])
        if refs is not None:
            # expand so that the ray embedding has sample dimension
            N_expand = refs.shape[1]
            rays_dt = rays_dt.expand(-1, N_expand, -1, -1)
        return {'d': rays_dt}

class SHEncoder(nn.Module):
    def __init__(self, input_dims=3, degree=4):

        super().__init__()

        self.input_dim = input_dims
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.dims = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.dims), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                # result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result, None