import torch.nn as nn

from core.utils.ray_utils import *
from core.utils.skeleton_utils import get_bone_align_transforms

class RayCast(nn.Module):
    '''
    Base raycasting module
    '''

    def __init__(
        self,
        near=0.,
        far=100.,
        N_samples=48,
        N_importance=32,
        **kwargs,
    ):
        '''
        Parameters
        ----------
        near: float, near plane for the ray, use only as default value when ray is not in cylinder
        far: float, far plane for the ray, also used only when the ray is not within the cynlider
        '''
        super(RayCast, self).__init__()
        self.near = near
        self.far = far
        self.N_samples = N_samples
        self.N_importance = N_importance
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(
        self, 
        batch, 
        N_samples=None, 
        pts=None, 
        weights=None, 
        z_vals=None, 
        importance=False, 
        **kwargs
    ):

        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        cyls = batch['cyls']
        skts = batch['skts']

        N_rays = len(rays_o)

        near, far = self.get_near_far(
            rays_o,
            rays_d,
            cyls,
            near=self.near,
            far=self.far,
            skts=skts,
        )

        sample_info = {}
        if pts is None:
            if N_samples is None:
                N_samples = self.N_samples
            pts, z_vals = self.sample_pts(rays_o, rays_d, near, far, N_rays, N_samples)
        elif importance:
            if N_samples is None:
                N_samples = self.N_importance
            pts, z_vals, z_samples, sorted_idxs = self.sample_pts_is(
                                                      rays_o, 
                                                      rays_d, 
                                                      z_vals,
                                                      weights,
                                                      N_samples,
                                                      is_only=True,
                                                  )
            sample_info['sorted_idxs'] = sorted_idxs
        else:
            NotImplementedError(f'pts input is not None, but not in importance sampling mode!')
        sample_info['pts'] = pts
        sample_info['z_vals'] = z_vals

        return sample_info

    def get_near_far(self, rays_o, rays_d, cyls, near=0., far=100., **kwargs):
        return get_near_far_in_cylinder(rays_o, rays_d, cyls, near=near, far=far)
    
    def sample_pts(self, rays_o, rays_d, near, far, N_rays, N_samples,
                   perturb=0.0, lindisp=False, pytest=False, ray_noise_std=0.):

        z_vals = sample_from_lineseg(near, far, N_rays, N_samples,
                                     perturb, lindisp, pytest=pytest)

        # range of points should be bounded within 2pi.
        # so the lowest frequency component (sin(2^0 * p)) don't wrap around within the bound
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        if ray_noise_std > 0.:
            pts = pts + torch.randn_like(pts) * ray_noise_std

        return pts, z_vals

    def sample_pts_is(self, rays_o, rays_d, z_vals, weights, N_importance,
                      det=True, pytest=False, is_only=False, ray_noise_std=0.):

        z_vals, z_samples, sorted_idxs = isample_from_lineseg(z_vals, weights, N_importance, det=det,
                                                              pytest=pytest, is_only=is_only)

        pts_is = rays_o[...,None,:] + rays_d[...,None,:] * z_samples[...,:,None] # [N_rays, N_samples + N_importance, 3]

        if ray_noise_std> 0.:
            pts_is = pts_is + torch.randn_like(pts_is) * ray_noise_std


        return pts_is, z_vals, z_samples, sorted_idxs

class VolRayCast(RayCast):
    """ Only sample within volumes
    """

    def __init__(
        self, 
        *args, 
        vol_scale_fn=None,
        rest_pose=None,
        skel_type=None,
        rigid_idxs=None,
        training_bound=1.1,
        **kwargs
    ):
        '''
        vol_scale_fn: function, to inquire the current volume size
        '''

        super(VolRayCast, self).__init__(*args, **kwargs)
        assert vol_scale_fn is not None
        assert rest_pose is not None
        assert skel_type is not None
        self.vol_scale_fn = vol_scale_fn
        self.rest_pose = rest_pose
        self.skel_type = skel_type
        self.rigid_idxs = rigid_idxs
        self.training_bound = training_bound
        transforms, _ = get_bone_align_transforms(rest_pose.cpu().numpy(), skel_type)
        self.register_buffer('transforms', transforms)

    @torch.no_grad()
    def get_near_far(self, rays_o, rays_d, cyls, skts, near=0, far=100, **kwargs):
        near, far = get_near_far_in_cylinder(rays_o, rays_d, cyls, near=near, far=far)

        B, J = skts.shape[:2]
        align_transforms = self.transforms.to(rays_o.device)
        vol_scale = self.vol_scale_fn().reshape(1, -1, 3).detach().to(rays_o.device)
        if self.rigid_idxs is not None:
            skts = skts[:, self.rigid_idxs]
            align_transforms = align_transforms[self.rigid_idxs]
            vol_scale = vol_scale[:, self.rigid_idxs]

            J = len(self.rigid_idxs)

        # transform both rays origin and direction to the per-joint coordinate
        rays_ot = (skts[..., :3, :3] @ rays_o.reshape(B, 1, 3, 1) + skts[..., :3, -1:]).reshape(B, J, 3)
        rays_dt = (skts[..., :3, :3] @ rays_d.reshape(B, 1, 3, 1)).reshape(B, J, 3)

        # have to do this because self.transforms is not a buffer
        rays_ot = ((align_transforms[..., :3, :3] @ rays_ot[..., None]) + \
                    align_transforms[..., :3, -1:]).reshape(B, J, 3)
        rays_dt = (align_transforms[..., :3, :3] @ rays_dt[..., None]).reshape(B, J, 3)
        
        # scale the rays by the learned volume scale and find the intersections with the volumes
        bound_range = self.training_bound if self.training else 1.1

        # TODO: this function sometimes give blank points ... fix them?
        p_valid, v_valid, p_intervals = get_ray_box_intersections(
                                        rays_ot / vol_scale, 
                                        rays_dt / vol_scale, 
                                        bound_range=bound_range, # intentionally makes the bound a bit larger
                                    )

        # now undo the scale so we can calculate the near far in the original space
        vol_scale = vol_scale.expand(B, J, 3)
        p_intervals = p_intervals * vol_scale[v_valid][..., None, :]

        norm_rays = rays_dt[v_valid].norm(dim=-1)
        # find the step size (near / far)
        # t * norm_ray + ray_o = p -> t =  (p - ray_o) / norm_rays
        # -> distance is the norm 
        steps = (p_intervals - rays_ot[v_valid][..., None, :]).norm(dim=-1) / norm_rays[..., None]

        # extract near/far for each volume
        v_near = 100000 * torch.ones(B, J)
        v_far = -100000 * torch.ones(B, J)

        # find the near/far
        v_near[v_valid] = steps.min(dim=-1).values
        v_far[v_valid] = steps.max(dim=-1).values

        # pick the closest/farthest points as the near/far planes
        v_near = v_near.min(dim=-1).values
        v_far = v_far.max(dim=-1).values

        # merge the values back to the cylinder near far
        ray_valid = (v_valid.sum(-1) > 0)
        new_near = near.clone()
        new_far = far.clone()

        new_near[ray_valid, 0] = v_near[ray_valid]
        new_far[ray_valid, 0] = v_far[ray_valid]

        return new_near, new_far