import torch
from hydra.utils import instantiate
from .embedding import Optcodes
from core.utils.ray_utils import kp_to_valid_rays
from core.utils.skeleton_utils import (
    SMPLSkeleton,
    get_skel_profile_from_rest_pose,
    calculate_kinematic,
)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def batchify_fn(fn, fn_inputs, N_total, chunk=4096):
    """ Break evaluation into batches to avoid OOM
    """
    all_ret = {}

    for i in range(0, N_total, chunk):
        batch_inputs = {k: fn_inputs[k][i:i+chunk] if torch.is_tensor(fn_inputs[k]) else fn_inputs[k]
                        for k in fn_inputs}
        ret = fn(batch_inputs)
        if ret is None:
            continue

        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def merge_encodings(
        encoded,
        encoded_is,
        sorted_idxs,
        N_rays,
        N_total_samples,
        inplace=True
    ):
    """
    merge coarse and fine encodings.
    Parameters
    ----------
    encoded: dictionary of coarse encodings
    encoded_is: dictionary of fine encodings
    sorted_idxs: define how the [encoded, encoded_is] are sorted
    """
    if encoded_is is None:
        return encoded
    if encoded is None:
        return encoded_is
    gather_idxs = torch.arange(N_rays * (N_total_samples)).view(N_rays, -1)
    gather_idxs = torch.gather(gather_idxs, 1, sorted_idxs)

    merged = encoded if inplace else {}

    for k in encoded.keys():
        if k in ['valid_idxs'] or encoded[k] is None or isinstance(encoded[k], dict):
            continue
        merged[k] = merge_samples(encoded[k], encoded_is[k], gather_idxs, N_total_samples)

    # need special treatment here to preserve the computation graph.
    # (otherwise we can just re-encode everything again, but that takes extra computes)
    if 'pts' in encoded and encoded['pts'] is not None:
        if not inplace:
            merged['pts'] = encoded['pts']

        merged['pts_is'] = encoded_is['pts']
        merged['gather_idxs'] = gather_idxs

        merged['pts_sorted'] = merge_samples(encoded['pts'], encoded_is['pts'],
                                                gather_idxs, N_total_samples)

    return merged

def merge_samples(x, x_is, gather_idxs, N_total_samples):
    """
    merge coarse and fine samples.
    Parameteters
    ------------
    x: coarse samples of shape (N_rays, N_coarse, -1)
    x_is: importance samples of shape (N_rays, N_fine, -1)
    gather_idx: define how the [x, x_is] are sorted
    """
    if x is None or x.shape[-1] == 0:
        return None
    N_rays = x.shape[0]
    x_is = torch.cat([x, x_is], dim=1)
    sh = x_is.shape
    feat_size = np.prod(sh[2:])
    if x_is.shape[1] != N_total_samples:
        # extra signal: may not have the same shape as the standard ray samples
        return x_is
    x_is = x_is.view(-1, feat_size)[gather_idxs, :]
    x_is = x_is.view(N_rays, N_total_samples, *sh[2:])

    return x_is


class PmAvatar(nn.Module):

    def __init__(
            self,
            D,
            W,
            N_pm,
            view_W,
            pts_embedder,
            view_embedder,
            view_posi_enc,
            modulator,
            pose_extractor,
            freq_dim,
            win_fun,
            raycaster,
            skel_type,
            rest_pose,
            graph_net,
            voxel_feat=128,
            skips=[4],
            use_framecodes=False,
            framecode_ch=16,
            n_framecodes=0,
            density_noise_std=1.0,
            **kwargs,
    ):
        '''
        Parameters
        ----------
        D: int, depth of the MLP
        W: int, width of the MLP
        view_W: int, width of the view MLP
        pts_embedder: embedder module config, to encode 3d points w.r.t. body keypoints and poses
        pts_posi_enc: positional encoding config for the pts embedding
        view_embedder: embedder config, to encode view vectors w.r.t. body info
        view_posi_enc: positional encoding config for the view embedding
        raycast: ray casting module config
        skel_type: skeleton, define the details of the skeleton
        skips: list, layers to do skip connection
        use_framecodes: Bool, to use framecodes for each frame
        framecode_ch: int, size of the framecode
        n_framecodes: int, number of framecodes
        density_noise_std: float, noise to apply on density during training time
        '''

        super(PmAvatar, self).__init__()
        self.N_pm = N_pm
        self.freq_dim = freq_dim
        self.voxel_feat = voxel_feat
        self.D = D
        self.W = W
        self.view_W = view_W
        self.skips = skips
        self.use_framecodes = use_framecodes
        self.framecode_ch = framecode_ch
        self.n_framecodes = n_framecodes
        self.skel_type = skel_type
        self.density_noise_std = density_noise_std

        # initialize skeleton settings
        self.init_skeleton(skel_type, rest_pose)
        # instantiate embedder and network
        self.init_embedder(
            pts_embedder=pts_embedder,
            view_embedder=view_embedder,
            view_posi_enc=view_posi_enc,
            **kwargs
        )
        self.init_raycast(raycaster, **kwargs)
        self.init_density_net()
        self.init_radiance_net()
        self.init_graph_net(graph_net)

        self.N_joints = len(self.skel_type.joint_names)
        self.init_modulator(modulator)
        self.init_win_fun(win_fun)
        self.init_pose_extractor(pose_extractor)

    @property
    def dnet_input(self):
        return self.input_ch

    @property
    def vnet_input(self):
        if self.use_framecodes:
            return self.input_ch_view + self.framecode_ch + self.W
        return self.input_ch_view + self.W

    def init_skeleton(self, skel_type, rest_pose):
        self.register_buffer('rest_pose', torch.tensor(rest_pose))
        self.skel_profile = get_skel_profile_from_rest_pose(rest_pose, skel_type)

    def init_embedder(
        self,
        pts_embedder,
        pts_enc_pose,
        pts_enc_point,
        view_embedder,
        view_posi_enc,
        pose_embedder,
        **kwargs
    ):
        N_joints = len(self.skel_type.joint_names)
        # initialize points transformation
        self.pts_embedder = instantiate(
                                pts_embedder,
                                N_joints=N_joints,
                                N_dims=3,
                                skel_type=self.skel_type,
                                rest_pose=self.rest_pose.cpu().numpy(),
                            )

        self.pts_enc_pose = instantiate(pts_enc_pose)
        self.pts_enc_point = instantiate(pts_enc_point)
        # initialize pose transformation
        self.pose_embedder = instantiate(pose_embedder, N_joints=N_joints, N_dims=3)

        # initialize view transformation
        self.view_embedder = instantiate(view_embedder, N_joints=N_joints, N_dims=3, skel_type=self.skel_type)

        # initialize positional encoding for views (rays)
        self.view_dims = view_dims= self.view_embedder.dims
        self.view_posi_enc = instantiate(view_posi_enc, input_dims=view_dims)

        self.input_ch = self.freq_dim * self.N_pm
        self.input_ch_view = self.view_posi_enc.dims
        self.input_ch_pts = self.pts_enc_pose.out_dim
        self.input_ch_siren = self.pts_enc_point.out_dim * 24

    def init_raycast(
        self,
        raycast,
        **kwargs,
    ):
        self.raycast = instantiate(
                            raycast,
                            **kwargs,
                            vol_scale_fn=self.get_axis_scale,
                            rest_pose=self.rest_pose,
                            skel_type=self.skel_type,
                            _recursive_=False
                        )

    def init_density_net(self):

        W, D = self.W, self.D

        layers = [nn.Linear(self.dnet_input, W)]

        for i in range(D-1):
            if i not in self.skips:
                layers += [nn.Linear(W, W)]
            else:
                layers += [nn.Linear(W + self.dnet_input, W)]

        self.pts_linears = nn.ModuleList(layers)
        self.sigma_linear = nn.Linear(W, 1)

    def init_radiance_net(self):

        W, view_W = self.W, self.view_W

        # Note: legacy code, don't really need nn.ModuleList
        self.views_linears = nn.ModuleList([nn.Linear(self.vnet_input, view_W)])
        self.feature_linear = nn.Linear(W, view_W * 2)
        self.rgb_linear = nn.Linear(view_W, 3)

        if self.use_framecodes:
            self.framecodes = Optcodes(self.n_framecodes, self.framecode_ch)


    def init_win_fun(self, win_fun):
        self.win_fun = instantiate(
            win_fun,
            dim_in_pose=self.voxel_feat,
            dim_in_point=self.input_ch_pts
        )


    def init_graph_net(self, graph_net):

        self.graph_net = instantiate(
                             graph_net,
                             skel_type=self.skel_type,
                             per_node_input=6, #self.input_ch_graph,
                             output_ch=None, # determined by voxel_feat and voxel_res
                             voxel_feat=self.voxel_feat,
                             skel_profile=self.skel_profile,
                         )

    def init_modulator(self, modulator):
        self.modulator = instantiate(
            modulator,
            dim_in=self.input_ch_siren
        )


    def init_pose_extractor(self, pose_extractor):
        self.pose_extractor = instantiate(
            pose_extractor,
            dim_in=self.win_fun.dims,
            dim_out=self.freq_dim * 2 * self.N_pm
        )

    def get_axis_scale(self, rigid_idxs=None):
        axis_scale = self.graph_net.get_axis_scale()
        if rigid_idxs is None:
            return axis_scale
        return axis_scale[rigid_idxs]

    def forward(self, *args, forward_type='rays', **kwargs):
        if forward_type == 'rays':
            return self.forward_rays(*args, **kwargs)
        elif forward_type == 'render':
            return self.forward_render(*args, **kwargs)
        elif forward_type == 'geometry':
            return self.forward_geometry(*args, **kwargs)
        else:
            raise NotImplementedError(f'Unknown forward type {forward_type}')

    def forward_rays(self, batch):

        if 'rays_o' not in batch and 'rays_d' not in batch:
            raise NotImplementedError('Rays are not provided as input. '
                                      'For rendering image with automatically detected rays, call render(...)')

        # Step 1. cast ray
        bgs = batch.get('bgs', None)  # background color
        sample_info = self.raycast(batch)
        pts, z_vals = sample_info['pts'], sample_info['z_vals']

        # Step 2. model evaluation
        # TODO: do we need a get_nerf_inputs function?
        network_inputs = self.get_network_inputs(batch, pts)
        raw, encoded = self.evaluate_pts(network_inputs)

        if raw is None:
            return self._empty_outputs(batch)

        # Step 3. coarse rendering
        rendered = self.raw2outputs(
            raw,
            z_vals,
            batch['rays_d'],
            bgs=bgs,
            encoded=encoded,
        )

        if self.raycast.N_importance == 0:
            return {'rendered': rendered, 'encoded': encoded}

        # Step 4. importance sampling (if importance sampling enabled)
        rendered_coarse = rendered
        encoded_coarse = encoded

        weights = rendered_coarse['weights']
        is_sample_info = self.raycast(
            batch,
            pts=pts,
            z_vals=z_vals,
            weights=weights,
            importance=True,
        )
        pts_is = is_sample_info['pts']  # only importance samples
        z_vals = is_sample_info['z_vals']  # include both coarse and importance samples

        # Step 5. model evaluation (if importance sampling enabled)
        network_inputs = self.get_network_inputs(batch, pts_is)
        raw_is, encoded_is = self.evaluate_pts(network_inputs, encoded_coarse=encoded_coarse)

        # Step 6. merge coarse and importance prediction for rendering
        sorted_idxs = is_sample_info['sorted_idxs']
        N_rays = len(batch['rays_o'])
        N_total_samples = pts.shape[1] + pts_is.shape[1]
        encoded_is = merge_encodings(encoded_coarse, encoded_is, sorted_idxs, N_rays, N_total_samples)

        # Step 7. fine rendering
        if raw_is is not None:
            raw = merge_encodings({'raw': raw}, {'raw': raw_is}, sorted_idxs, N_rays, N_total_samples)['raw']
            rendered = self.raw2outputs(
                raw,
                z_vals,
                batch['rays_d'],
                bgs=bgs,
                encoded=encoded_is,
            )
        else:
            rendered = rendered_coarse

        return self.collect_outputs(rendered, rendered_coarse, encoded_is, encoded_coarse)

    def forward_render(self, render_data, render_factor=0, raychunk=1024 * 5):

        assert 'bones' in render_data, 'needs know the pose/bones (bones) parameter during rendering'
        if 'kp3d' not in render_data:
            kp3d, skts = calculate_kinematic(
                self.rest_pose,
                render_data['bones'],
                render_data.get('root_locs', None),
            )
            render_data['kp3d'] = kp3d
            render_data['skts'] = skts

        H, W, focals = render_data['hwf']
        kp3d = render_data['kp3d']
        cam_poses = render_data['c2ws']  # camera-to-world
        centers = render_data['center']

        if render_factor != 0:
            # change camera setting
            H, W = H // render_factor, W // render_factor
            focals = focals / render_factor
            if centers is not None:
                centers = centers / render_factor
            bgs = render_data['bgs']
            N, _H, _W, C = bgs.shape
            # (N, H, W, C) -> (N, C, H, W)
            bgs = bgs.permute(0, 3, 1, 2)
            bgs = F.interpolate(bgs, scale_factor=1 / render_factor, mode='bilinear', align_corners=False)
            # back to (N, H, W, C)
            bgs = bgs.permute(0, 2, 3, 1)
            render_data.update(bgs=bgs)

        if len(cam_poses) != len(kp3d):
            assert len(kp3d) == 1 or len(cam_poses) == 1, \
                f'Number of body poses should either match or can be broadcasted to number of camera poses. ' \
                f'Got {len(kp3d)} and {len(cam_poses)}'
            import pdb;
            pdb.set_trace()
            print

        rgb_imgs = []
        disp_imgs = []
        acc_maps = []

        for i in range(len(cam_poses)):

            # Step 1. find valid rays for this camera
            center = centers[i:i + 1] if centers is not None else None
            rays, valid_idxs, cyls, _ = kp_to_valid_rays(
                cam_poses[i:i + 1],
                H[i:i + 1].cpu().numpy(),
                W[i:i + 1].cpu().numpy(),
                focals[i:i + 1].cpu().numpy(),
                kps=kp3d[i:i + 1],
                centers=center,
            )
            rays_o, rays_d = rays[0]
            valid_idxs = valid_idxs[0]

            # initialize images
            bg = render_data['bgs'][i].cpu()

            if valid_idxs is None or (len(valid_idxs) == 0):
                rgb_imgs.append(bg.clone())
                disp_imgs.append(torch.zeros_like(bg[..., 0]).cpu())
                continue

            # flatten to (H * W, 3)
            rgb_img = bg.clone().flatten(end_dim=1)
            disp_img = torch.zeros_like(bg[..., 0]).flatten(end_dim=1)
            acc_img = torch.zeros_like(bg[..., 0]).flatten(end_dim=1)

            # Step 2. turn them into the format that forward_rays takes
            render_data.update(cyls=cyls)
            batch = self.to_ray_inputs(rays_o, rays_d, render_data, valid_idxs, index=i)

            # Step 3. forward
            with torch.no_grad():
                preds = batchify_fn(self.forward_rays, batch, N_total=len(rays_o), chunk=raychunk)

            # put the rendered values into images
            pred_rgb = preds['rgb_map']
            rgb_img[valid_idxs] = pred_rgb.cpu()
            rgb_img = rgb_img.reshape(H[i], W[i], 3)

            pred_disp = preds['disp_map']
            disp_img[valid_idxs] = pred_disp.cpu()
            disp_img = disp_img.reshape(H[i], W[i], 1)

            rgb_imgs.append(rgb_img)
            disp_imgs.append(disp_img)

            acc_map = preds['acc_map']
            acc_img[valid_idxs] = acc_map.cpu()
            acc_img = acc_img.reshape(H[i], W[i], 1)
            acc_maps.append(acc_img)

        rgb_imgs = torch.stack(rgb_imgs, dim=0)
        disp_imgs = torch.stack(disp_imgs, dim=0)
        acc_maps = torch.stack(acc_maps, dim=0)

        return {
            'rgb_imgs': rgb_imgs,
            'disp_imgs': disp_imgs,
            'acc_maps': acc_maps
        }

    def get_network_inputs(
            self,
            batch,
            pts,
            keys_from_batch=['kp3d', 'skts', 'bones', 'cam_idxs', 'rays_o', 'rays_d'],
            **kwargs
    ):
        '''
        get network input from batch
        '''
        ret = {k: batch[k] for k in keys_from_batch if k in batch}
        ret['pts'] = pts
        ret['N_unique'] = batch.get('N_unique', 1)
        return ret

    def to_ray_inputs(
            self,
            rays_o,
            rays_d,
            render_data,
            valid_idxs,
            index,
            keys_for_batch=['kp3d', 'skts', 'bones', 'cyls', 'cam_idxs', 'bgs'],
            pixel_data=['bgs'],
    ):
        """ Create a dicectionary in forward_rays format.
        This is used for rendering only.
        """
        ray_inputs = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'N_unique': 1,  # always one pose at a time
        }

        N_rays = len(rays_o)
        # TODO: hacky to define it here..

        for k in keys_for_batch:
            v = render_data[k]

            if len(v) > 1:
                v = v[index:index + 1]

            if k not in pixel_data:
                sh = v.shape[1:]
                v = v.expand(N_rays, *sh)
            else:
                v = v.flatten(end_dim=2)[valid_idxs.to(v.device)].to(rays_o.device)
            ray_inputs[k] = v
        return ray_inputs

    def evaluate_pts(self, inputs, geometry_only=False, geom_extra_rets=[], **kwargs):

        # Step 1. encode all pts feature
        density_inputs, encoded_pts = self.encode_pts(inputs)
        if density_inputs is None:
            # terminate because not valid points
            return None, None

        # Step 2. density prediction
        sigma, density_feature = self.inference_sigma(density_inputs)

        density = self.to_density(sigma)

        if geometry_only:
            geom_ret = {'density': density, 'sigma': sigma}
            if 'valid_idxs' in encoded_pts:
                geom_ret['valid_idxs'] = encoded_pts['valid_idxs']
            if 'surface_gradient' in encoded_pts:
                geom_ret['surface_gradient'] = encoded_pts['surface_gradient']
            for k in geom_extra_rets:
                if k in encoded_pts:
                    geom_ret[k] = encoded_pts[k]
                if k == 'density_feature':
                    geom_ret[k] = density_feature
            return None, geom_ret

        # Step 3. encode all ray feature
        view_inputs, encoded_views = self.encode_views(
            inputs,
            refs=encoded_pts['pts_t'],
            encoded_pts=encoded_pts
        )

        # Step 4. rgb prediction
        rgb = self.inference_rgb(view_inputs, density_feature)

        # Step 5: create final outputs
        shape = inputs['pts'].shape[:2]  # (N_rays, N_samples)
        outputs = self.fill_prediction(torch.cat([rgb, density], dim=-1), shape, encoded_pts)

        return outputs, self.collect_encoded(encoded_pts, encoded_views)

    def encode_pts(self, inputs):
        encoded = {}
        N_rays, N_samples = inputs['pts'].shape[:2]
        N_joints = len(self.skel_type.joint_names)
        # skts = inputs['skts']

        # get pts_t (3d points in local space which has been aligned)
        encoded_pts = self.pts_embedder(**inputs)  # [N_rays, N_samples, N_joints, 3]

        # Note: this return encoding for "unique bones/poses"
        # (See inputs['N_unique'])
        encoded_pose = self.pose_embedder(**inputs)
        pose_pe = encoded_pose['pose']

        # get graph feature
        h = self.graph_net(pose_pe)  # [N_batch, N_joints, gnn_feat]
        x_j = encoded_pts['pts_t'].reshape(N_rays * N_samples, N_joints, 3)
        x_j = x_j / self.get_axis_scale().reshape(1, N_joints, 3).abs()

        alpha, beta = self.graph_net.alpha, self.graph_net.beta
        p = torch.exp(-alpha * ((x_j ** beta).sum(-1))).detach()

        feat_x = self.pts_enc_pose(x_j)
        p, feat_pose = self.win_fun(feat_x, h, p)

        # extract "deformation feature"
        feat_pose = self.extract_pose_feature(feat_pose)

        x_j = self.pts_enc_point(x_j)
        x_j = (x_j * p[..., None])  # .sum(dim=1)  # [B, 3]
        x_j = x_j.reshape(x_j.shape[0], -1)

        vc_freq = self.modulator(x_j, feat_pose) # [B, feat_ffn]

        density_inputs = vc_freq

        encoded.update(
            **encoded_pts,
        )

        return density_inputs, encoded

    def extract_pose_feature(self, feat_pose):
        feat_pose = feat_pose.sum(dim=1)
        frequencies, phase_shifts = self.pose_extractor(feat_pose)
        B_freq, _ = frequencies.size()[:]
        frequencies = frequencies.reshape(B_freq, self.N_pm, self.freq_dim).permute(1, 0, 2)  # [3, B, freq_dim]
        phase_shifts = phase_shifts.reshape(B_freq, self.N_pm, self.freq_dim).permute(1, 0, 2)
        return [frequencies, phase_shifts]

    def encode_views(self, inputs, refs, encoded_pts):
        '''
        refs: reference tensor for expanding rays
        encoded_pts: point encoding that could be useful for encoding view
        '''

        N_rays, N_samples = refs.shape[:2]
        encoded = self.view_embedder(**inputs, refs=refs)
        d = encoded['d'].reshape(N_rays, N_samples, self.view_dims)

        # apply positional encoding (PE)
        d_pe = self.view_posi_enc(d, dists=encoded_pts.get('v', None))[0]

        view_inputs = d_pe
        if self.use_framecodes:
            N_rays, N_samples = refs.shape[:2]
            # expand from (N_rays, ...) to (N_rays, N_samples, ...)
            cam_idxs = inputs['cam_idxs']
            cam_idxs = cam_idxs.reshape(N_rays, 1, -1).expand(-1, N_samples, -1)
            framecodes = self.framecodes(cam_idxs.reshape(N_rays * N_samples, -1))
            framecodes = framecodes.reshape(N_rays, N_samples, -1)
            view_inputs = torch.cat([view_inputs, framecodes], dim=-1)

        view_inputs = view_inputs.flatten(end_dim=1)
        if 'valid_idxs' in encoded_pts:
            view_inputs = view_inputs[encoded_pts['valid_idxs']]

        return view_inputs, encoded

    def fill_prediction(self, preds, shape, valid_info):
        """ Create a full tensor from valid prediction.
        In evaluate_pts, we may avoid computation on some points that do not require prediction.
        The preds tensor shape is thus varying. This function turn the prediction into fixed size
        so that the prediction can be processed with batch operation.

        Args
        ----
        preds: [:, pred_size], first dimension varying
        shape: the actual shape of the first dimension in preds
        valid_info: information that requires to map preds back to a tensor of shape [*shape, pred_size]
        """
        output_dim = preds.shape[-1]

        if valid_info is not None and 'valid_idxs' in valid_info:
            valid_idxs = valid_info['valid_idxs']
            invalid_idxs = torch.where(valid_info['vol_invalid'].sum(-1).reshape(-1) == 24)[0]
            outputs = torch.zeros(np.prod(shape), output_dim)
            # by default force nothing there in the empty space

            # outputs[invalid_idxs, -1] = 0.
            outputs[valid_idxs] += preds
            outputs = outputs.reshape(*shape, output_dim)
        else:
            outputs = preds.reshape(*shape, output_dim)

        return outputs

    def inference_sigma(self, density_inputs):
        h = self.forward_density(density_inputs)
        sigma = self.sigma_linear(h)
        return sigma, h

    def inference_rgb(self, view_inputs, density_feature, rgb_eps=0.001):
        rgb = self.forward_view(view_inputs, density_feature)
        rgb = torch.sigmoid(rgb) * (1 + 2 * rgb_eps) - rgb_eps
        return rgb

    def forward_density(self, density_inputs):
        h = density_inputs

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h, inplace=True)
            if i in self.skips:
                h = torch.cat([density_inputs, h], -1)
        return h

    def forward_view(self, view_inputs, density_feature):
        # produce features for color/radiance
        feature = self.feature_linear(density_feature)
        h = torch.cat([feature, view_inputs], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h, inplace=True)

        return self.rgb_linear(h)

    def forward_geometry(
            self,
            render_data,
            chunk=1024 * 64 * 5,
            return_gradient=False,
            geom_extra_rets=[],
            **kwargs
    ):

        assert 'pts' in render_data, f'Query locations need to be provided for forward_geometry'
        if 'kp3d' not in render_data:
            kp3d, skts = calculate_kinematic(
                self.rest_pose,
                render_data['bones'],
                render_data.get('root_locs', None),
            )
            render_data['kp3d'] = kp3d
            render_data['skts'] = skts

        pts = render_data['pts']
        kp3d = render_data['kp3d']
        skts = render_data['skts']
        bones = render_data['bones']

        # only deals with one pose at a time
        assert len(kp3d) == 1
        assert len(skts) == 1
        assert len(bones) == 1

        pts_shape = pts.shape
        pts = pts.reshape(1, -1, 3)
        N_samples = pts.shape[1]

        # pre-allocate
        density = torch.zeros(N_samples, 1)
        sigma = torch.zeros(N_samples, 1)
        extra_dict = {k: None for k in geom_extra_rets}
        gradient = None
        if return_gradient:
            gradient = torch.zeros(N_samples, 3)
        for i in range(0, N_samples, chunk):
            chunk_pts = pts[:, i:i + chunk]
            geom_inputs = {
                'kp3d': kp3d,
                'skts': skts,
                'bones': bones,
                'N_unique': 1,
                'pts': chunk_pts,
            }
            _, preds = self.evaluate_pts(
                geom_inputs,
                geometry_only=True,
                geom_extra_rets=geom_extra_rets,
            )
            if preds is None:
                continue
            preds = {k: v.detach() for k, v in preds.items()}
            if 'valid_idxs' in preds:
                density[i + preds['valid_idxs']] = preds['density']
                sigma[i + preds['valid_idxs']] = preds['sigma']
                if gradient is not None:
                    pred_gradient = preds['surface_gradient'].flatten(end_dim=1)
                    pred_gradient = pred_gradient[preds['valid_idxs']]
                    gradient[i + preds['valid_idxs']] = pred_gradient

                for k in geom_extra_rets:
                    extra_tensor = preds[k]
                    if extra_dict[k] is None:
                        extra_dict[k] = torch.zeros(N_samples, extra_tensor.shape[-1])
                    extra_dict[k][i + preds['valid_idxs']] = extra_tensor
            else:
                density[i:i + chunk] = preds['density']
                sigma[i:i + chunk] = preds['sigma']
                if gradient is not None:
                    gradient[i:i + chunk] = preds['surface_gradient'].flatten(end_dim=1)

                for k in geom_extra_rets:
                    extra_tensor = preds[k]
                    if extra_dict[k] is None:
                        extra_dict[k] = torch.zeros(N_samples, extra_tensor.shape[-1])
                    extra_dict[k][i:i + chunk] = extra_tensor

        outputs = {
            'density': density,
            'sigma': sigma,
            **extra_dict,
        }

        if gradient is not None:
            outputs.update(gradient=gradient.reshape(*pts_shape[:-1], 3))
        return outputs

    def collect_encoded(self, encoded_pts, encoded_views):
        ret = {}
        return ret

    def collect_outputs(self, ret, ret0=None, encoded=None, encoded0=None):
        """ Collect outputs into a dictionary for loss computation/rendering

               Parameter
               ---------
               ret: dictionary of the fine-level rendering
               ret0: dictionary of the coarse-level rendering
               encoded: dictionary of the fine-level model info
               encoded0: dictionary of the corase-level model info

               """

        collected = {'rgb_map': ret['rgb_map'], 'disp_map': ret['disp_map'],
                     'acc_map': ret['acc_map'], }
        if not self.training:
            return collected

        collected['T_i'] = ret['weights']
        collected['alpha'] = ret['alpha']

        if ret0 is not None:
            collected['rgb0'] = ret0['rgb_map']
            collected['disp0'] = ret0['disp_map']
            collected['acc0'] = ret0['acc_map']
            collected['alpha0'] = ret0['alpha']

        if encoded is not None and 'surface_gradient' in encoded and self.training:
            collected['surface_gradient'] = encoded['surface_gradient']
            if encoded0 is not None:
                collected['surface_gradient0'] = encoded0['surface_gradient']

        collected['vol_scale'] = self.get_axis_scale()  # self.get_Gbox_Sigma()
        return collected

    def raw2outputs(
            self,
            raw,
            z_vals,
            rays_d,
            bgs=None,
            **kwargs
    ):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
            bgs: [num_rays, 3]. Background color
            act_fn: activation function for the density
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = raw[..., :3]

        alpha = 1. - torch.exp(-raw[..., 3] * dists)

        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                          :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1) + 1e-10))

        invalid_mask = torch.ones_like(disp_map)
        invalid_mask[torch.isclose(weights.sum(-1), torch.tensor(0.))] = 0.
        disp_map = disp_map * invalid_mask

        acc_map = torch.minimum(torch.sum(weights, -1), torch.tensor(1.))

        if bgs is not None:
            rgb_map = rgb_map + (1. - acc_map)[..., None] * bgs

        return {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map,
                'weights': weights, 'alpha': alpha}

    def _empty_outputs(self, batch):
        N_rays = len(batch['rays_o'])
        empty = torch.zeros(N_rays, 3)
        rgb_empty = batch.get('bgs', empty)
        return {'rgb_map': rgb_empty, 'disp_map': empty[:, 0], 'acc_map': empty[:, 0]}

    def to_density(self, sigma):
        """ Turn sigma from raw prediction (value unbounded) to density (value >= 0.)
        """
        noise = 0.
        if self.training and self.density_noise_std > 0.:
            noise = torch.randn_like(sigma) * self.density_noise_std

        sigma = sigma + noise

        return F.relu(sigma, inplace=True)
