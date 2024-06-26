import glob
import torch
import lpips
import imageio
import os, cv2
import numpy as np
from smplx import SMPL
from smplx.lbs import vertices2joints
import torch.nn.functional as F
from pytorch_msssim import SSIM
from PIL import Image, ImageFont, ImageDraw
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .ray_utils import kp_to_valid_rays
from .skeleton_utils import axisang_to_rot, world_to_cam
from skimage.metrics import structural_similarity as compare_ssim

DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
}

# READERS
def read_tfevent(path, guidance=DEFAULT_SIZE_GUIDANCE):
    event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
    event_acc.Reload()
    return event_acc

def read_tag_scalars(tags, events):

    if not isinstance(events, list):
        events = [events]
    if not isinstance(tags, list):
        tags = [tags]

    return_dict = {}
    for tag in tags:
        return_dict[tag] = []
        return_dict[tag + "_steps"] = []
    return_dict["num_events"] = len(events)

    for event in events:
        for tag in tags:
            data_list = event.Scalars(tag)
            values = list(map(lambda x: x.value, data_list))
            steps = list(map(lambda x: x.step, data_list))
            return_dict[tag].append(values)
            return_dict[tag + "_steps"].append(steps)

    return return_dict

def read_events_from_paths(log_paths):
    events = []
    for log_path in log_paths:
        event_paths = glob.glob(os.path.join(log_path, "events.*"))
        event = None
        for event_path in event_paths:
            e = read_tfevent(event_path)
            if event is None:
                event = e
            # TODO: handle cases that have multiple events
        events.append(event)
    return events

def read_eval_result(log_path, dir_name="val_*_val", step=10000, prefix="Val"):
    """
    dir_name: directory name in the log path
    step: interval between the logged numbers
    """
    num_events = 0
    nonempty_paths = []
    scalar_dict = {f"{prefix}/PSNR": [], f"{prefix}/PSNR_steps": [],
                   f"{prefix}/SSIM": [], f"{prefix}/SSIM_steps": []}
    psnr_path = glob.glob(os.path.join(log_path, dir_name, "psnr.txt"))
    ssim_path = glob.glob(os.path.join(log_path, dir_name, "ssim.txt"))

    if len(psnr_path) < 1:
        return None

    psnr_path = psnr_path[0]
    ssim_path = ssim_path[0]

    with open(psnr_path, "r") as f:
        psnrs = []
        steps = []
        for i, line in enumerate(f.readlines()):
            psnrs.append(float(line))
            steps.append(step * (i + 1))
        scalar_dict[f"{prefix}/PSNR"].append(psnrs)
        scalar_dict[f"{prefix}/PSNR_steps"].append(steps)

    with open(ssim_path, "r") as f:
        ssims = []
        steps = []
        for i, line in enumerate(f.readlines()):
            ssims.append(float(line))
            steps.append(step * (i + 1))
        scalar_dict[f"{prefix}/SSIM"].append(ssims)
        scalar_dict[f"{prefix}/SSIM_steps"].append(steps)

    num_events += 1
    scalar_dict["num_events"] = num_events
    return scalar_dict

def get_best_values_n_steps(scalar_dict, tag, maximum=True):

    reduce_fn = np.argmax if maximum else np.argmin

    n_return = len(scalar_dict[tag])
    values = np.array(scalar_dict[tag])
    best_idx = reduce_fn(values, axis=-1)
    best_values = values[np.arange(len(values)), best_idx]
    best_steps = np.array(scalar_dict[tag+"_steps"])[np.arange(len(values)), best_idx]

    return best_values, best_steps

def retrieve_best_vid_files(log_paths, best_steps, keyword_str="_%06d", postfix="rgb.mp4"):
    vid_names = []

    for log_path, best_step in zip(log_paths, best_steps):
        search_path = os.path.join(log_path, (f"*{keyword_str}*{postfix}") % best_step)
        fn = glob.glob(search_path)
        if len(fn) > 1:
            for f in fn:
                if "text_" in f and ".mp4" in f:
                    os.remove(f)
            fn = [f for f in fn if "text_" not in f]
        try:
            assert len(fn) == 1, "Bad keyword string, multiple files are found!"
        except:
            import pdb; pdb.set_trace()
            print(0)
        vid_names.append(fn[0])
    return vid_names

def concat_vid(vid_names, output_name, nrows=1, ncols=None, texts=None,
               base_cmd="ffmpeg -y"):
    if texts is not None:
        if len(texts) != len(vid_names):
            import pdb; pdb.set_trace()
            print()
        assert len(texts) == len(vid_names), "Text lists should be as the same length as vid_names!"
        tmp_vid_names = []
        for vid_name, text in zip(vid_names, texts):
            tmp_vid_name = add_text_to_vid(vid_name, text)
            tmp_vid_names.append(tmp_vid_name)
        vid_names = tmp_vid_names
    if ncols is None:
        ncols = len(vid_names) // nrows
    vid_names = np.array(vid_names).reshape(nrows, ncols)

    # concat horizontally
    for j, row in enumerate(vid_names):
        cmd = base_cmd
        for name in row:
            cmd += f" -i {name}"
        if nrows == 1:
            # back out here if we only have one row
            cmd += f" -filter_complex hstack={len(row)} {output_name}"
            os.system(cmd)

            # clean up temporary text file
            if texts is not None:
                for vid_name in vid_names.reshape(-1):
                    os.remove(vid_name)
            return

        cmd += f" -filter_complex hstack={len(row)} {j}__tmp.mp4"
        os.system(cmd)

    # concat vertically
    cmd = base_cmd
    for j in range(len(vid_names)):
        cmd += f" -i {j}__tmp.mp4"
    cmd += f" -filter_complex vstack={len(vid_names)} {output_name}"
    os.system(cmd)

    # Clean up temporary videos
    for i in range(len(vid_names)):
        if os.path.exists(f"{i}__tmp.mp4"):
            os.remove(f"{i}__tmp.mp4")

    if texts is not None:
        for vid_name in vid_names.reshape(-1):
            if os.path.exists(vid_name):
                os.remove(vid_name)

def add_text_to_vid(vid_name, text,
                    font_type="DejaVuSans-Bold", font_size=16,
                    text_loc=(10, 30)):

    font = ImageFont.truetype(font_type, font_size)


    # setup name
    pd = os.path.dirname(vid_name)
    new_name = os.path.join(pd, "text_" + vid_name.split("/")[-1])

    # setup video read/write
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = cv2.VideoWriter(new_name, fourcc, 14, (512, 512))
    reader = cv2.VideoCapture(vid_name)
    while reader.isOpened():
        ret, frame = reader.read()
        if ret:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text(text_loc, text, font=font)
            text_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            vid.write(text_frame)
        else:
            break
    vid.release()
    reader.release()
    return new_name

def compute_psnr(render, gt):
    mse = np.mean((render - gt)**2)
    return -10 * np.log(mse) / np.log(10)

def evaluate_psnrs(render, gt, mask=None):
    render = render.copy()
    gt = gt.copy()
    # if mask is not None:
    #     x, y, w, h = cv2.boundingRect(mask * 255)
    #     render = render[y:y + h, x:x + w].copy()
    #     gt = gt[y:y + h, x:x + w].copy()
    # psnr = compute_psnr(render, gt)

    psnr = compute_psnr(render[mask > 0], gt[mask > 0])

    return psnr

def evaluate_psnrs2(render, gt, mask=None):
    render = render.copy()
    gt = gt.copy()
    if mask is not None:
        x, y, w, h = cv2.boundingRect(mask * 255)
        render = render[y:y + h, x:x + w].copy()
        gt = gt[y:y + h, x:x + w].copy()
    psnr = compute_psnr(render, gt)
    return psnr


def evaluate_ssims(render, gt, mask=None):
    render = render.copy()
    gt = gt.copy()

    if mask is not None:
        x, y, w, h = cv2.boundingRect(mask * 255)
        render = render[y:y + h, x:x + w].copy()
        gt = gt[y:y + h, x:x + w].copy()
    ssim = compare_ssim(render, gt, channel_axis=-1)
    return ssim


def evaluate_lpips(render, gt, lpips_vgg, lpips_alex, mask=None):
    render = render.copy()
    gt = gt.copy()
    if mask is not None:
        x, y, w, h = cv2.boundingRect(mask * 255)
        render = render[y:y + h, x:x + w].copy()
        gt = gt[y:y + h, x:x + w].copy()
    H, W, C = gt.shape

    render_tensor = torch.tensor(render).reshape(1, H, W, C).permute(0, 3, 1, 2).float().cuda()
    gt_tensor = torch.tensor(gt).reshape(1, H, W, C).permute(0, 3, 1, 2).float().cuda()
    # to [-1, 1]
    render_tensor = render_tensor * 2 - 1.
    gt_tensor = gt_tensor * 2 - 1.
    with torch.no_grad():
        d = lpips_vgg(gt_tensor, render_tensor).item()
        d_alex = lpips_alex(gt_tensor, render_tensor).item()

    return d, d_alex

def evaluate_metirc_img(img, img_gt, img_mask, lpips_vgg, lpips_alex):
    # score
    psnrs = evaluate_psnrs(img, img_gt, mask=img_mask)
    ssims = evaluate_ssims(img, img_gt, mask=img_mask[..., None].astype(np.uint8))
    lpips, lpips_alex = evaluate_lpips(img, img_gt, mask=img_mask[..., None].astype(np.uint8), lpips_vgg=lpips_vgg, lpips_alex=lpips_alex)
    return {'psnr': psnrs, 'ssim': ssims, 'lpips': lpips, 'lpips_alex': lpips_alex}


def evaluate_metric_imgs(imgs, imgs_gt, imgs_mask):
    lpips_vgg = lpips.LPIPS(net='vgg', eval_mode=True).cuda()
    lpips_alex = lpips.LPIPS(net='alex', eval_mode=True).cuda()
    assert len(imgs) == len(imgs_gt)
    assert len(imgs_mask) == len(imgs_gt)
    result_sum = {'psnr': 0., 'ssim': 0., 'lpips': 0., 'lpips_alex': 0.}
    for img, img_gt, img_mask in zip(imgs, imgs_gt, imgs_mask):
        result = evaluate_metirc_img(img, img_gt, img_mask, lpips_vgg, lpips_alex)
        for k in result.keys():
            result_sum[k] += result[k]
    for k in result_sum.keys():
        result_sum[k] = result_sum[k]/len(imgs)
    return result_sum

def evaluate_metric_h36m(method_images, path_eval):
    gt_mask_paths = sorted(glob.glob(os.path.join(path_eval, '*_mask.png')))
    gt_img_paths = sorted([f for f in glob.glob(os.path.join(path_eval, '*.png'))
                           if 'mask.png' not in f])
    # img_paths = sorted(glob.glob(os.path.join(path_img, '*.png')))

    gt_images = np.array([imageio.imread(p) for p in gt_img_paths]).astype(np.float32) / 255.
    gt_masks = np.array([imageio.imread(p) for p in gt_mask_paths]).astype(np.float32) / 255.
    # method_images = np.array([imageio.imread(p) for p in img_paths]).astype(np.float32) / 255.
    method_images = np.array(method_images).astype(np.float32) / 255.

    if gt_images.shape[1] == 1002:
        gt_images = gt_images[:, 1:-1]
        gt_masks = gt_masks[:, 1:-1]
    if method_images.shape[1] == 1002:
        method_images = method_images[:, 1:-1]

    result = evaluate_metric_imgs(method_images, gt_images, gt_masks)
    return result


def evaluate_metirc_img2(img, img_gt, img_mask, lpips_vgg, lpips_alex):
    # score
    psnrs = evaluate_psnrs2(img, img_gt, mask=img_mask[..., None].astype(np.uint8))
    ssims = evaluate_ssims(img, img_gt, mask=img_mask[..., None].astype(np.uint8))
    lpips, lpips_alex = evaluate_lpips(img, img_gt, mask=img_mask[..., None].astype(np.uint8), lpips_vgg=lpips_vgg, lpips_alex=lpips_alex)
    return {'psnr': psnrs, 'ssim': ssims, 'lpips': lpips, 'lpips_alex': lpips_alex}


def evaluate_metric_imgs2(imgs, imgs_gt, imgs_mask):
    lpips_vgg = lpips.LPIPS(net='vgg', eval_mode=True).cuda()
    lpips_alex = lpips.LPIPS(net='alex', eval_mode=True).cuda()
    assert len(imgs) == len(imgs_gt)
    assert len(imgs_mask) == len(imgs_gt)
    result_sum = {'psnr': 0., 'ssim': 0., 'lpips': 0., 'lpips_alex': 0.}
    for img, img_gt, img_mask in zip(imgs, imgs_gt, imgs_mask):
        result = evaluate_metirc_img2(img, img_gt, img_mask, lpips_vgg, lpips_alex)
        for k in result.keys():
            result_sum[k] += result[k]
    for k in result_sum.keys():
        result_sum[k] = result_sum[k]/len(imgs)
    return result_sum

def evaluate_metric_mocap(method_images, gt_images, gt_masks):
    method_images = np.array(method_images).astype(np.float32) / 255.
    gt_images = np.array(gt_images).astype(np.float32) / 255.
    gt_masks = np.array(gt_masks).astype(np.float32) / 255.
    result = evaluate_metric_imgs2(method_images, gt_images, gt_masks)
    return result


def evaluate_metric_perfcap(method_images, gt_images, gt_masks):
    method_images = np.array(method_images).astype(np.float32) / 255.
    gt_images = np.array(gt_images).astype(np.float32) / 255.
    gt_masks = np.array(gt_masks).astype(np.float32) / 255.
    result = evaluate_metric_imgs2(method_images, gt_images, gt_masks)
    return result

@torch.no_grad()
def evaluate_metric(
        pred_rgbs,
        gt_rgbs,
        gt_masks,
        c2ws,
        kp3d,
        hwf,
        centers,
        render_factor=0,
    ):

    # initialize modules for evaluation
    ssim_eval = SSIM(size_average=False)
    lpips_model = lpips.LPIPS(net='vgg', eval_mode=True)

    # get valid_idxs to create a bounding box mask
    H, W, focals = hwf
    _, valid_idxs, _, _ = kp_to_valid_rays(
                              c2ws, 
                              H, 
                              W, 
                              focals,
                              centers=centers, 
                              kps=kp3d,
                          )
    # assume all images have the same resolution
    H = H if isinstance(H, int) else H[0]
    W = W if isinstance(W, int) else W[0]


    bbox_masks = np.zeros((len(valid_idxs), H * W, 1), dtype=np.float32)
    for i in range(len(valid_idxs)):
        valid_idx = valid_idxs[i].cpu()
        bbox_masks[i, valid_idx] = 1
    bbox_masks = np.array(bbox_masks).reshape(-1, H, W, 1)

    # remove images without any person in it
    valid_imgs = np.where(gt_masks.reshape(gt_masks.shape[0], -1).sum(-1) > 0)[0]
    pred_rgbs = pred_rgbs[valid_imgs]
    gt_rgbs = gt_rgbs[valid_imgs]
    gt_masks = gt_masks[valid_imgs]
    bbox_masks = bbox_masks[valid_imgs] 

    # upsample the image to match resolution if needed
    if render_factor > 0:
        # (N, H, W, C) -> (N, C, H, W)
        pred_rgbs = pred_rgbs.permute(0, 3, 1, 2)
        pred_rgbs = F.interpolate(pred_rgbs, size=gt_rgbs.shape[1:3], mode='bilinear',
                                align_corners=False)

    # (N, H, W, C) -> (N, C, H, W)
    gt_rgbs = gt_rgbs.permute(0, 3, 1, 2)

    ssim, sqr_diff = [], []
    # In case GPU memory blows up
    for pred_rgb, gt_rgb in zip(pred_rgbs, gt_rgbs):
        ssim.append(ssim_eval(pred_rgb[None], gt_rgb[None]).cpu())
        sqr_diff.append((pred_rgb[None] - gt_rgb[None]).pow(2.).cpu())
    ssim = torch.cat(ssim)
    sqr_diff = torch.cat(sqr_diff)
    # (N, C, H, W) -> (N, H, W, C)
    ssim = ssim.permute(0, 2, 3, 1).numpy()
    sqr_diff = sqr_diff.permute(0, 2, 3, 1).numpy()

    # Part 1. metrics on bounding boxes
    # compute LPIPS within bounding boxes
    lpips_vals = []
    for i, (bbox_mask, gt_rgb, pred_rgb) in enumerate(zip(bbox_masks, gt_rgbs, pred_rgbs)):
        rx, ry, rw, rh = cv2.boundingRect(bbox_mask.astype(np.uint8))

        # (C, H, W), normalize to [-1., 1.]
        cropped_gt = gt_rgb[:, ry:ry+rh, rx:rx+rw] 
        cropped_rgb = pred_rgb[:, ry:ry+rh, rx:rx+rw] 
        lpips_val = lpips_model(cropped_gt[None], cropped_rgb[None]).item()
        lpips_vals.append(lpips_val)
    lpips_avg = np.mean(lpips_vals)

    # values for pixels that are within a bounding box
    denom = np.maximum(bbox_masks.reshape(len(c2ws), -1).sum(-1) * 3., 1.)# avoid dividing by zero

    bbox_psnr = -10. * np.log10((sqr_diff * bbox_masks[..., :1]).reshape(len(c2ws), -1).sum(-1) / denom)
    bbox_psnr[bbox_psnr == np.inf] = 0.
    bbox_psnr = bbox_psnr.mean()

    bbox_ssim = (ssim * bbox_masks[..., :1]).reshape(len(c2ws), -1).sum(-1) / denom
    bbox_ssim[bbox_ssim == np.inf] = 0.
    bbox_ssim = bbox_ssim.mean()


    # Part 2. metrics on foreground pixels (note: this is stricter than bbox mask)
    denom = np.maximum(gt_masks.reshape(len(c2ws), -1).sum(-1) * 3., 1.)

    fg_psnr = -10. * np.log10((sqr_diff * gt_masks[..., :1]).reshape(len(c2ws), -1).sum(-1) / denom)
    fg_psnr[fg_psnr == np.inf] = 0.
    fg_psnr = fg_psnr.mean()

    fg_ssim = (ssim * gt_masks[..., :1]).reshape(len(c2ws), -1).sum(-1) / denom
    fg_ssim[fg_ssim == np.inf] = 0.
    fg_ssim = fg_ssim.mean()

    return {"psnr_bbox": bbox_psnr, "ssim_bbox": bbox_ssim, "psnr_fg": fg_psnr, "ssim_fg": fg_ssim, 'lpips': lpips_avg}

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """
    n,m = X.shape
    ny,my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)
    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    if reflection !=  'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform

class Criterion_MPJPE(torch.nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_batch, label_batch):
        diff = torch.norm(pred_batch - label_batch, p=2, dim=-1)
        if self.reduction == 'mean':
            metric = diff.mean()
        elif self.reduction == 'sum':
            metric = diff.sum()
        else:
            metric = diff
        return metric

class Criterion3DPose_ProcrustesCorrected(torch.nn.Module):
    """
    Normalize translaion, scale and rotation in the least squares sense, then apply the specified criterion
    """
    def __init__(self, criterion):
        super(Criterion3DPose_ProcrustesCorrected, self).__init__()
        self.criterion = criterion
    def forward(self, pred_batch, label_batch):
        #Optimal scale transform
        preds_procrustes = []
        batch_size = pred_batch.size()[0]
        num_joints = pred_batch.size()[-2]
        num_dim = pred_batch.size()[-1]
        assert num_dim == 3
        for i in range(batch_size):
            d, Z, tform = procrustes(label_batch[i].data.cpu().numpy().reshape(num_joints, num_dim),
                                     pred_batch[i].data.cpu().numpy().reshape(num_joints, num_dim))
            preds_procrustes.append(Z.reshape((num_joints, num_dim)))
        pred_batch_aligned = torch.tensor(np.stack(preds_procrustes)).to(pred_batch.device)
        return self.criterion(pred_batch_aligned, label_batch), pred_batch_aligned

class Criterion3DPose_leastQuaresScaled(torch.nn.Module):
    """
    Normalize the scale in the least squares sense, then apply the specified criterion
    """
    def __init__(self, criterion):
        super(Criterion3DPose_leastQuaresScaled, self).__init__()
        self.criterion = criterion
    def forward(self, pred, label):
        #Optimal scale transform
        batch_size = pred.size()[0]
        pred_vec = pred.view(batch_size,-1)
        gt_vec = label.view(batch_size,-1)
        dot_pose_pose = torch.sum(torch.mul(pred_vec,pred_vec),1,keepdim=True)
        dot_pose_gt   = torch.sum(torch.mul(pred_vec,gt_vec),1,keepdim=True)
        s_opt = dot_pose_gt / dot_pose_pose
        s_opt = s_opt[..., None]
        return self.criterion.forward(s_opt*pred, label), s_opt*pred


class SMPLEvalHelper(SMPL):
    # steal from SPIN
    def __init__(self, *args, **kwargs):
        super(SMPLEvalHelper, self).__init__(*args, **kwargs)
        J_regressor_extra = np.load("smpl/data/J_regressor_h36m.npy")
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPLEvalHelper, self).forward(*args, **kwargs)
        h36m_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        return smpl_output, h36m_joints


SPIN_TO_CANON = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
H36M_TO_17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_14 = H36M_TO_17[:14]
@torch.no_grad()
def evaluate_pampjpe_from_smpl_params(gt_kps, betas, bones, bone_orders="xyz",
                                      ret_kp=False, ret_pck=False,
                                      align_kp=False, pck_threshold=150,
                                      reduction="mean",
                                      use_normalize=False):

    assert betas.dim() == 2
    if betas.shape[0] == 1:
        betas = betas.expand(len(gt_kps), -1)

    rots = axisang_to_rot(bones.view(-1, 3)).view(*bones.shape[:2], 3, 3)


    smpl = SMPLEvalHelper("smpl/SMPL_NEUTRAL.pkl").to(rots.device)
    _, pred_kps = smpl(betas=betas,
                       body_pose=rots[:, 1:],
                       global_orient=rots[:, :1],
                       pose2rot=False)
    pred_kps = pred_kps[:, SPIN_TO_CANON] # to the same scale
    # H36M TO 14
    #pred_kps = pred_kps[:, H36M_TO_14] # to the same scale
    #print("TO 14")
    #gt_kps = gt_kps[:, :14]

    mpjpe_crit = Criterion_MPJPE(reduction=reduction).to(rots.device)
    pampjpe_crit = Criterion3DPose_ProcrustesCorrected(mpjpe_crit).to(rots.device)

    if use_normalize:
        mpjpe_crit = Criterion3DPose_leastQuaresScaled(mpjpe_crit)

    pampjpe, aligned_kps = pampjpe_crit(pred_kps,
                           torch.FloatTensor(gt_kps).to(pred_kps.device))

    gt_kps_trans = gt_kps.copy()
    pred_kps_trans = pred_kps.clone()
    gt_kps_trans = gt_kps_trans - gt_kps_trans[:, 14:15]
    pred_kps_trans = pred_kps_trans - pred_kps_trans[:, 14:15]
    #gt_kps_trans = gt_kps_trans - gt_kps_trans[:, :1]
    #pred_kps_trans = pred_kps_trans - pred_kps_trans[:, :1]
    mpjpe = mpjpe_crit(pred_kps_trans,
                       torch.FloatTensor(gt_kps_trans / 1000).to(pred_kps.device))
    if use_normalize:
        mpjpe = mpjpe[0]

    mpjpe  = mpjpe * 1000

    if not ret_kp:
        return pampjpe, mpjpe
    if align_kp:
        return pampjpe, mpjpe, aligned_kps
    if ret_pck:
        # in /1000 scale to avoid numerical issue
        pck_threshold = pck_threshold
        pck = (pampjpe < pck_threshold).float().mean()

        thresholds = torch.linspace(0, 150, 31).tolist()
        auc = []
        for i, t in enumerate(thresholds):
            pck_at_t = (pampjpe < t).float().mean().item()
            auc.append(pck_at_t)

        return pampjpe, mpjpe, pck, np.mean(auc)


    def pck(actual, expected, included_joints=None, threshold=150):
        dists = euclidean_losses(actual, expected)
        if included_joints is not None:
            dists = dists.gather(-1, torch.LongTensor(included_joints))
        return (dists < threshold).double().mean().item()

    return pampjpe, mpjpe, pred_kps

def estimates_to_kp2ds(kps, exts, img_height, img_width, focals,
                       pose_scale=1.0, pelvis_locs=None,
                       pelvis_order="xyz", our_exts=True):
    """
    our_exts: if the extrinsic is in our coordinate system
    """

    assert kps.shape[-2] == 17

    if pelvis_locs is not None:
        if pelvis_order == "xyz":
            kps = kps.copy()
            pelvis_locs = pelvis_locs.copy()
            pelvis_locs[..., 1:] *= -1
        kps[..., 14, :] = pelvis_locs[:, 0]

    kps = kps * pose_scale
    if our_exts:
        kps[..., 1:] *= -1
    kp2ds = np.array(
                [world_to_cam(kp, ext, img_height, img_width, focal)
                 for (kp, ext, focal) in zip(kps, exts, focals)]
            )

    return kp2ds

