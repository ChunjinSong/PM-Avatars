import os
import glob
import hydra
import imageio
import numpy as np
import logging, logging.config, yaml
import torch.nn as nn
from tqdm import tqdm, trange
from core.trainer import Trainer

from torch.utils.tensorboard import SummaryWriter

from core.dataset.load_data import *
from core.utils.evaluation_helpers import evaluate_metric

os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['TORCH_CUDA_IPC_MEMORY_POOL_MAX_SPLIT_SIZE_MB'] = '0'
CONFIG_BASE = 'configs/'
logger = None

def prepare_render_data(
        render_data,
        required=['c2ws', 'skts', 'bones', 'kp3d', 'hwf', 'center', 'cam_idxs', 'bg_idxs', 'imgs', 'fgs']
):
    render_tensor = {}
    for k, v in render_data.items():
        if k not in required:
            continue
        if v is None:
            continue
        if k == 'bg_idxs':
            bg_imgs = render_data['bgs'][v]
            render_tensor['bgs'] = torch.tensor(bg_imgs).cpu()
            continue

        if isinstance(v, np.ndarray):
            render_tensor[k] = torch.tensor(v)
        elif isinstance(v, tuple):
            render_tensor[k] = [torch.tensor(v_) for v_ in v]
        else:
            raise NotImplementedError(f'{k} is in unknown datatype')
    return render_tensor

def build_model(config, data_attrs, ckpt=None):
    n_framecodes = data_attrs["n_views"]
    # don't use dataset near far: you will derive it from cyl anyway
    data_attrs.pop('far', None)
    data_attrs.pop('near', None)
    model = instantiate(config, **data_attrs, n_framecodes=n_framecodes, _recursive_=False)

    if ckpt is not None:
        ret = model.load_state_dict(ckpt['model'])
        tqdm.write(f'ckpt loading: {ret}')
        if logger is not None:
            logger.info(f'ckpt loading: {ret}')


    return model

def find_ckpts(config, log_path, ckpt_path=None):

    start = 0
    if ckpt_path is None and 'ckpt_path' in config:
        ckpt_path = config.get('ckpt_path')
    elif ckpt_path is None:
        ckpt_paths = sorted(glob.glob(os.path.join(log_path, '*.th')))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]

    if ckpt_path is None:
        tqdm.write(f'No checkpoint found: start training from scratch')
        logger.info(f'No checkpoint found: start training from scratch')
        return None, start

    ckpt = torch.load(ckpt_path)
    start = ckpt['global_iter']
    tqdm.write(f'Resume training from {ckpt_path}, starting from step {start}')
    if logger is not None:
        logger.info(f'Resume training from {ckpt_path}, starting from step {start}')
    return ckpt, start

def train(config):
    # create directory and save config
    expname, basedir = config.expname, config.basedir
    log_path = os.path.join(basedir, expname)
    os.makedirs(log_path, exist_ok=True)
    OmegaConf.save(config=config, f=os.path.join(log_path, 'config.yaml'))

    img_path = os.path.join(log_path, 'image')
    os.makedirs(img_path, exist_ok=True)

    with open("./configs/logconf.yaml", "r") as f:
        dict_conf = yaml.safe_load(f)
    dict_conf['handlers']['fh']['filename'] = os.path.join(log_path,  dict_conf['handlers']['fh']['filename'])
    logging.config.dictConfig(dict_conf)
    global logger
    logger = logging.getLogger()

    # tensorboard
    writer = SummaryWriter(log_path)

    # prepare dataset and relevant information
    data_info = build_dataloader(config)
    dataloader = data_info['dataloader']
    render_data = data_info['render_data']
    data_attrs = data_info['data_attrs']

    # build model
    ckpt, start = find_ckpts(config, log_path)
    model = build_model(config.model, data_attrs, ckpt)
    model = nn.DataParallel(model)

    logger.info(model)
    logger.info(f"#parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    trainer = Trainer(
                  config=config.trainer,
                  loss_config=config.losses,
                  full_config=config,
                  model=model,
                  ckpt=ckpt
              )

    # training loop
    start = start + 1
    data_iter = iter(dataloader)

    for i in trange(start, config.iters+1):
        batch = next(data_iter)
        training_stats = trainer.train_batch(batch, global_iter=i)

        # save periodically
        if (i % config.i_save) == 0:
            trainer.save_ckpt(global_iter=i, path=os.path.join(log_path, f'{i:07d}.th'))

        if (i % config.i_print) == 0:
            # logging
            trainer.save_ckpt(global_iter=i, path=os.path.join(log_path, f'latest.th'))
            to_print = ['total_loss', 'loss_rgb', 'psnr', 'lr', 'lr2', 'avg_norm']  # things to print out
            mem = torch.cuda.max_memory_allocated() / 1024. / 1024.
            output_str = expname
            output_str = f'{output_str}, Iter: {i:07d}'
            for k, v in training_stats.items():
                if k in to_print:
                    output_str = f'{output_str}, {k}: {v:.6f}'
                writer.add_scalar(f'Stats/{k}', v, i)
            for name, cur_para in model.module.named_parameters():
                if len(cur_para) == 0:
                    continue
                para_norm = torch.norm(cur_para.grad.detach(), 2)
                data_norm = torch.norm(cur_para.data.detach(), 2)
                writer.add_scalar('Grad/%s_norm' % name, para_norm, i)
                writer.add_scalar('Data/%s_norm' % name, data_norm, i)

            output_str = f'{output_str}, peak_mem: {mem:.6f}'
            tqdm.write(output_str)
            logger.info(output_str)

        if (i % config.i_testset) == 0:
            tqdm.write('Running validation data ...')
            logger.info('Running validation data ...')
            model.eval()
            render_tensor = prepare_render_data(render_data)

            preds = model.module(render_tensor, render_factor=config.render_factor, forward_type='render')
            model.train()
            img_path_i = os.path.join(img_path, str(i))
            os.makedirs(img_path_i, exist_ok=True)
            for i_img in range(preds['rgb_imgs'].size(dim=0)):
                img = preds['rgb_imgs'][i_img]

                path = os.path.join(img_path_i, str(i_img) + '.jpg')
                imageio.imwrite(path, (img.cpu().numpy() * 255).astype(np.uint8))

            metrics = evaluate_metric(
                preds['rgb_imgs'].to('cuda'),
                torch.tensor(render_data['imgs']),
                render_data['fgs'],
                render_tensor['c2ws'],
                render_tensor['kp3d'],
                render_data['hwf'],
                render_data['center'],
                render_factor=config.render_factor,
            )
            output_str = expname
            output_str = f'{output_str}, Iter: {i:07d}'
            for k, v in metrics.items():
                writer.add_scalar(f'Val/{k}', v, i)
                output_str = f'{output_str}, {k}: {v:.6f}'
            tqdm.write(output_str)
            logger.info(output_str)


@hydra.main(version_base='1.3', config_path=CONFIG_BASE, config_name='pm_avatar.yaml')
def cli(config):
    return train(config)

if __name__== '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.multiprocessing.set_start_method('spawn')
    torch.autograd.set_detect_anomaly(True)
    cli()

