from core.losses import *

def to_device(data, device='cuda'):
    data_device = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            data_device[k] = v.to(device)
        else:
            data_device[k] = v
    return data_device

def decay_optimizer_lr(
        init_lr,
        init_lr_freq,
        decay_steps,
        decay_rate, 
        optimizer,
        global_step=None, 
    ):

    optim_step = global_step

    new_lrate = init_lr * (decay_rate ** (optim_step / decay_steps))
    new_lrate_freq = init_lr_freq * (decay_rate ** (optim_step / decay_steps))
    lr_bound = 1e-5
    new_lrate = new_lrate if new_lrate > lr_bound else lr_bound
    new_lrate_freq = new_lrate_freq if new_lrate_freq > lr_bound else lr_bound

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate
    optimizer.param_groups[0]['lr'] = new_lrate_freq
    return new_lrate, new_lrate_freq

@torch.no_grad()
def get_gradnorm(module):
    total_norm  = 0.0
    cnt = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        cnt += 1
    avg_norm = (total_norm / cnt) ** 0.5
    total_norm = total_norm ** 0.5
    return total_norm, avg_norm

class Trainer(object):
    """ For training models
    """
    def __init__(
        self, 
        config,
        loss_config,
        full_config,
        model,
        ckpt=None,
        **kwargs,
    ):
        self.config = config
        self.loss_config = loss_config
        self.full_config = full_config
        self.model = model

        # initialize optimizerni
        self.init_optimizer(ckpt)

        # initialize loss function
        self.init_loss_fns()

    def init_optimizer(self, ckpt=None):

        self.lr_basic = self.config.optim.lr
        self.lr_freq = self.full_config.model.lr_freq

        pts_enc_pose = 'pts_enc_pose'
        pts_enc_point = 'pts_enc_point'
        graph_net = 'graph_net'
        win_fun = 'win_fun'
        pose_extractor = 'pose_extractor'
        modulator = 'modulator'

        ### Add regularization to the backbone parameters
        freq_params = list(filter(lambda kv: graph_net in kv[0] or
                                             pose_extractor in kv[0] or
                                             win_fun in kv[0] or
                                             modulator in kv[0] or
                                             pts_enc_point in kv[0] or
                                            pts_enc_pose in kv[0],
                                 self.model.named_parameters()))

        other_params = list(filter(lambda kv: graph_net not in kv[0] and
                                              pose_extractor not in kv[0] and
                                              win_fun not in kv[0] and
                                              modulator not in kv[0] and
                                              pts_enc_point not in kv[0] and
                                              pts_enc_pose not in kv[0],
                                   self.model.named_parameters()))

        freq_params = [d[1] for d in freq_params]
        other_params = [d[1] for d in other_params]

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": freq_params,
                    "lr": self.lr_freq
                },
                {
                    "params": other_params,
                    "lr": self.lr_basic
                }
            ])

        if ckpt is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self.optimizer.zero_grad()
    
    def init_loss_fns(self):
        self.loss_fns = [
            eval(k)(**v)
        for k, v in self.loss_config.items()]
    
    def train_batch(self, batch, global_iter=1):

        device_cnt = 1
        if isinstance(self.model, nn.DataParallel):
            if len(self.model.device_ids) > 1:
                device_cnt = len(self.model.device_ids)

        # Step 1. model prediction
        batch = to_device(batch, 'cuda')
        batch['N_unique'] = self.full_config.N_sample_images // device_cnt
        batch['device_cnt'] = device_cnt
        preds = self.model(batch)

        # Step 2. compute loss
        # TODO: used to have pose-optimization here ..
        loss, stats = self.compute_loss(batch, preds, global_iter=global_iter)

        # clean up after step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        total_norm, avg_norm = get_gradnorm(self.model)
        self.optimizer.zero_grad()

        # Step 3. post-update stuff

        # change/renew optimizer if needed

        # change learning rate
        new_lr, new_lrate_freq = decay_optimizer_lr(
            init_lr=self.config.optim.lr,
            init_lr_freq=self.full_config.model.lr_freq,
            decay_steps=self.config.lr_sched.decay_steps,
            decay_rate=self.config.lr_sched.decay,
            optimizer=self.optimizer,
            global_step=global_iter,
        )
        stats.update(lr=new_lr)
        stats.update(lr2=new_lrate_freq)
        stats.update(avg_norm=avg_norm)
        # TODO: A-NeRF cutoff update

        return stats
    
    def compute_loss(self, batch, preds, global_iter=1):

        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        total_loss = torch.tensor(0.0)
        loss_stats = {}
        for loss_fn in self.loss_fns:
            loss, loss_stat = loss_fn(batch, preds, global_iter=global_iter, model=model)
            total_loss += loss
            loss_stats.update(**loss_stat)
        

        # get extra stats that's irrelevant to loss
        loss_stats.update(psnr=img2psnr(preds['rgb_map'], batch['target_s']).item())
        if 'rgb0' in preds:
            loss_stats.update(psnr0=img2psnr(preds['rgb0'], batch['target_s']).item())
        loss_stats.update(alpha=preds['acc_map'].mean().item())
        if 'acc_map0' in preds:
            loss_stats.update(alpha0=preds['acc0'].mean().item())
        loss_stats.update(total_loss=total_loss.item())
        
        return total_loss, loss_stats
    
    def save_ckpt(self, global_iter, path):

        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'global_iter': global_iter,
            },
            path,
        )
        