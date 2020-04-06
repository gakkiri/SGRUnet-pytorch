from model.Dnet import D
from model.Gnet import SGRU
from loss import CRNsLoss

import os, logging
import torch

try:
    import apex
    APEX = True
except:
    APEX = False


class Trainer(object):
    def __init__(self, config, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_net = SGRU()
        self.loss_net = D(config.dnet_slug)
        self.losser = CRNsLoss(config, self.loss_net)

        if config.resume:
            self.resume_model()
        else:
            self.optimizer = torch.optim.AdamW(self.train_net.parameters(), lr=config.lr)
            self.start_epoch = 1
            self.best_loss = 1e6

        if config.gpu:
            self.train_net = self.train_net.cuda()
            self.loss_net = self.loss_net.cuda()
            self.losser = self.losser.cuda()

        if config.apex and APEX:
            self.train_net, self.optimizer = apex.amp.initialize(self.train_net, self.optimizer, opt_level="O1",
                                                                 verbosity=0)
        self.config = config
        self.logger = self.init_logger()
        self.logger.info('Trainer OK!')

    def init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        handler = logging.FileHandler(os.path.join(self.config.log_dir, "log.txt"))
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
        return logger

    def write_log(self, avg_loss, epoch, step=None, loss=None, mode='TRAIN'):
        log = f'[{mode}]epoch: %3d' % (epoch)
        if step is not None:
            log += f'  step: %3d/{len(self.train_loader)}' % (step)
        if loss is not None:
            log += f'  loss: %.3f(%.3f)' % (loss, avg_loss)
        else:
            log += f'  *epoch loss: %.3f\n' % (avg_loss)
        self.logger.info(log)

    def train(self):
        self.logger.info('Start trainning...\n')
        for epoch in range(self.start_epoch, self.config.epoch + 1):
            loss = self.train_one_epoch(epoch)
            self.write_log(epoch, loss)

            if self.config.val:
                loss = self.val_one_epoch(epoch)
                self.write_log(epoch, loss, mode='EVAL')

            if epoch % self.config.save_interval == 0:
                self.save_model(epoch, loss < self.best_loss)

            self.best_loss = min(self.best_loss, loss)

    def train_one_epoch(self, epoch):
        self.train_net.train()
        total_loss = 0
        for step, (imgs_rgb, imgs_bw, _) in enumerate(self.train_loader):
            if self.config.gpu:
                imgs_rgb = imgs_rgb.cuda()
                imgs_bw = imgs_bw.cuda()
            self.optimizer.zero_grad()

            imgs_rgb_gen = self.train_net(imgs_bw)
            loss = self.losser(imgs_bw, imgs_rgb, imgs_rgb_gen)

            if self.config.apex and APEX:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if step % self.config.log_interval == 0:
                self.write_log(total_loss / (step + 1), epoch, step, loss.item())
        return total_loss / (step + 1)

    def val_one_epoch(self, epoch):
        self.train_net.eval()
        total_loss = 0
        with torch.no_grad():
            for step, (imgs_rgb, imgs_bw, _) in enumerate(self.val_loader):
                if self.config.gpu:
                    imgs_rgb = imgs_rgb.cuda()
                    imgs_bw = imgs_bw.cuda()

                imgs_rgb_gen = self.train_net(imgs_rgb)
                loss = self.losser(imgs_bw, imgs_rgb, imgs_rgb_gen)

                total_loss += loss.item()
                if step % self.config.log_interval == 0:
                    self.write_log(total_loss / (step + 1), epoch, step, loss.item(), mode='EVAL')
        return total_loss / (step + 1)

    def save_model(self, epoch, is_best=False):
        state = {
            'model': self.train_net.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer,
            'loss': self.best_loss.item()
        }
        if is_best:
            torch.save(state, os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth'))
        torch.save(state, os.path.join(self.config.checkpoint_dir, 'checkpoint.pth'))

    def resume_model(self):
        if self.config.resume_from_best:
            path = os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth')
        else:
            path = os.path.join(self.config.checkpoint_dir, 'checkpoint.pth')
        ckp = torch.load(path)
        model_static_dict = ckp['model']
        self.optimizer = ckp['optimizer']
        self.start_epoch = ckp['epoch']
        self.best_loss = ckp['loss']
        self.train_net.load_state_dict(model_static_dict)
