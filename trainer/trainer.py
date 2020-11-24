import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from data_loader.mixup import mixup_criterion, mixup_data
from torch.autograd import Variable


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, mixup=False, alpha=1.0):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.mixup = self.config['other_args']['mixup']
        self.alpha = self.config['other_args']['alpha']
        print("Training using mixup = {} with alpha = {}".format(self.mixup, self.alpha))

        print('metrics functions are ', metric_ftns)
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # error debugging
            # if batch_idx < 6872:
            #     print(batch_idx)
            #     continue
            data, target = data.to(self.device), target.to(self.device)

            # mixup
            if self.mixup:
                data, target_a, target_b, lam = mixup_data(data, target,
                                                        self.alpha, use_cuda=True)
                data, target_a, target_b = map(Variable,
                                            (data, target_a, target_b))


            # print('sanity check', [data.min(), data.max(), data.shape, type(data[0,0,0])])
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.mixup:
                loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
            else:
                loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # try:
                #     self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                # except TypeError:
                #     self.writer.add_image('input', make_grid((data.cpu()[:,0,:,:]).unsqueeze(1), nrow=8, normalize=True))
            if batch_idx == self.len_epoch:
                break


        # sanity check 
        # print(target)

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # try:
                #     self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                # except TypeError:
                #     self.writer.add_image('input', make_grid(data.cpu()[:,0,:,:].unsqueeze(1), nrow=8, normalize=True))
        # add histogram of model parameters to the tensorboard
        # Here everything is being added. Hence large size of tf files

        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')


        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
