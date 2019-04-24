import sys, pickle
import os
from PIL import Image
import torch, PIL
import numpy as np
import torch, time
import torch.optim
import torch.utils.data
import torch.nn as nn
import utility
import torch.backends.cudnn as cudnn
import torch.nn.init
import models
import progressbar


class Engine(object):
    def __init__(self, args):
        cudnn.benchmark = True
        self.gpus = torch.cuda.device_count()
        self.epoch = 0
        self.args = args
        self.initDataloader(args)
        self.model, self.optimizer = self.initModelOptimizer(args)
        if args['continue'] > 0:
            self.loadModel()

    def initDataloader(self, args):
        if args['data']['dataset'] == 'movingMnist':
            from dataset.mm import MovingMNIST
            self.trainFolder = MovingMNIST(is_train=True,
                                           root=args['data']['dataroot'],
                                           n_frames_input=args['Numin'],
                                           n_frames_output=args['Numpre'],
                                           num_objects=args['data']['num_digits'])
            self.testFolder = MovingMNIST(is_train=False,
                                          root=args['data']['dataroot'],
                                          n_frames_input=args['Numin'],
                                          n_frames_output=args['Numpre'],
                                          num_objects=args['data']['num_digits'])
            self.trainLoader = torch.utils.data.DataLoader(self.trainFolder,
                                                           batch_size=args['data']['train']['batchSize'] * self.gpus,
                                                           shuffle=args['data']['train']['shuffle'],
                                                           num_workers=args['data']['train']['numWorkers'] * self.gpus)
            self.testLoader = torch.utils.data.DataLoader(self.testFolder,
                                                          batch_size=args['data']['test']['batchSize'] * self.gpus,
                                                          shuffle=args['data']['test']['shuffle'],
                                                          num_workers=args['data']['test']['numWorkers'] * self.gpus)

    def initModelOptimizer(self, args):

        print(f'Initializing models & optimizer... \r', end='')
        self.best = 1e6
        self.best_rmse = 1e6
        self.best_mae = 1e6
        self.best_rmse_index = 0
        self.best_mae_index = 0
        self.best_index = 0

        with utility.Timer() as t:
            str = args['model']['arch']
            inch = args['data']['inChannel']

            model = models.__dict__[str](inch, inch)

            # print(model)
            # model = models.__dict__[args['model']['arch']]((50, 64), 100, [100], [(3, 3)], 1)

            model.apply(self.weights_init)
            model = torch.nn.DataParallel(model).cuda()

            if args['model']['optimizer'] in ['Adam', 'SGD', 'RMSprop']:
                optimizer = torch.optim.__dict__[args['model']['optimizer']](
                    # filter(lambda p: p.requires_grad, model.parameters()),
                    params=model.parameters(),
                    lr=args['model']['learningRate'],
                    weight_decay=args['model']['weightDecay'])

            self.cs = nn.BCELoss(reduction='sum').cuda()
            self.l2 = nn.MSELoss(reduction='sum').cuda()
            self.l1 = nn.L1Loss(reduction='sum').cuda()
        print('Model [%s] & Optimizer [%s] initialized. %.2fs' % (
            args['model']['arch'], args['model']['optimizer'], t.interval))
        return model, optimizer

    def loadModel(self):
        checkpoint = torch.load('ckpt/model_best.ckpt')
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        self.model.load_state_dict(checkpoint['net'])
        self.epoch = checkpoint['epoch'] + 1
        print('Model Best Loaded!')

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.02)

    def train(self):
        self.model.train()
        self.epoch += 1
        record = utility.Record()
        self.current_time = time.time()
        numIter = len(self.trainLoader)

        progress = progressbar.ProgressBar(max_value=numIter).start()
        for i, (idx, img, delta, _, _) in enumerate(self.trainLoader):
            progress.update(i + 1)
            inputVar = delta.cuda()
            targetVar = img.cuda()
            predTarget = self.model(inputVar, self.args['Numpre'], self.args['Numin'])
            if self.args['Loss'] == 'cs':
                loss = self.cs(predTarget, targetVar) / predTarget.size(0) / self.args['Numpre']
            elif self.args['Loss'] == 'l1':
                loss = self.l1(predTarget, targetVar) + self.l2(predTarget, targetVar)
                loss = loss / predTarget.size(0) / self.args['Numpre']

            self.optimizer.zero_grad()
            record.add(loss.data)
            loss.backward()

            self.optimizer.step()
        progress.finish()
        utility.clear_progressbar()
        print(f'--------------------------------------------------')
        print(f'The epoch :{self.epoch}')
        print(f'The epoch cost time:{(time.time()-self.current_time):.2f}s')
        print(f'The training loss is {record.mean():.4f}')

    def validate(self, epoch, training):
        num_pre = self.args['Numpre']
        if not training:
            self.loadModel()
        with torch.no_grad():
            self.model.eval()
            mse = utility.TestResult()
            mae = utility.TestResult()
            self.current_time = time.time()
            numIter = len(self.testLoader)

            progress = progressbar.ProgressBar(max_value=numIter).start()

            for i, (idx, img, delta, base, groudNameList) in enumerate(self.testLoader):
                progress.update(i + 1)

                img = img.cuda()
                tem_delta = base.cuda()
                inputVar = delta.cuda()
                predTarget = self.model(inputVar, self.args['Numpre'], self.args['Numin'])

                for j in range(num_pre):
                    if self.args['data']['sub']:
                        tem_delta = tem_delta + predTarget[:, j]
                    else:
                        tem_delta = predTarget[:, j]

                    MSE = self.l2(tem_delta, img[:, j]) / img.size(0)
                    MAE = self.l1(tem_delta, img[:, j]) / img.size(0)
                    mse.add(j, MSE.cpu() * img.size(0))
                    mae.add(j, MAE.cpu() * img.size(0))

                    # print(loss.data.cpu())
                # record.count += img.size(0)

                mse.count += img.size(0)
                mae.count += img.size(0)
                path = './'
                if (not os.path.exists(path + 'gen')):
                    os.mkdir(path + 'gen')
                    path = path + 'gen'

                if i == 8 and self.args['gif']:
                    # generate gif
                    timestep = self.args['Numpre']
                    gif = [[] for t in range(timestep)]
                    for t in range(timestep):
                        row = []
                        for r in range(0, 40, 10):
                            c_old = img[r, t].size(0)
                            h = img[r, t].size(1)
                            w = img[r, t].size(2)
                            c = int(np.sqrt(c_old))
                            row.append(
                                img[r, t].reshape(c, c, h, w).transpose(0, 2).transpose(1, 2).transpose(2, 3).reshape(
                                    h * c, w * c))
                            row.append(predTarget[r, t].reshape(c, c, h, w).transpose(0, 2).transpose(1, 2).transpose(2,
                                                                                                                      3).reshape(
                                h * c, w * c))
                        gif[t].append(row)
                        pname = path + '/_%d_%d.png' % (t, epoch)
                        utility.save_tensors_image(pname, row)

                    fname = path + '/_%d.gif' % (epoch)
                    utility.save_gif(fname, gif)
        progress.finish()
        utility.clear_progressbar()
        current = sum(mse.mean().cpu().numpy()) / num_pre
        rmse = np.sqrt(current)
        current_mae = sum(mae.mean().cpu().numpy()) / num_pre


        if self.best > current:
            self.best = current
            self.best_index = self.epoch
            if training:
                state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': self.epoch}
                torch.save(state, './ckpt/model_best.ckpt')

        if self.best_rmse > rmse:
            self.best_rmse = rmse
            self.best_rmse_index = self.epoch

        if self.best_mae > current_mae:
            self.best_mae = current_mae
            self.best_mae_index = self.epoch

        print(f'The Best MSE: {self.best, self.best_index, current}')
        print(f'The Best RMSE: {self.best_rmse, self.best_rmse_index, rmse}')
        print(f'The Best MAE: {self.best_mae, self.best_mae_index, current_mae}')
        # print(f'The epoch cost time:{(time.time() - self.current_time):.2f}s')
        print(f'The Validation mse is {mse.mean().cpu().numpy()[:num_pre]}')
        # print(f'The Validation FAR is {far.mean().cpu().numpy()[:num_pre]}')


