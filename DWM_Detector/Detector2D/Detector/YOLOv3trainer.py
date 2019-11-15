import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .models import *
from .tester import Tester
import sys 
sys.path.append('../')
from ..utils.datasets import *
from ..utils.utils import init_seeds, plot_images
from tqdm import tqdm
import time
from ..utils import torch_utils
import random
# warnings.filterwarnings("ignore")
import os
results_file = 'results.txt'
import numpy as np
import glob
import math
from ..config import opter


class ODtrianer():
    def __init__(self,status:dict,opt:opter,tester):
        self.status = status
        self.opt= opt
        self._init()
        self.Tester = tester

    def _initParameter(self):
        self.cfg = self.opt.cfg
        self.hyp = self.opt.hyp
        # data =opt.model
        self.img_size = self.opt.img_size
        self.epochs = self.opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
        self.batch_size = self.opt.batch_size
        self.accumulate = self.opt.acumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
        self.weights = self.opt.model_dir  # initial training weights
        self.device = torch_utils.select_device(self.opt.device)
        if 'pw' not in self.opt.arc:  # remove BCELoss positive weights
            self.hyp['cls_pw'] = 1.
            self.hyp['obj_pw'] = 1.

        # Initialize
        init_seeds()
        multi_scale = False

        if multi_scale:
            img_sz_min = round(self.img_size / 32 / 1.5) + 1
            img_sz_max = round(self.img_size / 32 * 1.5) - 1
            self.img_size = img_sz_max * 32  # initiate with maximum multi_scale size
            print('Using multi-scale %g - %g' % (img_sz_min * 32, self.img_size))

        # Configure run
        self.data_dict = self.opt.model
        # data_dict = parse_data_cfg(data)
        self.train_path = self.data_dict['train']
        self.lb_lists = self.data_dict['lb_list']
        self.nc = int(self.opt.classes)  # number of classes

        self.inter_plot_folder = os.path.join(self.opt.model_dir.split('.')[0], 'intermediate_results')
        self.tvimg_plot_folder = os.path.join(self.opt.model_dir.split('.')[0], 'tv_img')
        self.wdir = os.path.join(self.opt.model_dir.split('.')[0], 'trainOutPt')
        self.last = os.path.join(self.wdir, 'last.pt')
        self.best = os.path.join(self.wdir, 'best.pt')
        self.best_fitness = 0
        self.saved_bestn = 0
        self.epoch = 0
        self.cutoff=-1

    def _init(self):
        self._initParameter()
        self.init_model()
        self.initOptim()
        self._transfer()
        self._initLr('LambdaLR')
        self._datasetInit()

    def init_model(self):
        self.model = Darknet(self.cfg, arc=self.opt.arc).to(self.device)
        

    def initOptim(self):
        pg0, pg1 = [], []  # optimizer parameter groups
        for k, v in dict(self.model.named_parameters()).items():
            if 'Conv2d.weight' in k:
                pg1 += [v]  # parameter group 1 (apply weight_decay)
            else:
                pg0 += [v]  # parameter group 0

        if self.opt.optimizer=='adam':
            self.optimizer = optim.Adam(pg0, lr=self.hyp['lr0'])
            # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
        else:
            self.optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        del pg0, pg1

    def _transfer(self):
        if self.weights.endswith('.pt'):  # pytorch format
            # possible weights are 'last.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
            chkpt = torch.load(self.weights, map_location=self.device)
            self.status['load_weights'] = True
            # load model
            if self.opt.transfer == 1:
                chkpt['model'] = {k: v for k, v in chkpt['model'].items() if self.model.state_dict()[k].numel() == v.numel()}
                self.model.load_state_dict(chkpt['model'], strict=False)
                self.status['transfer'] = True
            else:
                self.model.load_state_dict(chkpt['model'])

            # load optimizer
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_fitness = chkpt['best_fitness']
                self.status['load_optimizer'] = True

                # load results
            if chkpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(chkpt['training_results'])  # write results.txt

            self.start_epoch = chkpt['epoch'] + 1
            # start_epoch = 0
            del chkpt

        elif len(self.weights) > 0:  # darknet format
            # possible weights are 'yolov3.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            self.cutoff = load_darknet_weights(self.model, self.weights)

        if self.opt.transfer:  # transfer learning edge (yolo) layers
            nf = int(self.model.module_defs[self.model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)

            if False:
                for p in self.optimizer.param_groups:
                    # lower param count allows more aggressive training settings: i.e. SGD ~0.1 lr0, ~0.9 momentum
                    p['lr'] *= 100  # lr gain
                    if p.get('momentum') is not None:  # for SGD but not Adam
                        p['momentum'] *= 0.9

            for p in self.model.parameters():
                if  p.numel() == nf:  # train (yolo biases)
                    p.requires_grad = True
                elif self.opt.transfer and p.shape[0] == nf:  # train (yolo biases+weights)
                    p.requires_grad = True
                else:  # freeze layer
                    p.requires_grad = False

    def _fit_one(self, save=True):

        mloss = torch.zeros(4).to(self.device)
        lt = len(self.dataloader)
        for i, (imgs, targets, paths, _) in enumerate(self.dataloader):
            # batch -------------------------------------------------------------
            imgs = imgs.to(self.device)
            print(f'第{i}个batch，一共有{lt}个batch')
            targets = targets.to(self.device)
            self.status['batch_num'] = i

            # Plot images with bounding boxes

            if i % 10 == 0:
                train_batch_name = os.path.join(self.inter_plot_folder, 'train_batch%g.jpg' % i)
                if not os.path.exists(self.inter_plot_folder):
                    os.makedirs(self.inter_plot_folder)
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=train_batch_name)

            # Run model
            pred = self.model(imgs)
            # Compute loss
            loss, loss_items = compute_loss(pred, targets, self.model)

            # TODO :验证是否有问题
            self.status['batch_loss'] = loss.item()

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                raise Exception('WARNING: non-finite loss, ending training')

            # Scale loss by nominal batch_size of 64

            # loss *= batch_size / 64

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if i % self.accumulate == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

            for idx in range(4):
                self.ss[idx].append(mloss[idx])

            if i%10==0:
                realTimePlotResults(self.ss, self.tvimg_plot_folder)

            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)

            self.status['mem'] = mem
            self.status['targets_num'] = len(targets)
            self.status['img_size'] = self.img_size
            self.status['GIoU'] = mloss[0]
            self.status['Objectness'] = mloss[1]
            self.status['Classification'] = mloss[2]
            self.status['Train loss'] = mloss[3]

        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                self.chkpt = {'epoch': 0,
                         'best_fitness': 0,
                         'training_results': f.read(),
                         'model': self.model.module.state_dict() if type(
                             self.model) is nn.parallel.DistributedDataParallel else self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict()}

            # Save last checkpoint
            torch.save(self.chkpt, self.last)

        return mloss

    def _initLr(self,typeOfLr):
        # Scheduler https://github.com/ultralytics/yolov3/issues/238
        self.typeOfLr = ['LambdaLR','MultiStepLR']
        assert typeOfLr in self.typeOfLr , 'Input Learing rater is out of field'

        if typeOfLr=='MultiStepLR':
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                      milestones=[round(self.opt.epochs * x) for x in [0.8, 0.9]],
                                                      gamma=0.1)

        elif typeOfLr=='LambdaLR':

            lf = lambda x: 1 - x / self.epochs  # linear ramp to zero
            lf = lambda x: 10 ** (self.hyp['lrf'] * x / self.epochs)  # exp ramp
            lf = lambda x: 1 - 10 ** (self.hyp['lrf'] * (1 - x / self.epochs))  # inverse exp ramp
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        if self.opt.transfer:
            self.scheduler.last_epoch = self.start_epoch - 1
        else:
            self.scheduler.last_epoch = 0

    def fit_n(self,start_epoch,epochs):
        self.clear_result_list()
        self.clear_test_reult_list()
        for epoch in tqdm(range(start_epoch, epochs), desc='train in epochs'):
            self.epoch = epoch
            self.status['epoch'] = epoch
            self.model.train()
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

            self.freeze()

            self.status['NewSavedModel'] = ''

            self.status['Intermediate results chpt'] = []
            self.mloss = self._fit_one()

            # realTimePlotResults(self.ss, self.tvimg_plot_folder)
            self.scheduler.step()
            results = self.test(epoch)

            self.save_(results)


    def save_(self,results):
        fitness = results[2]  # mAP
        if fitness > self.best_fitness:
            self.best_fitness = fitness

        if self.best_fitness == fitness:
            self.saved_bestn = self.saved_bestn + 1
            self.status['NewSavedBestModel'] = self.saved_bestn
            torch.save(self.chkpt, self.best)

        # Save backup every 10 epochs (optional)

        if self.epoch > 0 and self.epoch % 10 == 0:
            torch.save(self.chkpt, os.path.join(self.wdir, 'backup{}_trianloss{}_testmAP{}.pt'.format(self.epoch, self.mloss[3],
                                                                                            self.status['mAP'])))
            self.status['Intermediate results chpt'].append('backup%g.pt' % self.epoch)
        # Delete checkpoint
        del self.chkpt


    def test(self,epoch=None):
        if epoch==None:
            final_epoch = True
        else :
            final_epoch = epoch + 1 == self.epochs

        # Calculate mAP (always test final epoch, skip first 10 if opt.nosave)
        # TODO: 测试标准的iou_Map
        if not final_epoch:
            with torch.no_grad():
                test = Tester(opter, self.model, 'val', 'ioumAP')
                results, maps = test.test()
                # results, maps = test4.test(self.cfg,
                #                            self.data_dict,
                #                            batch_size=self.batch_size,
                #                            img_size=self.opt.img_size,
                #                            model=self.model,
                #                            conf_thres=0.5,  # 0.1 for speed
                #                            save_json=False,
                #                            valOrTest='test')
        # ss = mem + lbox, lobj, lcls, loss + targets + img_size + mp, mr, map, mf1, 3*loss

        for idx in range(7):
            self.testss[idx].append(results[idx])
        realTimePlotResults(self.testss, self.tvimg_plot_folder)
        self.status['Precision'] = self.testss[0]
        self.status['Recall'] = self.testss[1]
        self.status['mAP'] = self.testss[2]
        self.status['F1'] = self.testss[3]
        self.status['GIou'] = self.testss[4]
        self.status['valObjectness'] = self.testss[5]
        self.status['valClassification'] = self.testss[6]
        return results

    def freeze(self, freeze_epoch=3, freeze_backbone=True):
        '''
        :param epoch:
        :param epochs:
        :return:
        '''
        # freeze_backbone = True
        if freeze_backbone and self.epoch < freeze_epoch:
            for name, p in self.model.named_parameters():
                if int(name.split('.')[1]) < self.cutoff:  # if layer < 75
                    p.requires_grad = False

    def fitOne(self):
        self.clear_result_list()
        self._fit_one()

    def clear_result_list(self):
        self.ss = [[],[],[],[]]

    def clear_test_reult_list(self):
        self.testss = [[],[],[],[],[],[],[]]

    def _distributeTrain(self):
        if torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl',  # 'distributed backend'
                                    init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                    world_size=1,  # number of nodes for distributed training
                                    rank=0)  # distributed training node rank
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model.yolo_layers = self.model.module.yolo_layers

    def _datasetInit(self):
        self.dataset = LoadImagesAndLabels(self.train_path,
                                      self.img_size,
                                      self.batch_size,
                                      augment=False,
                                      hyp=self.hyp,  # augmentation hyperparameters
                                      rect=False,  # rectangular training
                                      image_weights=None,
                                      cache_labels=True if self.epochs > 10 else False,
                                      cache_images= self.opt.cache_images, labels=self.lb_lists)

        # Dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=self.batch_size,
                                                 num_workers=1,
                                                 shuffle=True,
                                                 # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=self.dataset.collate_fn)

        self.model.nc = self.nc  # attach number of classes to model
        self.model.arc = self.opt.arc  # attach yolo architecture
        self.model.hyp = self.hyp  # attach hyperparameters to model
        # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
        model_info(self.model, report='summary')  # 'full' or 'summary'
        self.nb = len(self.dataloader)