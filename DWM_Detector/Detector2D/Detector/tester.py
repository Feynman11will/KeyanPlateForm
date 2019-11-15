from matplotlib import patches
from torch.utils.data import DataLoader
import sys
sys.path.append('../')

from ..config.opter import opter
from ..utils.datasets import LoadImagesAndLabels

from .models import *


class Tester():
    def __init__(self,opter:opter,model:torch.nn.Module,valOrTest:str, test_type:str='ioumAP'):
        '''
        :param opter:
        :param model: 模型位置
        :param valOrTest: validation 阶段还是test阶段
        :param test_type: 测试类型 ioumAP or mAP
        '''
        self.opter = opter
        self.cfg  = opter.cfg
        self.data = opter.model,
        self.weights = opter.test_model_dir,
        self.batch_size = opter.batch_size
        self.img_size = opter.img_size,

        self.iou_thres = self.opter.iou_thres,
        self.conf_thres = self.opter.conf_thres,
        self.nms_thres =  self.opter.nms_thres ,
        self.model = model,
        self.test_result_path = self.opter.test_result_path,
        self.valOrTest = valOrTest
        self.test_type=test_type
        if self.test_type=='ioumAP':
            self.a_s = np.linspace(0.4, 0.75, 8)
        elif self.test_type == 'mAP':
            self.a_s = np.array([0.5])


    def test(self):

        if self.model is None:
            device = torch_utils.select_device('0')
            # Initialize model use the defalut model darknet
            self.model = Darknet(self.cfg, self.img_size).to(device)

            # Load weights
            if self.weights.endswith('.pt'):  # pytorch format
                self.model.load_state_dict(torch.load(self.weights, map_location=device)['model'])
            else:  # darknet format
                _ = load_darknet_weights(self.model, self.weights)

            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        else:
            self.device = next(self.model.parameters()).device  # get model device

        self.dataSetInit()

        return self.inference()


    def dataSetInit(self):
        self.nc = int(self.data['classes'])  # number of classes
        self.test_path = self.data['valid']  # path to test images
        self.names = load_classes(self.data['names'])  # class names
        self.lb_lists = self.data['lb_list']
        # Dataloader
        self.dataset = LoadImagesAndLabels(self.test_path, self.img_size, self.batch_size, labels=self.lb_lists)

        self.dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                num_workers=1,
                                pin_memory=True,
                                collate_fn=self.dataset.collate_fn)

    def inference(self):
        with torch.no_grad():
            self.loss = torch.zeros(3)
            s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
            for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(self.dataloader, desc=s)):  # 迭代每一张图像
                self.batch_i = batch_i

                targets = targets.to(self.device)
                # print(targets)
                imgs = imgs.to(self.device)
                self.bs, _, self.img_height, self.img_width = imgs.shape  # batch size, channels, height, width

                if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
                    plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

                inf_out, train_out = self.model(imgs)

                if hasattr(self.model, 'hyp'):  # if model has loss hyperparameters
                    self.loss += compute_loss(train_out, targets, self.model)[1][:3].cpu()

                self.seen = 0
                self.stats = [[], [], [], [], [], [], [], []]
                self.aps = []
                output = non_max_suppression(inf_out, conf_thres=self.conf_thres, nms_thres=self.nms_thres)

                self.statics( output, imgs, targets)

            return  self.comput_statics()

    def comput_statics(self):
        l = []
        for i in range(len(self.a_s)):
            l.append(list(zip(*self.stats[i])))

        # 先将每一个列变成一个列表，然后使用numpy变换成一个列向量，最终输出的是四个列向量
        stats = []

        for i in range(len(self.a_s)):
            stats.append([np.concatenate(x, 0) for x in l[i]])  # to numpy
        # correct中为1的预测值 以及tcls是一一对应的
        for i in range(len(self.a_s)):
            if len(stats[i]):
                if len(self.a_s)!=1:
                    p, r, ap, f1, ap_class = ap_per_class1(*stats[i])
                elif len(self.a_s==1):
                    p, r, ap, f1, ap_class = ap_per_class(*stats[i])
                print(ap)
                if self.a_s[i] == 0.5:
                    mp, mr, mf1, ap_class, ap = p.mean(), r.mean(), f1.mean(), ap_class, ap
                map = ap.mean()
                self.aps.append(ap[0])
                nt = np.bincount(stats[i][3].astype(np.int64), minlength=self.nc)  # number of targets per class
            else:
                nt = torch.zeros(1)

        # Return results
        maps = np.zeros(self.nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

        aps = np.array(self.aps)
        # print(aps)
        map = aps.mean()
        print(map)

        return (mp, mr, map, mf1, *(self.loss / len(self.dataloader)).tolist()), maps

    def statics(self,output,imgs,targets):

        if self.batch_i%10==0:#每10个batch输出validation结果
            ns = np.ceil(self.bs ** 0.5).astype(int)
            fig, _axs = plt.subplots(nrows=2, ncols=4,figsize=(10, 10))
            axs = _axs.flatten()
            for idx, boxes in enumerate(output):
                # 输出的结构是xyxy objconf clsconf cls_pred
                axs[idx].imshow((imgs[idx]).permute(1, 2, 0).cpu())
                axs[idx].set_title('the {}th img'.format(idx))
                for bbox in boxes:
                    x1 = bbox[0]
                    y1 = bbox[1]
                    w = (bbox[0]+bbox[2])/2
                    h = (bbox[1] + bbox[3]) / 2
                    bboxp = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor='green', facecolor="none")
                # Add the bbox to the plot
                    axs[idx].add_patch(bboxp)
            fig.tight_layout()
            # plt.show()
            if not os.path.exists(self.test_result_path):
                os.makedirs(self.test_result_path)

            if self.valOrTest=="test":
                self.fname = os.path.join(self.test_result_path,'testPredImg.jpg')
            else:
                self.fname = os.path.join(self.test_result_path, 'valPredImg.jpg')

            fig.savefig(self.fname, dpi=200)
            # display.clear_output(wait=True)

        for idx,iou_threshold in enumerate(self.a_s):  #ioumAP
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                self.seen += 1

                # 如果没有目标就在统计列表中：加入一个空的列表和空的张量，并将分类lables加入到其中，
                if pred is None:
                    if nl:
                        self.stats[idx].append(([], torch.Tensor(), torch.Tensor(), tcls))
                    continue

                clip_coords(pred, (self.img_height, self.img_width))

                # Assign all predictions as incorrect
                correct = [0] * len(pred)
                if nl:
                    detected = []
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    tbox[:, [0, 2]] *= self.img_width
                    tbox[:, [1, 3]] *= self.img_height

                    for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                        # Break if all targets already located in image
                        if len(detected) == nl:
                            break

                        # Continue if predicted class not among image classes
                        if pcls.item() not in tcls:
                            continue

                        m = (pcls == tcls_tensor).nonzero().view(-1)

                        iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                        if iou > iou_threshold and m[bi] not in detected:  # and pcls == tcls[bi]:
                            correct[i] = 1
                            detected.append(m[bi])

                # Append statistics (correct, conf, pcls, tcls)
                self.stats[idx].append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))
