import os 
import sys 
# sys.path.append('../')
# sys.path.append('./')
from .YOLOv3trainer import ODtrianer
from .tester import Tester as tester
from ..config import opter as opt
from dw_sql_operator import SqlOperator 
import json
from ..utils.cfgGenerate import outcfg

from ..kmeans import kmeans_out

class YOLOv3():
    def __init__(self,trainer,tester,taskId,opter):
        # 初始化顺序不可翻转
        self.database = {"DataBase": {
        "localhost": "127.0.0.1",
        "host": "127.0.0.1",
        "port": 3306,
        "user": "root",
        "password": "deepwise",
        "database": "deepwise_research_dy"
            }}
        self.sql = SqlOperator(logger=None, conf = self.database)
        self.taskId  = taskId
        self.opter = opter
        self.initOpter()

        self.statusDict = {}
        self.tester  = tester
        self.trainer = trainer(self.statusDict,opt = self.opter,tester = self.tester)
        
    def initOpter(self):
    
        tasks = list(self.sql.get_from_sql(table = 'project', key = 'id', key_value=self.taskId))[0]
        self.parse(tasks)
        
    def train(self,length=1):
        assert length < 100, '训练数量epochs尽量不大于100个'
        if length==1:
            self.trainer.fitOne()
        else :
            self.trainer.fit_n(0,length)
    def test(self):
        pass

    def parse(self, task_confg):
        modelId = task_confg['modelId']
        modelDetail = list(self.sql.get_from_sql(table = 'model_detail', key = 'id', key_value=modelId))[0]
        self.opter.model_dir = modelDetail['modelPath']
        # 超参数部分
        task = json.loads(task_confg['modelConf'])
        # task = task_confg['modelConf']
        self.opter.hyp = task['hyperparameter']

        # 数据集路径
        self.opter.ds_dir = task['trainParameter']['model']['ds_path']
        # 模型数据集输入路径
        task['trainParameter']['model']['train'] = os.path.join(self.opter.ds_dir, 'labellist/train.txt')
        task['trainParameter']['model']['test'] = os.path.join(self.opter.ds_dir, 'labellist/test.txt')
        task['trainParameter']['model']['lb_list'] = os.path.join(self.opter.ds_dir, 'labels')
        task['trainParameter']['model']['names'] = os.path.join(self.opter.ds_dir, 'names.txt')
        self.opter.model = task['trainParameter']['model']

        # 训练配置选项
        self.opter.epochs = task['trainParameter']['n_epochs']
        self.opter.batch_size = task['trainParameter']['batch_size']
        self.opter.acumulate = task['trainParameter']['acumulate']
        self.opter.transfer = task['trainParameter']['transfer']
        self.opter.img_size = task['trainParameter']['img_size']
        self.opter.resume = task['trainParameter']['resume']
        self.opter.cache_images = task['trainParameter']['cache_images']
        self.opter.notest = task['trainParameter']['notest']
        self.opter.arc = task['trainParameter']['loss_func']
        self.opter.optimizer = task['trainParameter']['optimizer']
        self.opter.augment = task['trainParameter']['augment']
        # self.opter.map_type = task['trainParameter']['map_type']
        self.opter.device = task['trainParameter']['device']
        # 数据增强
        self.opter.brightEnhancement = task['brightEnhancement']
        self.opter.spaceEnhancemenet = task['spaceEnhancemenet']
        self.opter.classes = task['classes']
        self.opter.kmeans = task['kmeans']
        '''
        在解析参数的时候就开始运行kemeans对数据集anchor box 进行聚类
        配置模型cfg文件的获取
        '''
        ds_path = self.opter.ds_dir
        print('dataset path ---------------------\n',ds_path)
        shellpath = os.path.join(ds_path, 'create_custom_model.sh')

        if self.opter.kmeans == True:
            parant_path = self.opter.model['ds_path']
            kmeans_out(self.opter)
            kmeans_path = os.path.join(parant_path, 'kmeans.txt')
            print('kmeans_path path ---------------------\n', kmeans_path)
            outcfg(shellpath, nc=self.opter.classes, kmeans_path=None)
        else:
            #TODO:完成kmeans
            outcfg(shellpath, self.opter.classes)

        self.opter.cfg = os.path.join(ds_path, 'yolov3-nc{}.cfg'.format(self.opter.classes))

        # inference 部分
        self.opter.source = task['inference']['source']
        self.opter.output_folder= os.path.join(self.opter.source,'output')
        self.opter.conf_thres = task['inference']['conf_thres']
        self.opter.nms_thres = task['inference']['nms_thres']
        self.opter.iou_thres = task['inference']['iou_thres']
        self.opter.device = task['inference']['device']
        self.opter.view_img = task['inference']['view_img']
        # self.opter.test_result_path = task['inference']['test_result_path']
        self.opter.test_result_path = os.path.join(self.opter.model_dir.split('.')[0],'test_reuslt_path')
        self.opter.test_model_dir = task['inference']['test_model_dir']
        self.opter.map_type = task['inference']['map_type']


if __name__=='__main__':

    Detector = YOLOv3(trainer = ODtrianer,tester = None,taskId = 111111,opter = opt)
    Detector.train(2)