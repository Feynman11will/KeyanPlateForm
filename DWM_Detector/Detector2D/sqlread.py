
import sys 
sys.path.append('../')
from dw_sql_operator import SqlOperator 
database = {"DataBase": {
        "localhost": "127.0.0.1",
        "host": "127.0.0.1",
        "port": 3306,
        "user": "root",
        "password": "deepwise",
        "database": "deepwise_research_dy"
            }}


sql = SqlOperator(logger=None, conf = database)

tasks = list(sql.get_from_sql(table = 'project', key = 'id', key_value=111111))


for task in tasks[0].items():

    print(task)

    if task[0] == 'modelId':
        val = task[1]
        model = sql.get_from_sql(table = 'model_detail', key = 'id', key_value=val)
        print('model details :',model)
'''
(u'comment', None)
(u'updateTime', None)
(u'isAuto', 0)
(u'process', 0)
(u'trainCount', 0)
(u'idName', u'')
(u'modelType', 5)
(u'auxiliaryInfo', None)
(u'trainNum', u'0')
(u'imgResults', None)
(u'modelConf', u'{"hyperparameter":{ "giou" : 1.582, "cls": 27.76,  "cls_pw": 1.446, "obj": 21.35,  "obj_pw": 3.941,"iou_t": 0.2635, "lr0": 0.00001,"lrf": -4.0, "momentum": 0.98,"weight_decay": 0.000004569,  "fl_gamma": 0.5,"hsv_s": 0.5703,"hsv_v": 0.3174, "degrees": 1.113,  "translate": 0.06797,  "scale": 0.1059, "shear": 0.5768},"trainParameter":{"model":{"ds_path":"/data1/wanglonglong/FeiYan"},"n_epochs":1,"batch_size":8,"acumulate":2,"transfer":0,"img_size":640,"resume":1,"cache_images":1,"notest":0,"loss_func":"defaultpw","optimizer":"adam","augment":1,"device":""},"brightEnhancement":1,"spaceEnhancemenet":1,"classes":1,"inference":{"source":"/data1/wanglonglong/01workspace/yolov3_orig/yolov3-xray-chest/data/samples","conf_thres": 0.5,"nms_thres":0.5,"iou_thres":0.5,"device":"0","view_img":1,"test_result_path":"/data1/wanglonglong/01workspace/yolov3PtResult/FeiyanOk/backup30.pt","map_type":"iouMap","test_model_dir":"/data1/wanglonglong/01workspace/yolov3PtResult/FeiyanOk/backup30.pt"},"kmeans":1}')
(u'id', 111111)
(u'featureConf', None)
(u'category', 1)
(u'auc', None)
(u'modelName', u'Object2D')
(u'advancedSetting', None)
(u'parentId', 0)
(u'progress', None)
(u'analysisId', None)
(u'type', 0)
(u'resultsIndetailPath', None)
(u'modelId', 11111)
(u'status', 10)
(u'topicId', 120)
(u'trainId', None)
(u'logPath', None)
(u'createBy', 0)
(u'testId', None)
(u'startTime', None)
(u'analysisType', 1)
(u'runTime', None)
(u'desc', None)
(u'statusDesc', None)
(u'name', u'')
(u'creationTime', datetime.datetime(2019, 11, 11, 17, 41, 1))
(u'trainType', 0)
(u'usedFeatures', None)
(u'endTime', None)
(u'remainTime', 0)
(u'result', None)
'''