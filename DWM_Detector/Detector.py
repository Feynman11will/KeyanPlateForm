import streamlit as st
import pandas as pd
import numpy as np
import time
from Detector2D.Detector.YOLOv3 import YOLOv3
from Detector2D.Detector.YOLOv3trainer import ODtrianer
from Detector2D.config.opter import opter

import multiprocessing as mp
import signal
# import markdown
import sys

#1. 模型路径在创建任务时已经被选择好
st.markdown('wocao wuqing')
st.write('''<head>
  <title>Hello, Streamlit!</title>
</head>''')
# print('wocao wuqing')
st.sidebar.title('2D目标检测器配置')
st.sidebar.markdown('----')
st.sidebar.markdown('1. 请选择任务类型')
options = st.sidebar.radio('1. 算法类型',('YOLOv3','其他'))
st.sidebar.markdown('----')

wonder_dict = {"hyperparameter":
     { "giou" : 1.582,
       "cls": 27.76,
       "cls_pw": 1.446,
       "obj": 21.35,
       "obj_pw": 3.941,
       "iou_t": 0.2635,
       "lr0": 0.00001,
       "lrf": -4.0,
       "momentum": 0.98,
       "weight_decay": 0.000004569,
       "fl_gamma": 0.5,
       "hsv_s": 0.5703,
       "hsv_v": 0.3174,
       "degrees": 1.113,
       "translate": 0.06797,
       "scale": 0.1059,
       "shear": 0.5768},
    "trainParameter":
        {"model":{"ds_path":"/data1/wanglonglong/FeiYan"},
         "n_epochs":10,
         "batch_size":8,
         "acumulate":2,
         "transfer":0,
         "img_size":640,
         "resume":1,
         "cache_images":1,
         "notest":0,
         "loss_func":"defaultpw",
         "optimizer":"adam",
         "augment":1,
         "device":""},
     "brightEnhancement":1,
     "spaceEnhancemenet":1,
     "classes":1,
     "inference":
         {"source":"/data1/wanglonglong/01workspace/yolov3_orig/yolov3-xray-chest/data/samples",
          "conf_thres": 0.5,
          "nms_thres":0.5,
          "iou_thres":0.5,
          "device":"0",
          "view_img":1,
          "test_result_path":"/data1/wanglonglong/01workspace/yolov3PtResult/FeiyanOk/backup30.pt"},
      "kmeans":1}

# wonder_dict.copy(deep=True)
def initYOLOv3():
    Detector = YOLOv3(trainer = ODtrianer,tester = None,taskId = 111111,opter = opter)
    return Detector


def yolov3options():
    epochs = st.sidebar.slider('epochs',0,100)
    return epochs


# @st.cache
def configModel():
    # import wx
    # wx.Frame.__init__(self,None,-1,title="wxApp.",size=(250,250),pos=(0,0))
    # mm=wx.DisplaySize()
    # st.write(mm)
    # weights_warning = None 
    # weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)")
    # weights_warning
    configdict={}
    st.sidebar.markdown('模型配置')
    st.sidebar.markdown('----')
    st.sidebar.markdown('训练参数')

    st.sidebar.info(
            """对模型的训练参数进行配置"""
        )
    
    st.markdown("<br>这是分隔符<br>", unsafe_allow_html=True)
    st.markdown("<font size='10'>卧槽，无情</font>", unsafe_allow_html=True)

    min_value = 2018
    max_value = 2020
    year_range = st.slider(
            "Select min and max Year",
            min_value=min_value,
            max_value=max_value,
            value=[min_value, max_value],
        )
    st.write(year_range)


    number = st.sidebar.number_input('输入数字',min_value = 0.00001,max_value = 10.00001, step=0.00001)
    number
    configdict['kmeans'] = st.sidebar.checkbox("kmeans")
    configdict['test']  = st.sidebar.checkbox("test")
    configdict['transfer'] = st.sidebar.checkbox("transfer")
    configdict['cache_images']=st.sidebar.checkbox("cache_images")
    optimizer = configdict['optimizer'] = st.sidebar.selectbox('优化器',('adam','sgd'))
    configdict['loss_func'] = st.sidebar.selectbox('损失函数',('defaultdw','focalloss',''))

    configdict['classes']  = st.sidebar.text_input('分类数量')

    configdict['epochs'] = st.sidebar.slider('epochs',1,100)
    configdict['batchsize'] = st.sidebar.slider('batchsize',1,16)
    configdict['accumulate'] = st.sidebar.slider('accumulate',1,4)
    configdict['img_size'] =  st.sidebar.slider("img_size",384,768,step=32)
    st.sidebar.markdown('----')
    st.sidebar.markdown('训练超参数')

    configdict['giou']  = st.sidebar.slider('giou',min_value=0.2,max_value=100.0,step=0.00001,value=1.532)
    configdict['clss']   = st.sidebar.slider('classes',min_value=0.2,max_value=100.0,step=0.00001,value=27.76)
    configdict['clspw']  = st.sidebar.slider('clspw',min_value=0.2,max_value=100.000,step=0.00001,value=1.446)
    configdict['iou_t']  = st.sidebar.slider('iou_t',min_value=0.2,max_value=100.000,step=0.00001,value=1.446)

    configdict['lr0'] = st.sidebar.text_input('学习率:范围从1e-10到1','1e-5')
 
    if optimizer=='sgd':
        configdict['lrf'] = st.sidebar.slider('lrf',min_value=-10.000,max_value=10.000,step=0.01,value=-4.0)
    configdict['momentum'] = st.sidebar.slider('momentum 动量',min_value=0.5,max_value=1.000,step=0.01,value=0.98)
    st.sidebar.markdown('----')
    st.sidebar.markdown('图像增强')
    brightEnhancement = configdict['brightEnhancement']  = st.sidebar.checkbox("brightEnhancement")
    if brightEnhancement==1:
        configdict['hsv_s']  = st.sidebar.slider('hsv_s',min_value=0.0,max_value=1.000,step=0.01,value=0.57)
        configdict['hsv_v'] = st.sidebar.slider('hsv_v',min_value=0.0,max_value=1.000,step=0.01,value=0.31)

    spaceEnhancemenet = configdict['spaceEnhancemenet']  = st.sidebar.checkbox("spaceEnhancemenet")
    if spaceEnhancemenet==1:
        configdict['translate'] = st.sidebar.slider('translate',min_value=0.0,max_value=1.000,step=0.01,value=0.57)
        configdict['degrees'] = st.sidebar.slider('degrees',min_value=0.0,max_value=360.,step=0.01,value=1.12)
        configdict['scale'] = st.sidebar.slider('scale',min_value=0.0,max_value=360.,step=0.01,value=1.12)
        configdict['shear'] = st.sidebar.slider('shear',min_value=0.0,max_value=360.,step=0.01,value=1.12)
        configdict['fl_gamma'] =st.sidebar.slider('fl_gamma',min_value=0.0,max_value=1.000,step=0.01,value=0.5)

    st.sidebar.markdown('----')
    st.sidebar.markdown('inference')

    configdict['conf_thres']  =st.sidebar.slider('conf_thres',min_value=0.0,max_value=1.000,step=0.01,value=0.5)
    configdict['nms_thres']  =st.sidebar.slider('nms_thres',min_value=0.0,max_value=1.000,step=0.01,value=0.5)
    configdict['view_img']  =st.sidebar.checkbox('view_img')
    
    return configdict


def train(epochs, progress_bar):
    trainend_status = st.empty()
    for i in range(epochs):
        trainend_status.markdown('正在训练')
        progress_bar.progress(i/epochs)
        time.sleep(1)
        print(i)
    st.balloons()
    trainend_status.markdown('训练未进行')

if options=='YOLOv3':

    from newceil import showtime
    configDict = configModel()
    epochs = configDict['epochs']
    st.sidebar.markdown('----')

    st.title('YOLOv3算法模型')
    if st.checkbox('显示配置项'):
        st.write(wonder_dict)
    
    st.subheader('1. 配置参数')
    if st.checkbox('显示配置参数'):
        st.markdown(f'- [x] &#8195;训练次数 : {epochs}')

    st.subheader('2. 运行状态')

    # Detector = initYOLOv3()

    if st.checkbox('显示网格效果'):
        showtime()
    st.sidebar.markdown('- [X] 模型状态')


    begintrain = st.sidebar.button('训练')
    endtrain = st.sidebar.button('停止训练')
    progress_bar = st.progress(0)
    p = mp.Process(target = train,args=(10, progress_bar))



    # from DWM_multipro import GracefulExitEvent,worker_proc,GracefulExitException
    # gee = GracefulExitEvent()
    # signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)
    if begintrain:
        #开启线程
        p.start()
        '''开始训练了'''
    'beginTrainxiaoshi'
    if endtrain:
        # 关闭进程
        raise Exception('结束当前进程')
        # if not p.is_alive():
        #     p.join(1)
        '''停止训练了'''
        pass



elif options=='其他':
    st.sidebar.markdown('其他配置')

