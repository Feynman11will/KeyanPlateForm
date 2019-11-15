
class opter():
    def __init__(self, var=0, epochs=10, batch_size=8, accumulate=2, cfg='cfg/yolov3-xray.cfg',
                 data='./data/pneumonia.data', multi_scale=False, img_size=608,
                 rect=False, resume=False, transfer=False, notest=False,
                 cache_images=False,
                 arc='defaultpw', device='0', adam=True, tv_img_path='./results'):
        self.var = var
        self.epochs = epochs
        self.batch_size = batch_size
        self.accumulate = accumulate
        self.cfg = cfg
        self.data = data
        self.multi_scale = multi_scale
        self.img_size = img_size
        self.rect = rect
        self.resume = resume
        self.transfer = transfer
        self.notest = notest
        self.cache_images = cache_images
        self.arc = arc
        self.device = device
        self.adam = adam