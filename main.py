import torch
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size
from yolov7.utils.torch_utils import TracedModel


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    img_size = 640
    model = attempt_load('converted_yolov7x.pt', map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16
    print('Ok')
