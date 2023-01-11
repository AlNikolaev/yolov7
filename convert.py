import torch
from copy import deepcopy

if __name__ == '__main__':
    import sys
    sys.path.append('yolov7')

    device = torch.device('cpu')
    original_model = torch.load('yolov7x.pt', map_location=device)['model']
    from yolov7.models.yolo import Model
    new_model = Model('cfg/deploy/yolov7x.yaml')
    if new_model.yaml != original_model.yaml:
        raise RuntimeError('Different config files')
    new_model.load_state_dict(original_model.state_dict())
    for p1, p2 in zip(new_model.parameters(), original_model.parameters()):
        if not torch.eq(p1, p2).all():
            print('RuntimeError. Different tensors')
    ckpt = {'model': deepcopy(new_model).half()}
    torch.save(ckpt, 'converted_yolov7x.pt')
    print('Finish to convert')
