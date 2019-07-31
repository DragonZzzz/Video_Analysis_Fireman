import torch
from torch import nn
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
from collections import deque
from PIL import Image

import argparse
import os

from network import C3D_model

def setup(ckpt_path:str)->nn.Module:
    '''配置模型的GPU/CPU环境
    
    Args:
        ckpt_path(str): 模型检查点路径
    Return:
        model(nn.Module): 配置好的模型
        device(torch.device): 模型可用的环境
    '''
    model = C3D_model.C3D(num_classes=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # 恢复检查点
    if ckpt_path is not None:
        assert os.path.exists(ckpt_path), '无效的路径{}'.format(ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()

    return model, device

def get_target_name_from(src_ckpt_name:str)->str:
    '''从检查点文件名中提取任务信息'''
    ckpt_name = src_ckpt_name.replace('C3D-', '')
    ckpt_name = ckpt_name.split('_')
    assert len(ckpt_name) == 2, '无法从{}中恢复任务名，请按照C3D-[任务名]_epoch-[n].pth.tar命名'.format(src_ckpt_name)
    return ckpt_name[0]

def load_model_from_ckpts(ckpt_root_path:str)->list:
    '''从包含多个检查点的目录中恢复模型'''
    ckpt_filenames = os.listdir(ckpt_root_path)
    models = []
    for ckpt_name in tqdm(ckpt_filenames, desc='Loading'):
        ckpt_full_path = os.path.join(ckpt_root_path, ckpt_name)
        model_target_name = get_target_name_from(ckpt_name)
        model, device = setup(ckpt_full_path)
        models.append((model_target_name, model, device))
    return models

def extract_imgs_from_video(video_path:str, step:int = 1, n:int = 30)->deque:
    '''从视频中提取图像
    
    Args:
        video_path(str): 视频路径
        step(int): 视频中提取图像的步长
        n(int): 每次提取图像的帧数
    '''
    video_fp, has_frame = cv2.VideoCapture(video_path), True
    queue = deque([], maxlen=n) # 使用一个定长队列来存储
    index = 0
    # 首次直接读取n帧
    while len(queue) < n and has_frame:
        has_frame, frame = video_fp.read()
        queue.append(frame)
        index += 1
    # 之后每次读取step帧
    count = 0
    while has_frame:
        if count == 0:
            yield queue, index
        has_frame, frame = video_fp.read()
        if frame is not None:
            queue.append(frame)
            count = (count + 1) % step
            index += 1
    yield queue, index


def get_video_info(video_path:str)->(int, int, int, int):
    '''获取视频宽/高/帧速率/帧总数信息'''
    video_fp = cv2.VideoCapture(video_path)
    return [int(video_fp.get(i)) for i in [3, 4, 5, 7]]


def transform(x)->torch.Tensor:
    x = Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    mean = np.array([90.0, 98.0, 102.0]) / 255
    x = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean, [1.0, 1.0, 1.0]),
        transforms.ToPILImage()
    ])(x)
    return torch.Tensor(np.array(x))


def transform_to_tensor(clip:list)->torch.Tensor:
    '''视频数据转换为tensor'''
    # 视频数转换帧后增加一维，使其变为batchsize * n * h * w * c
    clip = torch.stack(list(map(transform, clip))).unsqueeze(0) # type: torch.Tensor
    clip = clip.permute(0, 4, 1, 2, 3) # 修改成channel first的格式
    return clip

def eval_on_video(video_path:str, ckpt_root_path:str, step:int, n:int, threshold: float):
    for path in [video_path, ckpt_root_path]:
        assert path is not None and os.path.exists(path), '路径不存在：{}'.format(path)
    print('Loading models from ckpts.')
    models = load_model_from_ckpts(ckpt_root_path)
    print('{} models are loaded: {}'.format(len(models), [x[0] for x in models]))
    w, h, rate, total = get_video_info(video_path)
    print('Video info:\n\tsize: {} * {}\n\trate: {}\n\ttotal: {}'.format(w, h, rate, total))

    results = np.zeros((total, len(models), 2)) # shape = (总帧数 * 总类数 * 2)

    for clip, frame_index in tqdm(extract_imgs_from_video(video_path, step, n),
            total=((total - n) // step + 2), desc='Inferencing'):
        clip = transform_to_tensor(clip)
        for i, (_, model, device) in enumerate(models):
            clip = torch.autograd.Variable(clip, requires_grad=False).to(device)
            with torch.no_grad():
                output = model.forward(clip)
            output = torch.nn.Softmax(dim=1)(output).cpu().numpy()
            results[frame_index - n:frame_index, i] = output
    
    # 保存结果
    targets = [m[0] for m in models]
    flags = [False] * len(targets) # 用来存储动作出现与否
    count = [0] * len(targets)
    output_file_name = os.path.basename(video_path).rsplit('.')[0] + '_output.csv'
    with open(output_file_name, 'w') as f:
        f.write(','.join(targets))
        f.write('\n')
        for frame in results:
            for i, p in enumerate(frame):
                f.write('{:.4f}, '.format(p[1]))
                if p[1] > threshold and not flags[i]:
                    count[i] += 1 # 只在上升沿记录该动作
                flags[i] = p[1] > threshold # 更新存储状态
            f.write('\n')
        f.write('\n\n次数统计\n')
        for i, name in enumerate(targets):
            f.write('{}, {}\n'.format(name, count[i]))

    # 结果写入视频
    output_video_name = os.path.basename(video_path).rsplit('.')[0] + '_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_wp = cv2.VideoWriter(output_video_name, fourcc, rate, (w, h))
    for frame, index in extract_imgs_from_video(video_path, 1, 1):
        frame = frame[0]
        probs = results[index - 1, :, 1]
        probs_sorted = sorted(zip(probs, targets), reverse=True)
        for i, (p, name) in enumerate(probs_sorted):
            if p > threshold:
                cv2.putText(frame, '{}: {}'.format(name, p), (20, 40 * i + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        video_wp.write(frame)
    video_wp.release()

    print('results have been saved to: \n\t{}\n\t{}'.format(output_file_name, output_video_name))
    return results

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 eval.py -i path/to/video -r path/to/checkpoints')
    parser.add_argument('-i', '--video', help='path to video')
    parser.add_argument('-r', '--ckpt_root_path', help='path to the checkpoints')
    parser.add_argument('-t', '--threshold', default=0.8, type=float)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    eval_on_video(args.video, args.ckpt_root_path, 15, 30, args.threshold)