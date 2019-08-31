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
    device_name = 'cuda:0' if use_cuda else 'cpu'
    device = torch.device(device_name)

    params = {} if use_cuda else {'map_location': 'cpu'}

    # 恢复检查点
    if ckpt_path is not None:
        assert os.path.exists(ckpt_path), '无效的路径{}'.format(ckpt_path)
        ckpt = torch.load(ckpt_path, **params)
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
        models.append(load_model_from_ckpt(ckpt_full_path))
    return models

def load_model_from_ckpt(ckpt_path:str)->tuple:
    '''从指定文件中恢复模型'''
    ckpt_filename = os.path.basename(ckpt_path)
    model_target = get_target_name_from(ckpt_filename)
    model, device = setup(ckpt_path)
    return (model_target, model, device)

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
    '''处理整段视频

    Args:
        video_path(str): 视频路径
        ckpt_root_path(str): 检查点路径
        step(int): 视频处理的步长
        n(int): 每次用于输入网络的视频长度
        threshold(float): 界定动作发生的概率阈值

    Return:
        restults(list): 逐帧的各个动作发生的概率
        timeline(list): 每个动作发生的时间段
        targets(list): 每个任务的名称
    '''
    for path in [video_path, ckpt_root_path]:
        assert path is not None and os.path.exists(path), '路径不存在：{}'.format(path)

    model_ckpts = os.listdir(ckpt_root_path)
    w, h, rate, total = get_video_info(video_path)
    print('Video info:\n\tsize: {} * {}\n\trate: {}\n\ttotal: {}'.format(w, h, rate, total))

    results = np.zeros((total, len(model_ckpts), 2)) # shape = (总帧数 * 总类数 * 2)

    # 依次加载各个模型
    targets = []
    for i, model_file_name in enumerate(model_ckpts):
        model_full_name = os.path.join(ckpt_root_path, model_file_name)
        target_name, model, device = load_model_from_ckpt(model_full_name)
        targets.append(target_name)
        # 开始处理单个视频
        for clip, frame_index in tqdm(extract_imgs_from_video(video_path, step, n),
                total=((total - n) // step + 2), desc='Inferencing {}'.format(target_name)):
            clip = transform_to_tensor(clip)
            clip = torch.autograd.Variable(clip, requires_grad=False).to(device)
            with torch.no_grad():
                output = model.forward(clip)
            output = torch.nn.Softmax(dim=1)(output).cpu().numpy()
            results[frame_index - n:frame_index, i] = output
    
    # 保存结果
    flags = [False] * len(targets) # 用来存储动作出现与否
    count = [0] * len(targets) # 用来存储每个动作出现的次数
    timeline = [[] for i in targets]
    # 用来存储动作出现的时间段，同一个动作在同一视频中有可能出现多次，shape = (target_count, times, 2)

    trans_zh = {
        'acrossLadder': '跨节',
        'firststep': '首步掉落',
        'objectfall': '物体掉落',
        'sameLadder': '同节',
        'sendHose': '水带压线'
    }

    # 获取输出文件名称
    file_base_name = os.path.basename(video_path).rsplit('.')[0]
    output_file_name = file_base_name + '_output.csv'
    with open(output_file_name, 'w') as f:
        f.write(','.join([trans_zh[name] for name in targets]))
        f.write('\n')
        for frame_index, frame in enumerate(results):
            # frame是shape = (target_count, 1)的数组，存储了每一帧对应的n个动作发生的概率
            for target_index, p in enumerate(frame):
                f.write('{:.4f}, '.format(p[1]))
                if p[1] > threshold and not flags[i]:
                    count[target_index] += 1 # 上升沿记录该动作发生一次，并记录动作开始的帧索引
                    timeline[target_index].append({
                        'start_frame': frame_index,
                        'end_frame': -1
                    })
                if p[1] <= threshold and flags[target_index] and len(timeline[target_index]) > 0:
                    # 在下降沿记录动作的结束帧索引
                    if timeline[target_index][-1]['end_frame'] < 0:
                        timeline[target_index][-1]['end_frame'] = frame_index
                flags[target_index] = p[1] > threshold # 更新存储状态
            f.write('\n')

    should_show_time = False # 不输出帧数时间段

    summary_file_name = file_base_name + '_summary.csv'
    with open(summary_file_name, 'w') as f:
        f.write('次数统计\n类目,次数,惩时(s),{}\n'.format('出现时间段' if should_show_time else ''))
        for i, name in enumerate(targets):
            time_penalty = '-' if name is 'objectfall' else str(count[i] * 5)
            time_line_str = ', '.join(['({} ~ {})'.format(seg['start_frame'], seg['end_frame']) for seg in timeline[i]])
            f.write('{}, {}, {}, {}\n'.format(trans_zh[name], count[i], time_penalty, time_line_str if should_show_time else ''))
            print('{}:\n\t出现{}次'.format(trans_zh[name], count[i]))
            if count[i] > 0 and should_show_time:
                print('\t出现帧数段: ' + time_line_str)
            if name != 'objectfall' and count[i] > 0:
                print('\t罚时: {}s'.format(time_penalty))

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
                cv2.putText(frame, '{}: {}'.format(trans_zh[name], p), (20, 40 * i + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        video_wp.write(frame)
    video_wp.release()

    print('results have been saved to: \n\t{}\n\t{}\n\t{}'.format(output_file_name, output_video_name, summary_file_name))
    return results.tolist(), timeline, targets

def run_as_service():
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route('/detect', methods=['POST', 'GET'])
    def __handle_post():
        print(request.form)
        video_path = request.form['video']
        ckpt_path = request.form['ckpt']
        threshold = float(request.form['threshold'])
        step = int(request.form['step'])
        length = int(request.form['length'])
        results, timeline, targets = eval_on_video(video_path, ckpt_path, step, length, threshold)
        ret = {
            'raw_data': results,
            'timeline': {key: timeline[i] for i, key in enumerate(targets)}
        }
        return jsonify(ret)
    
    app.run(debug=True, host='127.0.0.1', port=5000)

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 eval.py -i path/to/video -r path/to/checkpoints')
    parser.add_argument('-i', '--video', help='path to video')
    parser.add_argument('-r', '--ckpt_root_path', help='path to the checkpoints')
    parser.add_argument('-t', '--threshold', default=0.8, type=float)
    parser.add_argument('-web', default=0, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    step = 15
    length = 30
    if args.web > 0:
        run_as_service()
    else:
        eval_on_video(args.video, args.ckpt_root_path, 15, 30, args.threshold)