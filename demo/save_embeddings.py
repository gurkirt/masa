import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import gc
import resource
import argparse
import cv2
import tqdm
import numpy as np
import pdb
import math 
import json

import pickle


import torch
import torchvision
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm

import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmcv.ops.nms import batched_nms
from mmdet.datasets import CocoDataset
from demo.video_frame_loader import VideoFrameReader

import masa
from masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline
from masa.models.sam import SamPredictor, sam_model_registry
from utils import filter_and_update_tracks
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Ensure the right start method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

def set_file_descriptor_limit(limit):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))

# Set the file descriptor limit to 65536
set_file_descriptor_limit(65536)

def visualize_frame(args, visualizer, frame, track_result, frame_idx, fps=None):
    visualizer.add_datasample(
        name='video_' + str(frame_idx),
        image=frame[:, :, ::-1],
        data_sample=track_result[0],
        draw_gt=False,
        show=False,
        out_file=None,
        pred_score_thr=args.score_thr,
        fps=fps,)
    frame = visualizer.get_image()
    gc.collect()
    return frame


class ImageList(object):
    def __init__(self, base_dir) -> None:
        self.images_dir = f'{base_dir}/images/'
        self.labels_dir = f'{base_dir}/labels/'

        self.image_list = []
        skip_count = 0
        gskip = 0
        dskip = 0
        print('Start loading {}')
        lenth = 10
        distances = []
        total_dirs = 0
        for snippet_dir in tqdm(os.listdir(self.images_dir)):
            total_dirs += 1
            label_file = os.path.join(self.labels_dir, snippet_dir+'.json')
            with open(label_file, 'r') as f:
                data = json.load(f)
            
            if data['sensor_geomtery']["gravity_vector"][2] < 0.5:
                # print("Skip because gravity vector", snippet_dir, data['sensor_geomtery']["gravity_vector"])
                gskip += 1
                continue

            if data['valid_frames'] < lenth:
                skip_count += 1
                continue
            
            data_start = data['track'][0]
            data_end = data['track'][-1]
            X, Y, Z = data_start['X_head'], data_start['Y_head'], data_start['Z_head']
            X_, Y_, Z_ = data_end['X_head'], data_end['Y_head'], data_end['Z_head']

            distance = np.sqrt((X-X_)**2 + (Y-Y_)**2 + (Z-Z_)**2)
            if distance < 3.0:
                dskip += 1
                continue

            distances.append(distance)
            

            # snippet_list = []
            imgs = [im for im  in sorted(os.listdir(os.path.join(self.images_dir, snippet_dir))) if im.endswith('.png')]
            video_samples = []
            assert len(imgs) == len(data['track']), "{} {}".format(len(imgs), len(data['track']))
            for i, im in enumerate(imgs):
                track_data = data['track'][i]
                pixel_width, pixel_height = track_data['pixel_width'], track_data['pixel_height']
                x_head, y_head = max(0, min(track_data['x_head'], pixel_width)), max(0, min(track_data['y_head'], pixel_height))
                x_feet, y_feet = max(0, min(track_data['x_feet'], pixel_width)), max(0, min(track_data['y_feet'], pixel_height))
                x_min_head = max(0 , min(pixel_width, x_head, track_data['x_head_top'], track_data['x_head_bottom'], track_data['x_head_right'], track_data['x_head_left']))
                y_min_head = max(0 , min(pixel_height, y_head, track_data['y_head_top'], track_data['y_head_bottom'], track_data['y_head_right'], track_data['y_head_left']))
                x_max_head = min(pixel_width, max(0, x_head, track_data['x_head_top'], track_data['x_head_bottom'], track_data['x_head_right'], track_data['x_head_left']))
                y_max_head = min(pixel_height, max(0, y_head, track_data['y_head_top'], track_data['y_head_bottom'], track_data['y_head_right'], track_data['y_head_left']))
                x_min_feet = max(0, min(pixel_width, x_feet, track_data['x_feet_top'], track_data['x_feet_bottom'], track_data['x_feet_right'], track_data['x_feet_left']))
                y_min_feet = max(0, min(pixel_height, y_feet, track_data['y_feet_top'], track_data['y_feet_bottom'], track_data['y_feet_right'], track_data['y_feet_left']))
                x_max_feet = min(pixel_width, max(0, x_feet, track_data['x_feet_top'], track_data['x_feet_bottom'], track_data['x_feet_right'], track_data['x_feet_left']))
                y_max_feet = min(pixel_height, max(0, y_feet, track_data['y_feet_top'], track_data['y_feet_bottom'], track_data['y_feet_right'], track_data['y_feet_left']))

                box = [min(x_min_head, x_min_feet), min(y_min_head, y_min_feet), max(x_max_head, x_max_feet), max(y_max_head, y_max_feet)]
                boxes = [box, [max(0, x_head-30), max(0, y_head-30),  min(pixel_width, x_head+30), min(pixel_height, y_head+30)]]
                video_samples.append([im, np.asarray(boxes)])

            self.image_list.append([snippet_dir, video_samples])

        plt.hist(distances, 100)
        plt.show()
        plt.savefig('histogram.png')
        plt.close()
        self.count = 0                  
        print(f"skip due len shorter than {lenth} :: ", skip_count)
        print("Skip due to gravity", gskip)
        print("Skip due to distance", dskip)
        print(f"List size is {len(self.image_list)} out of {total_dirs}")
        self.current_image = None
        print("Done Laoding Snippet Info")

    def __next__(self):
        if self.count < len(self.image_list):
            sample = self.image_list[self.count]
            self.count += 1
            return sample
        else:
            raise StopIteration
        # draw.rectangle((box[0], box[1], box[2], box[3]), fill=(0, 0, 0, 0), outline=(0, 0, 255, 255))
    
    def __len__(self):
        return len(self.image_list)
    
    def __iter__(self):
        self.count = 0
        return self
    


def parse_args():

    parser = argparse.ArgumentParser(description='MASA video demo')
    parser.add_argument('video', help='Video file', default=None)
    parser.add_argument('--det_config', help='Detector Config file')
    parser.add_argument('--masa_config', help='Masa Config file')
    parser.add_argument('--det_checkpoint', help='Detector Checkpoint file')
    parser.add_argument('--masa_checkpoint', help='Masa Checkpoint file')
    parser.add_argument( '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file', default=None)
    parser.add_argument('--save_dir', type=str, help='Output for video frames')
    parser.add_argument('--texts', help='text prompt')
    parser.add_argument('--line_width', type=int, default=5, help='Line width')
    parser.add_argument('--unified', action='store_true', help='Use unified model, which means the masa adapter is built upon the detector model.')
    parser.add_argument('--detector_type', type=str, default='mmdet', help='Choose detector type')
    parser.add_argument('--fp16', action='store_true', help='Activation fp16 mode')
    parser.add_argument('--no-post', action='store_true', help='Do not post-process the results ')
    parser.add_argument('--show_fps', action='store_true', help='Visualize the fps')
    parser.add_argument('--sam_mask', action='store_true', help='Use SAM to generate mask for segmentation tracking')
    parser.add_argument('--sam_path',  type=str, default='saved_models/pretrain_weights/sam_vit_h_4b8939.pth', help='Default path for SAM models')
    parser.add_argument('--sam_type', type=str, default='vit_h', help='Default type for SAM models')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    video_list = ImageList(base_dir='/workdisk/image_data/track_data/')
    # build the model from a config file and a checkpoint file
    
    if args.unified:
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
    else:
        if args.detector_type == 'yolov10':
            from ultralytics import YOLO10
            det_model = YOLO10(args.det_checkpoint, verbose=False)

        elif args.detector_type == 'yolov8':
            from ultralytics import YOLO
            det_model = YOLO(args.det_checkpoint)

        elif args.detector_type == 'mmdet':
            det_model = init_detector(args.det_config, args.det_checkpoint, palette='random', device=args.device)
            # build test pipeline
            det_model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
            test_pipeline = Compose(det_model.cfg.test_dataloader.dataset.pipeline)
        else:
            raise Exception('ATM only mmdet or YOLOv8 or YOLOv10 detection models are allowed')
        
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
        
    
    video_writer = None
    #### parsing the text input
    texts = args.texts
    if texts is not None:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg, with_text=True)
    else:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg)

    if texts is not None:
        masa_model.cfg.visualizer['texts'] = texts
    elif args.detector_type == 'mmdet':
        masa_model.cfg.visualizer['texts'] = det_model.dataset_meta['classes']
    elif args.detector_type == 'mmdet':
        masa_model.cfg.visualizer['texts'] = det_model.names

    # init visualizer
    masa_model.cfg.visualizer['save_dir'] = args.save_dir
    masa_model.cfg.visualizer['line_width'] = args.line_width
    if args.sam_mask:
        masa_model.cfg.visualizer['alpha'] = 0.5


    
    
    frames = []
    fps_list = []
    run_type = 'head'
    save_dir = os.path.join(f'/workdisk/image_data/track_data/{run_type}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    num_vidoes = len(video_list)
    for vi, sample in enumerate(tqdm(video_list)):
        snippet_dir = sample[0]        
        frame_idx = 0
        instances_list = []
        video_dir = os.path.join(video_list.images_dir, snippet_dir)
        video_reader = VideoFrameReader(video_dir)
        embeddings = []
        det_outputs = []
        track_outputs = []
        for frame in video_reader:
            
            if run_type == 'yolo_tracks':
                preds = det_model.predict(frame, verbose=False)
                boxes = preds[0].boxes.xyxy
                conf = preds[0].boxes.conf.unsqueeze(1)
                det_bboxes = torch.cat((boxes,conf ),1)
                det_labels = preds[0].boxes.cls
            elif run_type == 'head':
                boxes = torch.from_numpy(sample[1][frame_idx][1])
                # boxes[:, 2] = (boxes[:, 2] - boxes[:, 0])
                # boxes[:, 3] = (boxes[:, 3] - boxes[:, 1])
                # print('det bboxes', boxes)
                
                det_bboxes = torch.ones((boxes.shape[0],5))
                det_bboxes[:,:4] = boxes.float()
                det_labels = det_bboxes[:, 4]

            track_result = inference_masa(masa_model, frame, frame_id=frame_idx,
                                        video_len=len(video_reader),
                                        test_pipeline=masa_test_pipeline,
                                        det_bboxes=det_bboxes,
                                        det_labels=det_labels,
                                        fp16=args.fp16,
                                        show_fps=args.show_fps)
            

            if args.show_fps:
                track_result, fps = track_result
            
            track_result, box_feats = track_result
            if box_feats is None:
                embeddings.append(None)
                det_outputs.append(None)
                track_outputs.append(None)
                continue
            embeddings.append(box_feats.to('cpu'))
            track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(torch.float32)
            instances_list.append(track_result.to('cpu'))
            tr = track_result.to('cpu')
            tr = tr[0]
            det_outputs.append(torch.cat([tr.pred_instances.bboxes, tr.pred_instances.scores.unsqueeze(1), tr.pred_instances.labels.unsqueeze(1)], 1))
            track_outputs.append(torch.cat([tr.pred_track_instances.bboxes, tr.pred_track_instances.scores.unsqueeze(1), tr.pred_track_instances.labels.unsqueeze(1), tr.pred_track_instances.instances_id.unsqueeze(1)], 1))

            frame_idx += 1

        save_path = os.path.join(save_dir, snippet_dir+'.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump([embeddings, det_outputs, track_outputs], f)

 


def test_one_video():
    args = parse_args()
    args.out = True
    args.show_fps = False
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    main()
