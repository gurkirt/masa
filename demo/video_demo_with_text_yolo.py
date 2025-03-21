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

import torch
import torchvision
from torch.multiprocessing import Pool, set_start_method

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

def parse_args():

    parser = argparse.ArgumentParser(description='MASA video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('--det_config', help='Detector Config file')
    parser.add_argument('--masa_config', help='Masa Config file')
    parser.add_argument('--det_checkpoint', help='Detector Checkpoint file')
    parser.add_argument('--masa_checkpoint', help='Masa Checkpoint file')
    parser.add_argument( '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
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
    assert args.out, \
        ('Please specify at least one operation (save the '
         'video) with the argument "--out" ')

    # build the model from a config file and a checkpoint file
    if args.unified:
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
    else:
        if args.detector_type == 'yolov10':
            from ultralytics import YOLO10
            det_model = YOLO10(args.det_checkpoint)

        elif args.detector_type == 'yolov8':
            from ultralytics import YOLO
            det_model = YOLO(args.det_checkpoint)

        elif args.detector_type == 'mmdet':
            det_model = init_detector(args.det_config, args.det_checkpoint, palette='random', device=args.device)
            det_model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
            test_pipeline = Compose(det_model.cfg.test_dataloader.dataset.pipeline)
        else:
            raise Exception('ATM only mmdet or YOLOv8 or YOLOv10 detection models are allowed')
        
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
        
        

    if args.sam_mask:
        print('Loading SAM model...')
        device = args.device
        sam_model = sam_model_registry[args.sam_type](args.sam_path)
        sam_predictor = SamPredictor(sam_model.to(device))

    if os.path.isfile(args.video):
        video_reader = mmcv.VideoReader(args.video)
    elif os.path.isdir(args.video):
        video_reader = VideoFrameReader(args.video)
    else:
        raise Exception('Video should be a file or directory with frames ' + args.video)
    # video_reader = mmcv.
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
    visualizer = VISUALIZERS.build(masa_model.cfg.visualizer)
    # pdb.set_trace()
    if args.out:
        out_shape = (video_reader.width, video_reader.height)
        out_shape = (1024, 1024)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            out_shape)

    frame_idx = 0
    instances_list = []
    frames = []
    fps_list = []
    for frame in track_iter_progress((video_reader, len(video_reader))):

        # unified models mean that masa build upon and reuse the foundation model's backbone features for tracking
        if args.unified:
            track_result = inference_masa(masa_model, frame,
                                          frame_id=frame_idx,
                                          video_len=len(video_reader),
                                          test_pipeline=masa_test_pipeline,
                                          text_prompt=texts,
                                          fp16=args.fp16,
                                          detector_type=args.detector_type,
                                          show_fps=args.show_fps)
            if args.show_fps:
                track_result, fps = track_result
        else:
            if args.detector_type == 'mmdet':
                result = inference_detector(det_model, frame,
                                            text_prompt=texts,
                                            test_pipeline=test_pipeline,
                                            fp16=args.fp16)
                pred_bboxes = result.pred_instances.bboxes
                pred_scores = result.pred_instances.scores
                pred_labels = result.pred_instances.labels
            elif args.detector_type in ['yolov10', 'yolov8']:
                preds = det_model.predict(frame)
                boxes = preds[0].boxes.xyxy
                conf = preds[0].boxes.conf.unsqueeze(1)
                det_bboxes = torch.cat((boxes,conf ),1)
                det_labels = preds[0].boxes.cls
            else:
                raise NotImplementedError('Detector type {} not supported'.format(args.detector_type))
            
            if args.detector_type == 'mmdet':
                # Perfom inter-class NMS to remove nosiy detections
                det_bboxes, keep_idx = batched_nms(boxes=pred_bboxes,
                                                scores=pred_scores,
                                                idxs=pred_labels,
                                                class_agnostic=True,
                                                nms_cfg=dict(type='nms',
                                                                iou_threshold=0.5,
                                                                class_agnostic=True,
                                                                split_thr=100000))

                det_bboxes = torch.cat([det_bboxes,pred_scores[keep_idx].unsqueeze(1)],
                                                dim=1)
                det_labels = pred_labels[keep_idx]

            # pdb.set_trace()
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
            # pdb.set_trace()

        frame_idx += 1
        if 'masks' in track_result[0].pred_track_instances:
            if len(track_result[0].pred_track_instances.masks) >0:
                track_result[0].pred_track_instances.masks = torch.stack(track_result[0].pred_track_instances.masks, dim=0)
                track_result[0].pred_track_instances.masks = track_result[0].pred_track_instances.masks.cpu().numpy()

        track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(torch.float32)
        instances_list.append(track_result.to('cpu'))
        frames.append(frame)
        if args.show_fps:
            fps_list.append(fps)

    if not args.no_post:
        instances_list = filter_and_update_tracks(instances_list, (frame.shape[1], frame.shape[0]))

    if args.sam_mask:
        print('Start to generate mask using SAM!')
        for idx, (frame, track_result) in tqdm.tqdm(enumerate(zip(frames, instances_list))):
            track_result = track_result.to(device)
            track_result[0].pred_track_instances.instances_id = track_result[0].pred_track_instances.instances_id.to(device)
            track_result[0].pred_track_instances = track_result[0].pred_track_instances[(track_result[0].pred_track_instances.scores.float() > args.score_thr).to(device)]
            input_boxes = track_result[0].pred_track_instances.bboxes
            if len(input_boxes) == 0:
                continue
            sam_predictor.set_image(frame)
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            track_result[0].pred_track_instances.masks = masks.squeeze(1).cpu().numpy()
            instances_list[idx] = track_result



    if args.out:
        print('Start to visualize the results...')
        num_cores = max(1, min(os.cpu_count() - 1, 16))
        print('Using {} cores for visualization'.format(num_cores))

        if args.show_fps:
            with Pool(processes=num_cores) as pool:

                frames = pool.starmap(
                    visualize_frame, [(args, visualizer, frame, track_result.to('cpu'), idx, fps) for idx, (frame, fps, track_result) in enumerate(zip(frames, fps_list, instances_list))]
                )
        else:
            with Pool(processes=num_cores) as pool:
                frames = pool.starmap(
                    visualize_frame, [(args, visualizer, frame, track_result.to('cpu'), idx) for idx, (frame, track_result) in
                                      enumerate(zip(frames, instances_list))]
                )
        for frame in frames:
            if args.out:
                video_writer.write(frame[:, :, ::-1])

    if video_writer:
        video_writer.release()
    print('Done')

def test_one_video():
    args = parse_args()
    args.out = True
    args.show_fps = False
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    main()
