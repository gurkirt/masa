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
import random
from tqdm import tqdm
from demo.video_frame_loader import VideoFrameReader
from matplotlib import pyplot as plt
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')
from demo.save_embeddings import ImageList

def compute_iou_single(yolo_bbox, head_box):
    x1 = max(yolo_bbox[0], head_box[0])
    y1 = max(yolo_bbox[1], head_box[1])
    x2 = min(yolo_bbox[2], head_box[2])
    y2 = min(yolo_bbox[3], head_box[3])
    
    if (x2 - x1) < 0 or (y2 - y1) < 0:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    union = ((head_box[2] - head_box[0]) * (head_box[3] - head_box[1])) + \
            ((yolo_bbox[2] - yolo_bbox[0]) * (yolo_bbox[3] - yolo_bbox[1])) - intersection
    return intersection / union


def compute_iou(yolo_bboxes, head_box):
    ious = []
    for yolo_bbox in yolo_bboxes:
        ious += [compute_iou_single(yolo_bbox, head_box)]    
    return np.asarray(ious), ious.index(max(ious))


def get_frame_embeddings(snippet_det_outputs, snippet_embeddings, yolo_det_outputs, yolo_embeddings, fi, vi, path_count):
    frame_embeddings = []
    person_box = snippet_det_outputs[fi][0,:4]
    person_box = [person_box[0], person_box[1], person_box[2], person_box[3]]
    person_embedding = torch.cat((snippet_embeddings[fi][0], torch.Tensor([0, fi, vi, path_count] + person_box)))
    frame_embeddings.append(person_embedding)

    head_box = snippet_det_outputs[fi][1,:4]
    head_box = [head_box[0], head_box[1], head_box[2], head_box[3]]
    head_embedding = torch.cat((snippet_embeddings[fi][1], torch.Tensor([1, fi, vi, path_count] + head_box)))
    frame_embeddings.append(head_embedding)

    yolo_dets = yolo_det_outputs[fi].clone()
    
    # pdb.set_trace()
    # yolo_dets[:,2] += yolo_dets[:,0]
    # yolo_dets[:,3] += yolo_dets[:,1]

    ious, max_id = compute_iou(yolo_dets, head_box)
    yolo_head_embedding = yolo_embeddings[fi][max_id]
    yolo_head_box = [yolo_dets[max_id, 0], yolo_dets[max_id, 1], yolo_dets[max_id, 2], yolo_dets[max_id, 3]]
    yolo_head_embedding = torch.cat((yolo_head_embedding, torch.Tensor([2, fi, vi, path_count]+yolo_head_box)))
    frame_embeddings.append(yolo_head_embedding)
    num_yolo_boxes = yolo_dets.shape[0]
    
    for ny in range(num_yolo_boxes):
        if ny != max_id:
            yolo_box = [yolo_dets[ny,0], yolo_dets[ny,1], yolo_dets[ny,2], yolo_dets[ny,3]]
            yolo_embedding = torch.cat((yolo_embeddings[fi][ny], torch.Tensor([3, fi, vi, path_count]+yolo_box)))
            frame_embeddings.append(yolo_embedding)
    
    frame_embeddings = torch.stack(frame_embeddings)

    return frame_embeddings    


def plot_image(imgpath, box1, color=(0,255,0), box2=None):
    if isinstance(imgpath, str):
        img = cv2.imread(imgpath)
    else:
        img = imgpath

    for box in [box1, box2]:
        xmin, ymin, xmax, ymax = int(box1[0]), int(box1[1]), int(box1[2]), int(box1[3])
        # xmax, ymax = xmin+w, ymin+h
        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color, 2)
        if box2 is not None:

            xmin, ymin, xmax, ymax = int(box2[0]), int(box2[1]), int(box2[2]), int(box2[3])
            # xmax, ymax = xmin+w, ymin+h
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
    return img


def save_embeddings(args, video_list):
           
    
    from ultralytics import YOLO
    det_model = YOLO(args.det_checkpoint)


    num_vidoes = len(video_list)
    all_embeddings = torch.zeros((num_vidoes*6*6, 264))
    ecount = 0
    path_count = 0
    image_paths = []
    for vi, sample in enumerate(tqdm(video_list)):
        snippet_dir = sample[0]        
        # video_dir = os.path.join(video_list.images_dir, snippet_dir)
        run_type = 'head'
        load_path = os.path.join(args.data_dir, run_type, snippet_dir+'.pkl')
        with open(load_path, 'rb') as f:
            pdata = pickle.load(f)
        
        [snippet_embeddings, snippet_det_outputs, snippet_track_outputs] = pdata
        
        run_type = 'yolo_tracks'
        load_path = os.path.join(args.data_dir, run_type, snippet_dir+'.pkl')
        with open(load_path, 'rb') as f:
            pdata = pickle.load(f)
        
        [yolo_embeddings, yolo_det_outputs, yolo_track_outputs] = pdata
        num_frames = len(snippet_embeddings)
        
        
        video_embeddings = []
        frames_to_visit = [0, 1, 2, 4, 6, 9, 12, 16]
        for fi in range(num_frames):
            if fi not in frames_to_visit:
                continue
            
            frame_id = sample[1][fi][0]
            # pdb.set_trace()
            if snippet_embeddings[fi] is None or yolo_embeddings[fi] is None:
                continue
            if snippet_embeddings[fi].shape[0]<2 or yolo_embeddings[fi].shape[0]<1:
                continue
            
            frame_embeddings = get_frame_embeddings(snippet_det_outputs, snippet_embeddings, yolo_det_outputs, yolo_embeddings, fi, vi, path_count)
            video_embeddings.append(frame_embeddings)
            img_path = os.path.join(args.data_dir, 'images', snippet_dir, frame_id)
            image_paths.append(img_path)
            # print(img_path)
            path_count += 1
        
        num = 0
        # print(video_embeddings[0].shape, len(video_embeddings[1]))
        if len(video_embeddings)>1 and len(video_embeddings[0])>2 and len(video_embeddings[1])>2:
            for frame_embeddings in video_embeddings:
                num = frame_embeddings.shape[0]
                if ecount + num >= all_embeddings.shape[0]:
                    break
            
                all_embeddings[ecount:ecount+num,: ] = frame_embeddings
                ecount += num

        
            if ecount + num >= all_embeddings.shape[0]:
                all_embeddings = all_embeddings[:ecount]
                break
    
    all_embeddings = all_embeddings[:ecount]
    torch.save(all_embeddings, os.path.join(args.data_dir, 'all_embeddings.pth'))
    with open(os.path.join(args.data_dir, 'image_paths.pkl'), 'wb') as f:
        pickle.dump(image_paths, f)
        

def analyze_embeddings(args, video_list):
    all_embeddings = torch.load(os.path.join(args.data_dir, 'all_embeddings.pth'))
    print('all embeddings shape: ', all_embeddings.shape)
    # all_embeddings = all_embeddings[all_embeddings[:,258]<3000,:] 
    box_types = ['person', 'head', 'yolo']
    for label in [0,1,2]:
        label_embeddings = all_embeddings[all_embeddings[:,256]==label,:] 
        meta_info = label_embeddings[:, 256:].to('cuda')
        embeddings = F.normalize(label_embeddings[:,:256], p=2, dim=1)

        cosine_sim = torch.mm(embeddings, embeddings.t())
        cosine_sim -= torch.eye(cosine_sim.shape[0])*2
        video_nums = meta_info[:,2]
        rank1_video = video_nums[torch.argmax(cosine_sim, dim=1)]
        correct_videos = rank1_video == video_nums
        num_correct = torch.sum(correct_videos)
        num_videos = correct_videos.shape[0]
        rate = num_correct/num_videos 
        print(f'Box type {box_types[label]:8s} {num_correct:05d}/{num_videos} {100*rate:02.02f}')


def analyze_embeddings_topk(args, video_list):
    all_embeddings = torch.load(os.path.join(args.data_dir, 'all_embeddings.pth'))
    print('all embeddings shape: ', all_embeddings.shape)
    # all_embeddings = all_embeddings[all_embeddings[:,258]<3000,:] 
    box_types = ['person', 'head', 'yolo']
    for label in [0,1,2]:
        for target_frame in [1,  2, 4, 6, 9, 12, 16]:
            embeddings_src = all_embeddings[torch.logical_and(all_embeddings[:,256]==label, all_embeddings[:,257]==0),:] 
            embeddings_dst = all_embeddings[torch.logical_and(all_embeddings[:,256]==label, all_embeddings[:,257]==target_frame),:] 
            meta_info_src = embeddings_src[:, 256:]
            meta_info_dst = embeddings_dst[:, 256:]
            embeddings_src = F.normalize(embeddings_src[:,:256], p=2, dim=1).to('cuda')
            embeddings_dst = F.normalize(embeddings_dst[:,:256], p=2, dim=1).to('cuda')

            cosine_sim = torch.mm(embeddings_src, embeddings_dst.t())
            
            video_nums_src = meta_info_src[:,2]
            video_nums_dst = meta_info_dst[:,2]
            ranked_score, ranked_video = torch.topk(cosine_sim, 5, dim=1)
            
            ptstr = ''
            for topk in [1,3,5]:
                has_match_in_dst = 0
                matched_in_topk_dst = 0
                for vid, v in enumerate(video_nums_src):
                    if v in video_nums_dst:
                        has_match_in_dst += 1
                        if v in ranked_video[vid][:topk]:
                            matched_in_topk_dst += 1
            
                num_correct = matched_in_topk_dst  # number of correct predictions within top k predictions
                num_videos = has_match_in_dst
                rate = num_correct/num_videos 
                ptstr += f'Top{topk} {num_correct:05d}/{num_videos} Acc {100*rate:05.02f}% '
            
            print(f'Box type {box_types[label]:8s} Target Frame {target_frame:02d} ::'+ptstr) 
                  

def analyze_within_video_embeddings(args, video_list):
    all_embeddings = torch.load(os.path.join(args.data_dir, 'all_embeddings.pth'))    
    videos_to_consider = all_embeddings[all_embeddings[:,256]>2, 258]
    image_paths = []
    with open(os.path.join(args.data_dir, 'image_paths.pkl'), 'rb') as f:
        image_paths = pickle.load(f)
    print('videos to consider shape: ', videos_to_consider.shape)

    v2c = {}
    for v in videos_to_consider:
        v = int(v)
        if v not in v2c:
            v2c[v] = []

    for target_frame in [1, 2, 4, 6, 9, 12, 16]:
        for video in v2c:
            # if video>500:
            #     continue
            video_embeddings = all_embeddings[all_embeddings[:,258]==video,:]
            labelled_embeddings_src = video_embeddings[torch.logical_and(video_embeddings[:,256]==2, video_embeddings[:,257]==0), ]
            labelled_embeddings_dst = video_embeddings[torch.logical_and(video_embeddings[:,256]==2, video_embeddings[:,257]==target_frame), ]
            unlabelled_embeddings_dst = video_embeddings[torch.logical_and(video_embeddings[:,256]==3 , video_embeddings[:,257]== target_frame), ]
            if not(len(labelled_embeddings_src)>0 and len(labelled_embeddings_dst)>0 and len(unlabelled_embeddings_dst)>0):
                v2c[video] = []
                continue
            # print('labelled_embeddings_src shape: ', labelled_embeddings_src.shape)
            labelled_embeddings_src_ = F.normalize(labelled_embeddings_src[:,:256], p=2, dim=1)
            labelled_embeddings_dst_ = F.normalize(labelled_embeddings_dst[:,:256], p=2, dim=1)
            unlabelled_embeddings_dst_ = F.normalize(unlabelled_embeddings_dst[:,:256], p=2, dim=1)
            
            labelled_cosine_sim = torch.mm(labelled_embeddings_src_, labelled_embeddings_dst_.t())
            # labelled_cosine_sim -= torch.eye(labelled_cosine_sim.shape[0])*2

            unlabelled_cosine_sim = torch.mm(labelled_embeddings_src_, unlabelled_embeddings_dst_.t())
            labelled_max_val, labelled_max_id = torch.max(labelled_cosine_sim, dim=1)
            unlabelled_max_val, unlabelled_max_id = torch.max(unlabelled_cosine_sim, dim=1)
            
            correct = labelled_max_val > unlabelled_max_val

            num_correct = torch.sum(correct)
            if num_correct == 0:
                
                src_path = image_paths[int(labelled_embeddings_src[0,259])]
                dst_path = image_paths[int(labelled_embeddings_dst[labelled_max_id,259])]
                src_box = labelled_embeddings_src[0, 260:]

                src_person_box = video_embeddings[1, 260:]

                dst_box = labelled_embeddings_dst[labelled_max_id, 260:][0]
                unl_dst_box = unlabelled_embeddings_dst[unlabelled_max_id, 260:][0]
                dst_person_box = video_embeddings[video_embeddings[:,257]== target_frame, 260:][1]
                # print(src_path, dst_path, labelled_embeddings_dst[labelled_max_id, 259], unlabelled_embeddings_dst[unlabelled_max_id, 259])
                # pdb.set_trace()
                src_image = plot_image(src_path, src_box)
                src_image = plot_image(src_image, src_person_box, color=(255, 0, 0))
                dst_image = plot_image(dst_path, dst_box, box2=unl_dst_box)
                dst_image = plot_image(dst_image, dst_person_box, color=(255, 0, 0))

                joint_image = np.zeros((max(src_image.shape[0], dst_image.shape[0]), src_image.shape[1] + dst_image.shape[1], 3))
                joint_image[:src_image.shape[0], :src_image.shape[1]] = src_image
                joint_image[:dst_image.shape[0], src_image.shape[1]:] = dst_image
                save_dir = os.path.join(args.save_dir, f'gap_{target_frame:02d}')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"{video}_{labelled_max_val[0]:0.3f}_{unlabelled_max_val[0]:0.3f}.png")
                cv2.imwrite(save_path, joint_image)
                # pdb.set_trace()
            num_checks = correct.shape[0]
            rate = num_correct/num_checks 
            v2c[video] = [num_correct, num_checks, rate]

        total_correct = 0
        total_checks = 0
        videos_used = 0
        for v in v2c:
            if len(v2c[v]) == 3:
                total_correct += v2c[v][0]
                total_checks += v2c[v][1]
                videos_used +=1
        if total_checks >0:
            print(f'Target Frame {target_frame:02d} in {videos_used:04d} videos: {total_correct}/{total_checks} {100*total_correct/total_checks:0.2f}')
        else:
            print(f'total correct: {total_correct}/{total_checks} 0%')
    

    return v2c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workdisk/image_data/track_data/')
    parser.add_argument('--save_dir', type=str, default='/mnt/ml-backup/track_data_analysis/')
    parser.add_argument('--det_checkpoint', type=str, default='/mnt/ml-backup/gurkirt_trainings/yolov8_modelVxP3456cl_Conf7_75epochs_SGD0.01_ch1_imgsz1024_meters6.0_angle_bound0.1_asTrue_d27thMay/weights/last.pt')
    args = parser.parse_args()

    temp_file = 'video_list.pkl'
    if os.path.isfile(temp_file):
        with open(temp_file, 'rb') as handle:
            video_list = pickle.load(handle)
    else:
        video_list = ImageList(base_dir=args.data_dir)
        with open(temp_file, 'wb') as handle:
            pickle.dump(video_list, handle)
    

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # save_embeddings(args, video_list)
    # analyze_embeddings(args, video_list)
    # analyze_embeddings_topk(args, video_list)
    analyze_within_video_embeddings(args, video_list)


if __name__ == '__main__':
    main()
