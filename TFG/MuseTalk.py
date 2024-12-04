import os
import sys
sys.path.append('./Musetalk')
import os
import time
import re
# from huggingface_hub import snapshot_download
import requests
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from argparse import Namespace
import gdown
import imageio
import json
import shutil
import threading
import queue
from moviepy.editor import *
from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder,get_bbox_range
from musetalk.utils.blending import get_image,get_image_prepare_material,get_image_blending
from musetalk.utils.utils import load_all_model
import gradio as gr
# ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = "Musetalk/Musetalk/models"

def download_model():
    if not os.path.exists(CheckpointsDir):
        os.makedirs(CheckpointsDir)
        print("Checkpoint Not Downloaded, start downloading...")
        tic = time.time()
        snapshot_download(
            repo_id="TMElyralab/MuseTalk",
            local_dir=CheckpointsDir,
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        # weight
        os.makedirs(f"{CheckpointsDir}/sd-vae-ft-mse/")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=CheckpointsDir+'/sd-vae-ft-mse',
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        #dwpose
        os.makedirs(f"{CheckpointsDir}/dwpose/")
        snapshot_download(
            repo_id="yzd-v/DWPose",
            local_dir=CheckpointsDir+'/dwpose',
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        #vae
        url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
        response = requests.get(url)
        # 确保请求成功
        if response.status_code == 200:
            # 指定文件保存的位置
            file_path = f"{CheckpointsDir}/whisper/tiny.pt"
            os.makedirs(f"{CheckpointsDir}/whisper/")
            # 将文件内容写入指定位置
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}")
        #gdown face parse
        url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
        os.makedirs(f"{CheckpointsDir}/face-parse-bisent/")
        file_path = f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth"
        gdown.download(url, file_path, quiet=False)
        #resnet
        url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        response = requests.get(url)
        # 确保请求成功
        if response.status_code == 200:
            # 指定文件保存的位置
            file_path = f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
            # 将文件内容写入指定位置
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}")


        toc = time.time()

        print(f"download cost {toc-tic} seconds")
        print_directory_contents(CheckpointsDir)

    else:
        print("Already download the model.")
        
        
# download_model()  # for huggingface deployment.
def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

class MuseTalk_RealTime:
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device = device
        self.load = False
        # self.avatar_info = {
        #     "avatar_id":avatar_id,
        #     "video_path":video_path,
        #     "bbox_shift":bbox_shift   
        # }
        self.skip_save_images = False
        
        self.avatar_id = None
        self.avatar_path = None
        self.full_imgs_path = None
        self.coords_path = None
        self.latents_out_path = None
        self.video_out_path = None
        self.mask_out_path = None
        self.mask_coords_path = None
        self.avatar_info_path = None
        self.input_latent_list_cycle = None
        self.mask_coords_list_cycle = None
        self.mask_list_cycle = None
        self.frame_list_cycle = None

    def init_model(self):
        # load model weights
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        self.timesteps = torch.tensor([0], device=self.device)
        self.pe = self.pe.half()
        self.vae.vae = self.vae.vae.half()
        self.unet.model = self.unet.model.half()
        self.load = True
    
    def process_frames(self, 
                       res_frame_queue,
                       video_len):
        print(video_len)
        while True:
            if self.idx>=video_len-1:
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
      
            bbox = self.coord_list_cycle[self.idx%(len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx%(len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
                continue
            mask = self.mask_list_cycle[self.idx%(len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx%(len(self.mask_coords_list_cycle))]
            #combine_frame = get_image(ori_frame,res_frame,bbox)
            combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

            if self.skip_save_images is False:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png",combine_frame)
            self.idx = self.idx + 1
    
    def prepare_material(self, video_path, bbox_shift, progress=gr.Progress(track_tqdm=True)):
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.avatar_id = os.path.basename(video_path).split(".")[0]
        self.avatar_path = f"./results/avatars/{self.avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs" 
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path= f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path =f"{self.avatar_path}/mask"
        self.mask_coords_path =f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        # 若存在先删除
        if os.path.exists(self.full_imgs_path):
            shutil.rmtree(self.full_imgs_path)
            shutil.rmtree(self.mask_out_path)
            shutil.rmtree(self.video_out_path)
        osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_path,self.mask_out_path])
        print("preparing data materials ... ...")
        progress(0, desc = "preparing data materials ...")
        if os.path.isfile(video_path):
            video2imgs(video_path, self.full_imgs_path, ext = 'png')
        else:
            print(f"copy files in {video_path}")
            files = os.listdir(video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1]=="png"]
            for filename in files:
                shutil.copyfile(f"{video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        # bbox_shift_text = get_bbox_range(input_img_list, self.bbox_shift)
        
        progress(0, desc = "extracting landmarks...")
        print("extracting landmarks ...")
        coord_list, frame_list, bbox_shift_text = get_landmark_and_bbox(input_img_list, bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient 
        coord_placeholder = (0.0,0.0,0.0,0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        progress(0, desc = "saving masks...")
        for i,frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png",frame)
            
            face_box = self.coord_list_cycle[i]
            mask,crop_box = get_image_prepare_material(frame,face_box)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png",mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)
            
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
            
        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path)) 
        return video_path, bbox_shift_text
    
    def prepare_material_(self):
        print("preparing data materials ... ...")
        # with open(self.avatar_info_path, "w") as f:
        #     json.dump(self.avatar_info, f)
            
        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext = 'png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1]=="png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        bbox_shift_text = get_bbox_range(input_img_list, self.bbox_shift)
        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient 
        coord_placeholder = (0.0,0.0,0.0,0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i,frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png",frame)
            
            face_box = self.coord_list_cycle[i]
            mask,crop_box = get_image_prepare_material(frame,face_box)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png",mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)
            
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
            
        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path)) 
        return bbox_shift_text
    
    def inference_noprepare(self, audio_path,
                    source_video, bbox_shift,
                    batch_size = 4,
                    fps = 25,
                    progress = gr.Progress(track_tqdm=True)):
        
        out_vid_name = "res"
        if not self.avatar_path:
            video_path = source_video
            self.avatar_id = os.path.basename(video_path).split(".")[0]
            self.avatar_path = f"./results/avatars/{self.avatar_id}"
        os.makedirs(self.avatar_path+'/tmp',exist_ok =True)   
        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)   
        res_frame_queue = queue.Queue()
        self.idx = 0
        # # Create a sub-thread and start it
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num))
        process_thread.start()

        gen = datagen(whisper_chunks,
                      self.input_latent_list_cycle, 
                      batch_size)
        start_time = time.time()
        res_frame_list = []
        
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                         dtype=self.unet.model.dtype)
            audio_feature_batch = self.pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(latent_batch, 
                                      self.timesteps, 
                                      encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        # Close the queue and sub-thread after all tasks are completed
        process_thread.join()
        
        if self.skip_save_images is True:
            print('Total process time of {} frames without saving images = {}s'.format(
                        video_num,
                        time.time()-start_time))
        else:
            print('Total process time of {} frames including saving images = {}s'.format(
                        video_num,
                        time.time()-start_time))

        if out_vid_name is not None and self.skip_save_images is False: 
            # optional
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            print(cmd_img2video)
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name+".mp4") # on
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")
            shutil.rmtree(f"{self.avatar_path}/tmp")
            print(f"result is save to {output_vid}")
        print("\n")
        return output_vid
    def inference(self, audio_path,
                  source_video, bbox_shift, 
                  batch_size = 4,
                  fps = 25,
                  progress = gr.Progress(track_tqdm=True)):
        self.video_path = source_video
        self.bbox_shift = bbox_shift
        self.avatar_id = os.path.basename(source_video).split(".")[0]
        self.avatar_path = f"./results/avatars/{self.avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs" 
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path= f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path =f"{self.avatar_path}/mask"
        self.mask_coords_path =f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_path,self.mask_out_path])
        bbox_shift_text = None
        if os.path.exists(self.avatar_path):
            response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
            if response.lower() == "y":
                shutil.rmtree(self.avatar_path)
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_path,self.mask_out_path])
                bbox_shift_text = self.prepare_material_()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)
        else:
            print("*********************************")
            print(f"  creating avator: {self.avatar_id}")
            print("*********************************")
            osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_path,self.mask_out_path])
            bbox_shift_text = self.prepare_material_()
        
        if self.input_latent_list_cycle is None:
            self.input_latent_list_cycle = torch.load(self.latents_out_path)
        
        if self.mask_list_cycle is None:
            with open(self.coords_path, 'rb') as f:
                self.coord_list_cycle = pickle.load(f)
        
        if self.frame_list_cycle is None:
            input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.frame_list_cycle = read_imgs(input_img_list)
        
        if self.mask_coords_list_cycle is None:
            with open(self.mask_coords_path, 'rb') as f:
                self.mask_coords_list_cycle = pickle.load(f)
        
        if self.mask_list_cycle is None:
            input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
            input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.mask_list_cycle = read_imgs(input_mask_list)
        
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        if bbox_shift_text is None:
            bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)
        
        out_vid_name = "res"
        os.makedirs(self.avatar_path+'/tmp',exist_ok =True)   
        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)   
        res_frame_queue = queue.Queue()
        self.idx = 0
        # # Create a sub-thread and start it
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num))
        process_thread.start()

        gen = datagen(whisper_chunks,
                      self.input_latent_list_cycle, 
                      batch_size)
        start_time = time.time()
        res_frame_list = []
        
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                         dtype=self.unet.model.dtype)
            audio_feature_batch = self.pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(latent_batch, 
                                      self.timesteps, 
                                      encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        # Close the queue and sub-thread after all tasks are completed
        process_thread.join()
        
        if self.skip_save_images is True:
            print('Total process time of {} frames without saving images = {}s'.format(
                        video_num,
                        time.time()-start_time))
        else:
            print('Total process time of {} frames including saving images = {}s'.format(
                        video_num,
                        time.time()-start_time))

        if out_vid_name is not None and self.skip_save_images is False: 
            # optional
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            print(cmd_img2video)
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name+".mp4") # on
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")
            shutil.rmtree(f"{self.avatar_path}/tmp")
            print(f"result is save to {output_vid}")
        print("\n")
        return output_vid, bbox_shift_text
            
class MuseTalk:
    def __init__(self):
        # load model weights
        self.audio_processor, self.vae, self.unet, self.pe  = load_all_model()
        import platform
        if torch.cuda.is_available():
            device = "cuda"
        elif platform.system() == 'Darwin': # macos 
            device = "mps"
        else:
            device = "cpu"
        self.timesteps = torch.tensor([0], device=device)
    
        
    @torch.no_grad()
    def inference(self, audio_path, video_path, bbox_shift):
        args_dict={"result_dir":'./results/output', "fps":25, "batch_size":8, "output_vid_name":'', "use_saved_coord":False}#same with inferenece script
        args = Namespace(**args_dict)
        print(args)
        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename  = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}"
        result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
        crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
        os.makedirs(result_img_save_path,exist_ok =True)

        if args.output_vid_name=="":
            output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
        else:
            output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
        ############################################## extract frames from source video ##############################################
        if get_file_type(video_path)=="video":
            save_dir_full = os.path.join(args.result_dir, input_basename)
            os.makedirs(save_dir_full,exist_ok = True)
            # cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
            # os.system(cmd)
            # 读取视频
            reader = imageio.get_reader(video_path)

            # 保存图片
            for i, im in enumerate(reader):
                imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
        else: # input img folder
            input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = args.fps
        #print(input_img_list)
        ############################################## extract audio feature ##############################################
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        ############################################## preprocess input image  ##############################################
        if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
            print("using extracted coordinates")
            with open(crop_coord_save_path,'rb') as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            print("extracting landmarks...time consuming")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
            with open(crop_coord_save_path, 'wb') as f:
                pickle.dump(coord_list, f)
        bbox_shift_text=get_bbox_range(input_img_list, bbox_shift)
        i = 0
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)

        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        ############################################## inference batch by batch ##############################################
        print("start inference")
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
        res_frame_list = []
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            
            tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
            audio_feature_batch = torch.stack(tensor_list).to(self.unet.device) # torch, B, 5*N,384
            audio_feature_batch = self.pe(audio_feature_batch)
            
            pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
                
        ############################################## pad to full image ##############################################
        print("pad talking image to original video")
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i%(len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
        #                 print(bbox)
                continue
            
            combine_frame = get_image(ori_frame,res_frame,bbox)
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
            
        # cmd_img2video = f"ffmpeg -y -v fatal -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p temp.mp4"
        # print(cmd_img2video)
        # os.system(cmd_img2video)
        # 帧率
        fps = 25
        # 图片路径
        # 输出视频路径
        output_video = 'temp.mp4'

        # 读取图片
        def is_valid_image(file):
            pattern = re.compile(r'\d{8}\.png')
            return pattern.match(file)

        images = []
        files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
        files.sort(key=lambda x: int(x.split('.')[0]))

        for file in files:
            filename = os.path.join(result_img_save_path, file)
            images.append(imageio.imread(filename))
            

        # 保存视频
        imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')

        # cmd_combine_audio = f"ffmpeg -y -v fatal -i {audio_path} -i temp.mp4 {output_vid_name}"
        # print(cmd_combine_audio)
        # os.system(cmd_combine_audio)

        input_video = './temp.mp4'
        # Check if the input_video and audio_path exist
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video file not found: {input_video}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # 读取视频
        reader = imageio.get_reader(input_video)
        fps = reader.get_meta_data()['fps']  # 获取原视频的帧率

        # 将帧存储在列表中
        frames = images

        # 保存视频并添加音频
        # imageio.mimwrite(output_vid_name, frames, 'FFMPEG', fps=fps, codec='libx264', audio_codec='aac', input_params=['-i', audio_path])
        
        # input_video = ffmpeg.input(input_video)
        
        # input_audio = ffmpeg.input(audio_path)
        
        print(len(frames))

        # imageio.mimwrite(
        #     output_video,
        #     frames,
        #     'FFMPEG',
        #     fps=25,
        #     codec='libx264',
        #     audio_codec='aac',
        #     input_params=['-i', audio_path],
        #     output_params=['-y'],  # Add the '-y' flag to overwrite the output file if it exists
        # )
        # writer = imageio.get_writer(output_vid_name, fps = 25, codec='libx264', quality=10, pixelformat='yuvj444p')
        # for im in frames:
        #     writer.append_data(im)
        # writer.close()

        # Load the video
        video_clip = VideoFileClip(input_video)

        # Load the audio
        audio_clip = AudioFileClip(audio_path)

        # Set the audio to the video
        video_clip = video_clip.set_audio(audio_clip)

        # Write the output video
        video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=25)

        os.remove("temp.mp4")
        #shutil.rmtree(result_img_save_path)
        print(f"result is save to {output_vid_name}", bbox_shift_text)
        return output_vid_name, bbox_shift_text

    def check_video(self, video):
        if not isinstance(video, str):
            return video # in case of none type
        # Define the output video file name
        dir_path, file_name = os.path.split(video)
        if file_name.startswith("outputxxx_"):
            return video
        # Add the output prefix to the file name
        output_file_name = "outputxxx_" + file_name

        os.makedirs('./results',exist_ok=True)
        os.makedirs('./results/output',exist_ok=True)
        os.makedirs('./results/input',exist_ok=True)

        # Combine the directory path and the new file name
        output_video = os.path.join('./results/input', output_file_name)


        # # Run the ffmpeg command to change the frame rate to 25fps
        # command = f"ffmpeg -i {video} -r 25 -vcodec libx264 -vtag hvc1 -pix_fmt yuv420p crf 18   {output_video}  -y"

        # read video
        reader = imageio.get_reader(video)
        fps = reader.get_meta_data()['fps']  # get fps from original video

        # conver fps to 25
        frames = [im for im in reader]
        target_fps = 25
        
        L = len(frames)
        L_target = int(L / fps * target_fps)
        original_t = [x / fps for x in range(1, L+1)]
        t_idx = 0
        target_frames = []
        for target_t in tqdm(range(1, L_target+1)):
            while target_t / target_fps > original_t[t_idx]:
                t_idx += 1      # find the first t_idx so that target_t / target_fps <= original_t[t_idx]
                if t_idx >= L:
                    break
            target_frames.append(frames[t_idx])

        # save video
        imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
        return output_video

    
if __name__ == "__main__":
    # musetalk = MuseTalk()
    musetalk = MuseTalk_RealTime()
    audio_path = "Musetalk/data/audio/sun.wav"
    video_path = "Musetalk/data/video/yongen_musev.mp4"
    bbox_shift = 5
    video_path, bbox_shift_text = musetalk.prepare_material(video_path, bbox_shift)
    # print(video_path, bbox_shift_text)
    print("Inference Params:", audio_path, video_path, bbox_shift)
    res_video = musetalk.inference_noprepare(audio_path, video_path, bbox_shift)

    # output_video = musetalk.check_video(video_path)
    # print("output_video:", output_video)
    # res_video, bbox_shift_scale = musetalk.inference(audio_path, video_path, bbox_shift)
    # print(bbox_shift_scale)
    