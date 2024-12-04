'''
ultralytics
'''
import sys
sys.path.append('./')

import argparse
import copy
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from collections import defaultdict
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from src.modelsv2 import Wav2Lip as wav2lip_model
from src.utils import audio

import face_detection

os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.torchalign import FacialLandmarkDetector
from src.utils.utils import decompose_tfm, img_warp, img_warp_back_inv_m, metrix_M
from src.utils.utils import laplacianSmooth


torch.manual_seed(1234)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = wav2lip_model()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def get_video_fps(vfile):
    cap = cv2.VideoCapture(vfile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


class Wav2Lipv2():
    def __init__(self, checkpoint_path = 'checkpoints/wav2lipv2.pth',pretrained_model_dir = 'checkpoints/weights', 
                    pads = [0, 0, 0, 0], audio_smooth = True, rotate = False):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_det = YOLO(f'{pretrained_model_dir}/yolov8n-face/yolov8n-face.pt')

        lmk_net = FacialLandmarkDetector(f'{pretrained_model_dir}/wflw/hrnet18_256x256_p1/')
        lmk_net = lmk_net.to(self.device)
        self.lmk_net = lmk_net.eval()

        self.pads = pads

        self.checkpoint_path = checkpoint_path
        
        self.img_size = (256, 256)
        self.fps = 25

        self.a_alpha = 1.25
        self.audio_smooth = audio_smooth
        self.resize_factor = 1
        self.static = False
        self.kpts_smoother = None
        self.abox_smoother = None

        self.crop = [0, -1, 0 , -1]
        # 拉普拉斯金字塔融合图片大小
        self.lpb_size = 256
        self.mel_step_size = 16
        self.rotate = False
        self.model = load_model(self.checkpoint_path)
        print("Model loaded")

    @staticmethod
    def landmark_to_keypoints(landmark):
        lefteye = np.mean(landmark[60:68, :], axis=0)
        righteye = np.mean(landmark[68:76, :], axis=0)
        nose = landmark[54, :]
        leftmouth = (landmark[76, :] + landmark[88, :]) / 2
        rightmouth = (landmark[82, :] + landmark[92, :]) / 2
        return (lefteye, righteye, nose, leftmouth, rightmouth)

    @torch.no_grad()
    def detect_face(self, face_img):
        boxes = self.face_det(face_img,
                              imgsz=640,
                              conf=0.01,
                              iou=0.5,
                              half=True,
                              augment=False,
                              device=self.device)[0].boxes
        bboxes = boxes.xyxy.cpu().numpy()
        return bboxes

    @torch.no_grad()
    def detect_lmk(self, image, bbox=None):
        if isinstance(bbox, list):
            bbox = np.array(bbox)
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        bbox_tensor = torch.from_numpy(bbox[:, :4])
        landmark = self.lmk_net(img_pil, bbox=bbox_tensor, device=self.device).cpu().numpy()
        return landmark

    def prepare_batch(self, img_batch, mel_batch, img_size):
        img_size_h, img_size_w = img_size
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_batch = img_batch / 255.

        img_masked = img_batch.copy()
        img_masked[:, img_size_h // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3)
        # (B, 80, 16) -> (B, 80, 16, 1)
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        return img_batch, mel_batch

    def build_avatar(self, video_path, fps, max_frame_num=-1):
        full_frames = []
        if video_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(video_path)]
        else:
            video_stream = cv2.VideoCapture(video_path)
            print("fps={}".format(fps))
            print('Reading video frames...')

            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if self.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1] // self.resize_factor, frame.shape[0] // self.resize_factor))

                if self.rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = self.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)

                if max_frame_num > 0 and len(full_frames) >= max_frame_num or self.static:
                    video_stream.release()
                    break

        print("Number of frames available for inference: " + str(len(full_frames)))

        self.kpts_smoother = laplacianSmooth()
        self.abox_smoother = laplacianSmooth()

        frame_info_list = []
        print("-------          1")
        #batch_size = 32
        #for batch in tqdm(range(0, len(full_frames), batch_size)):
            #batch_frames = full_frames[batch:batch + batch_size]
            #batch_frame_info_list = []
            #for frame in batch_frames:
                #imginfo = self.get_input_imginfo(frame.copy())
                #batch_frame_info_list.append(imginfo)
            #frame_info_list.extend(batch_frame_info_list)
        for frame_id in tqdm(range(len(full_frames))):
            imginfo = self.get_input_imginfo(full_frames[frame_id].copy())
            frame_info_list.append(imginfo)
        print("-------          2")
        self.kpts_smoother = None
        self.abox_smoother = None

        frame_h, frame_w = full_frames[0].shape[:2]
        avatar = {
            'fps': fps,
            'frame_num': len(full_frames),
            'frame_h': frame_h,
            'frame_w': frame_w,
            'frame_info_list': frame_info_list
        }
        return avatar

    @torch.no_grad()
    def get_input_imginfo(self, frame):
        bbox = self.detect_face(frame.copy())[0][:5]
        landmark = self.detect_lmk(frame.copy(), [bbox])[0]
        keypoints = self.landmark_to_keypoints(landmark)

        keypoints = self.kpts_smoother.smooth(np.array(keypoints))

        m = metrix_M(face_size=200, expand_size=256, keypoints=keypoints)

        align_frame = img_warp(frame, m, 256, adjust=0)
        align_bbox = self.detect_face(align_frame.copy())[0][:4]

        align_bbox = self.abox_smoother.smooth(np.reshape(align_bbox, (-1, 2))).reshape(-1)

        # 重新warp 图片，保持scale 不变
        w, h = 256, 256
        rt, s = decompose_tfm(m)
        s_x, s_y = s[0][0], s[1][1]
        m = rt
        align_frame = cv2.warpAffine(frame, m, (math.ceil(w / s_x), math.ceil(h / s_y)), flags=cv2.INTER_CUBIC)
        inv_m = cv2.invertAffineTransform(m)

        face = copy.deepcopy(align_frame)
        h, w, c = align_frame.shape
        bbox = align_bbox
        bbox[0] *= (w - 1) / 255
        bbox[1] *= (h - 1) / 255
        bbox[2] *= (w - 1) / 255
        bbox[3] *= (h - 1) / 255

        rect = [round(f) for f in bbox[:4]]
        pady1, pady2, padx1, padx2 = self.pads
        y1 = max(0, rect[1] - pady1)
        y2 = min(h, rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(w, rect[2] + padx2)

        coords = (y1, y2, x1, x2)
        face = face[y1:y2, x1:x2]

        face = cv2.resize(face, self.img_size)

        return {
            'img': face,
            'frame': frame,
            'coords': coords,
            'align_frame': align_frame,
            'm': m,
            'inv_m': inv_m,
        }

    def get_input_imginfo_by_index(self, idx, avatar):
        return avatar['frame_info_list'][idx]

    def get_input_mel_by_index(self, index, wav_mel):
        # 处理音频
        T = 5
        mel_idx_multiplier = 80. / self.fps  # 一帧图像对应3.2帧音频
        start_idx = int((index - (T - 1) // 2) * mel_idx_multiplier)
        if start_idx < 0:
            start_idx = 0
        if start_idx + self.mel_step_size > len(wav_mel[0]):
            start_idx = len(wav_mel[0]) - self.mel_step_size
        mel = wav_mel[:, start_idx: start_idx + self.mel_step_size]
        return mel

    def get_intput_by_index(self, index, wav_mel, avatar):
        mel = self.get_input_mel_by_index(index, wav_mel)

        # 处理图片，视频为正序，倒序，正序，倒序，循环
        frame_num = avatar['frame_num']
        idx = index % frame_num
        idx = idx if index // frame_num % 2 == 0 else frame_num - idx - 1

        input_dict = {'mel': mel}
        input_imginfo = self.get_input_imginfo_by_index(idx, avatar)
        input_dict.update(copy.deepcopy(input_imginfo))
        return input_dict

    def run(self, video_path, audio_path, batch_size = 4, enhance = False, outfile=None, fps = 25):
        if outfile is None:
            key = str(uuid.uuid4().hex)
            outfile = ("results/result_voice_{}.mp4".format(key))


        fps = fps if video_path.split('.')[1] in ['jpg', 'png', 'jpeg'] else get_video_fps(video_path)
        self.fps = fps

        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav")
        temp_audio_file.name = "tempface.wav"
        if not audio_path.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, temp_audio_file.name)
            subprocess.call(command, shell=True)
            wav_path = temp_audio_file.name
        else:
            wav_path = audio_path

        wav = audio.load_wav(wav_path, 16000)
        wav_mel = audio.melspectrogram(wav)
        mel_idx_multiplier = 80. / fps
        gen_frame_num = int(len(wav_mel[0]) / mel_idx_multiplier)

        avatar = self.build_avatar(video_path, fps, max_frame_num=gen_frame_num)
        torch.cuda.empty_cache()

        img_size = self.img_size

        frame_h, frame_w = avatar['frame_h'], avatar['frame_w']

        temp_face_file = tempfile.NamedTemporaryFile(suffix=".mp4")
        temp_face_file.name = "tempface.mp4"
        out = cv2.VideoWriter(temp_face_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

        batch_data = defaultdict(list)

        start_infer = time.time()
        pure_model_time = 0.0

        for i in tqdm(range(gen_frame_num)):
            input_data = self.get_intput_by_index(i, wav_mel, avatar)
            # 组batch
            for k, v in input_data.items():
                batch_data[k + '_batch'].append(v)

            if len(batch_data.get('mel_batch')) == batch_size or i == gen_frame_num - 1:
                infer_size = len(batch_data['mel_batch'])

                img_batch = batch_data['img_batch']
                mel_batch = batch_data['mel_batch']
                frames = batch_data['frame_batch']
                coords = batch_data['coords_batch']
                align_frames = batch_data['align_frame_batch']
                ms = batch_data['m_batch']
                inv_ms = batch_data['inv_m_batch']

                if self.audio_smooth:
                    mel_batch.insert(0, self.get_input_mel_by_index(max(0, i - infer_size), wav_mel))
                    mel_batch.append(self.get_input_mel_by_index(min(i + 1, gen_frame_num - 1), wav_mel))

                img_batch, mel_batch = self.prepare_batch(img_batch, mel_batch, img_size)

                # pytorch 推理
                start_model = time.time()
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
                with torch.no_grad():
                    if self.audio_smooth:
                        audio_embedding = self.model.audio_forward(mel_batch, a_alpha=1.25)
                        audio_embedding = 0.2 * audio_embedding[:-2] + 0.6 * audio_embedding[1:-1] + 0.2 * audio_embedding[2:]
                        pred = self.model.inference(audio_embedding, img_batch)
                    else:
                        pred = self.model(mel_batch, img_batch, a_alpha=1.25)
                pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.

                end_model = time.time()
                pure_model_time += (end_model - start_model)

                for p, f, c, af, inv_m in zip(pred, frames, coords, align_frames, inv_ms):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    af[y1:y2, x1:x2] = p      
                    f = img_warp_back_inv_m(af, f, inv_m)
                    out.write(f)

                batch_data.clear()

        end_infer = time.time()
        latency_per_frame = (end_infer - start_infer) * 1000 / gen_frame_num
        latency_model = pure_model_time * 1000 / gen_frame_num
        print(f"每一帧延迟: {latency_per_frame:.3f} ms")
        print(f"每一帧延迟，纯模型: {latency_model:.3f} ms")

        out.release()
        if enhance:
            import imageio
            from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
            enhancer = 'gfpgan'
            background_enhancer = None
            video_save_dir = 'results'
            video_name_enhance = 'res_enhanced.mp4'
            enhanced_path = os.path.join(video_save_dir, 'temp_'+video_name_enhance)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhance) 
            return_path = av_path_enhancer
            full_video_path = temp_face_file.name
            try:
                enhanced_images_gen_with_len = enhancer_generator_with_len(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(self.fps))
            except:
                enhanced_images_gen_with_len = enhancer_list(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(self.fps))
            command = 'ffmpeg -y -i "{}" -i "{}" -strict -2 -q:v 1 "{}"'.format(wav_path, enhanced_path, outfile)
            subprocess.call(command, shell=platform.system() != 'Windows')
        else:
            command = 'ffmpeg -y -i "{}" -i "{}" -strict -2 -q:v 1 "{}"'.format(wav_path, temp_face_file.name, outfile)
            subprocess.call(command, shell=platform.system() != 'Windows')

        temp_face_file.close()
        temp_audio_file.close()
        return outfile


    
if __name__ == '__main__':
    current_dir = './'
    wav2lipv2 = Wav2Lipv2(os.path.join(current_dir,'checkpoints/wav2lipv2.pth'))
    wav2lipv2.run('Wav2Lip/video.mp4', 'Wav2Lip/video.mp4', batch_size = 16)

    # wav2lipv2.run(os.path.join(current_dir,'inputs/example.png'), os.path.join(current_dir,'answer.wav'), batch_size = 16)
    # wav2lipv2.run(os.path.join(current_dir,'inputs/example.png'), os.path.join(current_dir,'answer.wav'), batch_size = 16, enhance = True)
