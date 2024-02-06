import numpy as np
import cv2, os,  subprocess
from tqdm import tqdm
import torch
import platform

# import sys
# sys.path.append('..')
from src.models import Wav2Lip as wav2lip_mdoel
from src.utils import audio
import face_detection

class Wav2Lip:
    def __init__(self, path = 'checkpoints/wav2lip.pth'):
        self.fps = 25
        self.resize_factor = 1
        self.mel_step_size = 16
        self.static = False
        self.img_size = 96
        self.face_det_batch_size = 2
        self.box = [-1, -1, -1, -1]
        self.pads = [0, 10, 0, 0]
        self.nosmooth = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(path)

    def load_model(self, checkpoint_path):
        model = wav2lip_mdoel()
        print("Load checkpoint from: {}".format(checkpoint_path))
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    # def predict(self, face_path, audio_file, batch_size):
    #     if face_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
    #         return self.predict_img(face_path, audio_file, batch_size)
    #     elif face_path.split('.')[1] == 'mp4':
    #         return self.predict_video(face_path, audio_file, batch_size)
    #     else:
    #         return None
                   
    def predict(self, face, audio_file, batch_size):
        os.makedirs('results', exist_ok=True)
        os.makedirs('temp', exist_ok=True)
        frame = cv2.imread(face)
        if self.resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1]//self.resize_factor, frame.shape[0]//self.resize_factor))
        full_frames = [frame]
        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        mel_chunks = []
        mel_idx_multiplier = 80./self.fps 
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]
       
        batch_size = batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks, batch_size)
        
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                                total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter('temp/result.avi', 
                                        cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()

        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_file, 'temp/result.avi', 'results/example_answer.mp4')
        subprocess.call(command, shell=platform.system() != 'Windows')
        return 'results/example_answer.mp4'


    def datagen(self, frames, mels, batch_size):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not self.static:
                face_det_results = self.face_detect(frames) # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.static else i%len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))
                
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def face_detect(self, images):
        try:
            detector = face_detection.FaceAlignment(face_detection.LandmarksType.TWO_D, 
                                                    flip_input=False, device=self.device)
        except:
            detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                    flip_input=False, device=self.device)

        batch_size = self.face_det_batch_size
        
        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    # img_batch = torch.tensor(np.array(images[i:i + batch_size]), device=self.device)
                    # img_batch = img_batch.permute(0, 3, 1, 2)
                    # print(img_batch.shape, type(img_batch))
                    # predictions.extend(detector.get_landmarks_from_batch(img_batch))
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except Exception as e:
                print("Error in face detection: {}".format(e))
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
            
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results 
    
    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
    
if __name__ == '__main__':
    wav2lip = Wav2Lip('../checkpoints/wav2lip.pth')
    wav2lip.predict('../example.png', '../answer.wav', 2)