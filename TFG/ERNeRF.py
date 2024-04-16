import sys
sys.path.append('/data/dengkaijun/Linly-Talker/NeRF')
import torch
import argparse

from nerf_triplane.provider import NeRFDataset,NeRFDataset_Test
from nerf_triplane.utils import *
from nerf_triplane.network import NeRFNetwork

# torch.autograd.set_detect_anomaly(True)
# Close tf32 features. Fix low numerical accuracy on rtx30xx gpu.
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except AttributeError as e:
    print('Info. This pytorch version is not support with tf32.')


parser = argparse.ArgumentParser()
# parser.add_argument('path', type=str)

# parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")
# parser.add_argument('--test', action='store_true', help="test mode (load model and test dataset)")
parser.add_argument('--test_train', action='store_true', help="test mode (load model and train dataset)")
parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
parser.add_argument('--workspace', type=str, default='results')
parser.add_argument('--seed', type=int, default=0)

### training options
parser.add_argument('--iters', type=int, default=200000, help="training iters")
parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
parser.add_argument('--lr_net', type=float, default=1e-3, help="initial learning rate")
parser.add_argument('--ckpt', type=str, default='/data/dengkaijun/Linly-Talker/checkpoints/pretrained/ngp_kf.pth')
parser.add_argument('--num_rays', type=int, default=4096 * 16, help="num rays sampled per image for each training step")
parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
parser.add_argument('--max_steps', type=int, default=16, help="max num steps sampled per ray (only valid when using --cuda_ray)")
parser.add_argument('--num_steps', type=int, default=16, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

### loss set
parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")

### network backbone options
parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

parser.add_argument('--bg_img', type=str, default='white', help="background image")
parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
parser.add_argument('--fix_eye', type=float, default=-1, help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")

parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")

### dataset options
parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
# (the default value is for the fox dataset)
parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied (sigma)")
parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

parser.add_argument('--init_lips', action='store_true', help="init lips region")
parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

parser.add_argument('--torso', action='store_true', help="fix head and train torso")
parser.add_argument('--head_ckpt', type=str, default='', help="head model")

### GUI options
parser.add_argument('--gui', action='store_true', help="start a GUI")
parser.add_argument('--W', type=int, default=450, help="GUI width")
parser.add_argument('--H', type=int, default=450, help="GUI height")
parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

### else
parser.add_argument('--att', type=int, default=2, help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
parser.add_argument('--aud', type=str, default='', help="audio source (empty will load the default, else should be a path to a npy file)")
parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")

parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
parser.add_argument('--ind_num', type=int, default=10000, help="number of individual codes, should be larger than training dataset size")

parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

# asr
parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
# parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
# parser.add_argument('--asr_play', action='store_true', help="play out the audio")

# parser.add_argument('--asr_model', type=str, default='deepspeech')
parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto')
# parser.add_argument('--asr_model', type=str, default='facebook/wav2vec2-large-960h-lv60-self')
parser.add_argument('--pose', type=str, default='/data/dengkaijun/Linly-Talker/checkpoints/data_kf.json')
parser.add_argument('--asr_save_feats', action='store_true')
# audio FPS
parser.add_argument('--fps', type=int, default=50)
# sliding window left-middle-right length (unit: 20ms)
parser.add_argument('-l', type=int, default=10)
parser.add_argument('-m', type=int, default=50)
parser.add_argument('-r', type=int, default=10)

opt = parser.parse_args()
opt.O = True
opt.test = True
if opt.O:
    opt.fp16 = True
    opt.exp_eye = True
    
if opt.test:
    opt.smooth_path = True
    opt.smooth_eye = True
    opt.smooth_lips = True

opt.cuda_ray = True

opt.torso = True

class ERNeRF():
    def __init__(self):
        print(opt)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeRFNetwork(opt)
        # print(self.model)
        
    def predict(self, asr_wav):
        opt.aud = asr_wav
        criterion = torch.nn.MSELoss(reduction='none')
        metrics = []
        self.trainer = Trainer('ngp', opt, self.model, device=self.device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
        self.test_loader = NeRFDataset_Test(opt, device=self.device).dataloader()
        self.model.aud_features = self.test_loader._data.auds
        self.model.eye_areas = self.test_loader._data.eye_area
        
        self.trainer.test(self.test_loader, 
                          save_path = opt.workspace,
                          name = 'test')
        return os.path.join(opt.workspace, f"test_audio.mp4")    
    
if __name__ == '__main__':
    nerf = ERNeRF(opt)
    wav_path = '../checkpoints/ref.wav'
    nerf.predict(wav_path)