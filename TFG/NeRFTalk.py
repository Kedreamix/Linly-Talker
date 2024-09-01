import os
import sys
sys.path.append('./NeRF')
import torch

from nerf_triplane.provider import NeRFDataset_Test
from nerf_triplane.utils import *
from nerf_triplane.network import NeRFNetwork

# Disable tf32 features to fix low numerical accuracy on RTX30XX GPUs
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except AttributeError as e:
    print('Info. This PyTorch version does not support tf32.')

# Define options directly instead of using argparse
class Options:
    def __init__(self):
        self.test_train = False
        self.data_range = [0, -1]
        self.workspace = 'results'
        self.seed = 0
        self.iters = 200000
        self.lr = 1e-2
        self.lr_net = 1e-3
        self.ckpt = '../checkpoints/pretrained/ngp_kf.pth'
        self.num_rays = 4096 * 16
        self.cuda_ray = True
        self.max_steps = 16
        self.num_steps = 16
        self.upsample_steps = 0
        self.update_extra_interval = 16
        self.max_ray_batch = 4096
        self.warmup_step = 10000
        self.amb_aud_loss = 1
        self.amb_eye_loss = 1
        self.unc_loss = 1
        self.lambda_amb = 1e-4
        self.fp16 = True
        self.bg_img = 'white'
        self.fbg = False
        self.exp_eye = True
        self.fix_eye = -1
        self.smooth_eye = True
        self.torso_shrink = 0.8
        self.color_space = 'srgb'
        self.preload = 0
        self.bound = 1
        self.scale = 4
        self.offset = [0, 0, 0]
        self.dt_gamma = 1/256
        self.min_near = 0.05
        self.density_thresh = 10
        self.density_thresh_torso = 0.01
        self.patch_size = 1
        self.init_lips = False
        self.finetune_lips = False
        self.smooth_lips = True
        self.torso = True
        self.head_ckpt = ''
        self.gui = False
        self.W = 450
        self.H = 450
        self.radius = 3.35
        self.fovy = 21.24
        self.max_spp = 1
        self.att = 2
        self.aud = ''
        self.emb = False
        self.ind_dim = 4
        self.ind_num = 10000
        self.ind_dim_torso = 8
        self.amb_dim = 2
        self.part = False
        self.part2 = False
        self.train_camera = False
        self.smooth_path = True
        self.smooth_path_window = 7
        self.asr = False
        self.asr_model = 'ave'
        self.pose = '../checkpoints/data_kf.json'
        self.asr_save_feats = False
        self.fps = 50
        self.l = 10
        self.m = 50
        self.r = 10
        self.O = True
        self.test = True

# Initialize options
opt = Options()

if opt.O:
    opt.fp16 = True
    opt.exp_eye = True

if opt.test:
    opt.smooth_path = True
    opt.smooth_eye = True
    opt.smooth_lips = True

opt.cuda_ray = True
opt.torso = True

class NeRFTalk():
    def __init__(self):
        print(vars(opt))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeRFNetwork(opt)
        
    def init_model(self, ckpt_path, pose):
        criterion = torch.nn.MSELoss(reduction='none')
        opt.pose = pose
        metrics = []
        opt.ckpt = ckpt_path
        self.trainer = Trainer('ngp', opt, self.model, device=self.device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
                
    def predict(self, asr_wav):
        opt.aud = asr_wav
        
        self.test_loader = NeRFDataset_Test(opt, device=self.device).dataloader()
        self.model.aud_features = self.test_loader._data.auds
        self.model.eye_areas = self.test_loader._data.eye_area
        
        self.trainer.test(self.test_loader, 
                          save_path = opt.workspace,
                          name = 'test')
        return os.path.join(opt.workspace, f"test_audio.mp4")    
    
if __name__ == '__main__':
    nerf = NeRFTalk()
    nerf.init_model('./checkpoints/Obama_ave.pth', './checkpoints/Obama.json')
    print('init done')
    wav_path = './checkpoints/ref.wav'
    nerf.predict(wav_path)
