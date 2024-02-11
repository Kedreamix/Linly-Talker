# THG 构建智能数字人

### SadTalker

数字人生成可使用SadTalker（CVPR 2023）,详情介绍见 [https://sadtalker.github.io](https://sadtalker.github.io)

在使用前先下载SadTalker模型:

```bash
bash scripts/sadtalker_download_models.sh  
```

[Baidu (百度云盘)](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`)

> 如果百度网盘下载，记住是放在checkpoints文件夹下，百度网盘下载的默认命名为sadtalker，实际应该重命名为checkpoints



### Wav2Lip

数字人生成还可使用Wav2Lip（ACM 2020），详情介绍见 [https://github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)

在使用前先下载Wav2Lip模型：

| Model                        | Description                                           | Link to the model                                            |
| ---------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| Wav2Lip                      | Highly accurate lip-sync                              | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW) |
| Wav2Lip + GAN                | Slightly inferior lip-sync, but better visual quality | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) |
| Expert Discriminator         | Weights of the expert discriminator                   | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP) |
| Visual Quality Discriminator | Weights of the visual disc trained in a GAN setup     | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQVqH88dTm1HjlK11eNba5gBbn15WMS0B0EZbDBttqrqkg?e=ic0ljo) |

```python
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
```



### ER-NeRF（Comming Soon）

ER-NeRF（ICCV2023）是使用最新的NeRF技术构建的数字人，拥有定制数字人的特性，只需要一个人的五分钟左右到视频即可重建出来，具体可参考 [https://github.com/Fictionarry/ER-NeRF](https://github.com/Fictionarry/ER-NeRF)

后续会针对此更新
