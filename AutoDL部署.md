# AutoDL部署文档

首先感谢群友的贡献，特此整理一个AutoDL的部署文档来帮助大家



## 环境设置

在autodl上，我选的镜像是：**基础镜像=》pytorch=>1.11.0=>python版本3.8=》cuda版本11.3**



## 代码修改

以下是以`app_multi.py`为例，想使用其他启动模块，模仿这个改就行：

需要修改代码的部分：

- 在Linly-Talker-main=》configs.py：第2行 port = 6006
- 在Linly-Talker-main目录下新建个空目录：results
- 在Linly-Talker-main=》app_multi.py: 第259到264行，改成这样：server_port=port)

**似乎在Gradio4.0以上版本可以不使用证书也可以进行麦克风对话，但是有一些情况还是有bug，所以可以注意看看**

```bash
    #ssl_certfile=ssl_certfile,
    #ssl_keyfile=ssl_keyfile,
    #ssl_verify=False,                
    #debug=True)
```

备注：其他启动模块也都这么改，不然使用autodl ssh隧道工具的时候，端口映射不过去。



## 部署方法

1. 把主文件：**Linly-Talker-main.zip**，权重文件：**gfpgan.zip**、**Qwen.zip**、**sadtalker.zip**都传到`autodl`的**/root/autodl-tmp**上

注意：不要打包到一起传，要将这四个文件一个一个传。因为文件太大了，容易出现问题。

2. 下载压缩解压工具：

```bash
apt-get update && apt-get install -y zip
apt-get update && apt-get install -y unzip
```

3. 把步骤1中的包都解压到当前目录，再将三个权重文件移到主文件的根目录下

```bash
unzip  Linly-Talker-main.zip
unzip  gfpgan.zip
unzip  Qwen.zip
unzip  sadtalker.zip
```

需要将sadtalker改名为checkpoints

```bash
mv sadtalker checkpoints
```

注意，建议权重文件先解压到当前目录下，检查一下，再移到主文件根目录Linly-Talker-main下，例：mv checkpoints Linly-Talker-main。因为有时候解压会有垃圾文件和不创建同名目录的情况，可以检查一下，再移。然后可以把没用的垃圾文件删除掉。

4. 创建虚拟环境，并且安装必要的依赖包

```bash
conda create -n linly python=3.10
source activate linly
conda env list
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -q ffmpeg==4.2.2
pip install -r requirements_app.txt

# Qwen需要
pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
```

5. 执行python app_multi.py

```bash
python app_multi.py
```

6. 执行autodl ssh隧道工具

**在控制台 =》快捷工具=》windows 中，下载AutoDL-SSH-Tools.zip。**

解压该文件，执行AutoDL.exe文件。复制控制台中的ssh登录指令和密码到指定位置，点开始代理，然后点下面的URL

正常会弹出浏览器界面，登录成功。

7. 后续按照视频里操作就行了

注：不用了，一定要去控制台=》容器实例，把镜像实例关机，它是按时收费的，不关机会一直扣费的。

建议选北京区的，稍微便宜一些。

可以晚上部署，网速快，便宜的GPU也充足。白天部署，北京区的GPU容易没有。