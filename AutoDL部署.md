# 在AutoDL平台部署Linly-Talker (0基础小白超详细教程)

<!-- TOC -->

- [在AutoDL平台部署Linly-Talker 0基础小白超详细教程](#%E5%9C%A8autodl%E5%B9%B3%E5%8F%B0%E9%83%A8%E7%BD%B2linly-talker-0%E5%9F%BA%E7%A1%80%E5%B0%8F%E7%99%BD%E8%B6%85%E8%AF%A6%E7%BB%86%E6%95%99%E7%A8%8B)
    - [快速上手直接使用镜像以下安装操作全免](#%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B%E7%9B%B4%E6%8E%A5%E4%BD%BF%E7%94%A8%E9%95%9C%E5%83%8F%E4%BB%A5%E4%B8%8B%E5%AE%89%E8%A3%85%E6%93%8D%E4%BD%9C%E5%85%A8%E5%85%8D)
    - [一、注册AutoDL](#%E4%B8%80%E6%B3%A8%E5%86%8Cautodl)
    - [二、创建实例](#%E4%BA%8C%E5%88%9B%E5%BB%BA%E5%AE%9E%E4%BE%8B)
        - [登录AutoDL，进入算力市场，选择机器](#%E7%99%BB%E5%BD%95autodl%E8%BF%9B%E5%85%A5%E7%AE%97%E5%8A%9B%E5%B8%82%E5%9C%BA%E9%80%89%E6%8B%A9%E6%9C%BA%E5%99%A8)
        - [配置基础镜像](#%E9%85%8D%E7%BD%AE%E5%9F%BA%E7%A1%80%E9%95%9C%E5%83%8F)
        - [无卡模式开机](#%E6%97%A0%E5%8D%A1%E6%A8%A1%E5%BC%8F%E5%BC%80%E6%9C%BA)
    - [三、部署环境](#%E4%B8%89%E9%83%A8%E7%BD%B2%E7%8E%AF%E5%A2%83)
        - [进入终端](#%E8%BF%9B%E5%85%A5%E7%BB%88%E7%AB%AF)
        - [下载代码文件](#%E4%B8%8B%E8%BD%BD%E4%BB%A3%E7%A0%81%E6%96%87%E4%BB%B6)
        - [下载模型文件](#%E4%B8%8B%E8%BD%BD%E6%A8%A1%E5%9E%8B%E6%96%87%E4%BB%B6)
    - [四、Linly-Talker项目](#%E5%9B%9Blinly-talker%E9%A1%B9%E7%9B%AE)
        - [环境安装](#%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)
        - [端口设置](#%E7%AB%AF%E5%8F%A3%E8%AE%BE%E7%BD%AE)
        - [有卡开机](#%E6%9C%89%E5%8D%A1%E5%BC%80%E6%9C%BA)
        - [运行网页版对话webui](#%E8%BF%90%E8%A1%8C%E7%BD%91%E9%A1%B5%E7%89%88%E5%AF%B9%E8%AF%9Dwebui)
        - [端口映射](#%E7%AB%AF%E5%8F%A3%E6%98%A0%E5%B0%84)
        - [体验Linly-Talker（成功）](#%E4%BD%93%E9%AA%8Clinly-talker%E6%88%90%E5%8A%9F)

<!-- /TOC -->



## 快速上手直接使用镜像(以下安装操作全免)

若使用我设定好的镜像，可以直接运行即可，不需要安装环境，直接运行webui.py或者是app_talk.py即可体验，不需要安装任何环境，可直接跳到4.4即可

访问后在自定义设置里面打开端口，默认是6006端口，直接使用运行即可！

```bash
python webui.py
python app_talk.py
```

环境模型都安装好了，直接使用即可，镜像地址在：[https://www.codewithgpu.com/i/Kedreamix/Linly-Talker/Kedreamix-Linly-Talker](https://www.codewithgpu.com/i/Kedreamix/Linly-Talker/Kedreamix-Linly-Talker)，感谢大家的支持



## 一、注册AutoDL

[AutoDL官网](https://www.autodl.com/home) 注册账户好并充值，自己选择机器，我觉得如果正常跑一下，5元已经够了

![注册AutoDL](https://pic1.zhimg.com/v2-210a3e83c7d9d56900e1e4967106832f.png)

## 二、创建实例

### 2.1 登录AutoDL，进入算力市场，选择机器

这一部分实际上我觉得12g都OK的，无非是速度问题而已

![选择RTX 3090机器](https://pic1.zhimg.com/v2-a9c077dbd42d0c1d018db942a340f81b.png)



### 2.2 配置基础镜像

选择镜像，最好选择2.0以上可以体验克隆声音功能，其他无所谓

![配置基础镜像](https://picx.zhimg.com/v2-0a7770dd2e1449a097f72cc8d7e680c0.png)



### 2.3 无卡模式开机

创建成功后为了省钱先关机，然后使用无卡模式开机。
无卡模式一个小时只需要0.1元，比较适合部署环境。

![无卡模式开机](https://picx.zhimg.com/v2-792797164f527f103902949d2b55a036.png)

## 三、部署环境

### 3.1 进入终端

打开jupyterLab，进入数据盘（autodl-tmp），打开终端，将Linly-Talker模型下载到数据盘中。

![进入终端](https://pic1.zhimg.com/v2-ab0bb3d4c1dcada54a3cae20860a981b.png)



### 3.2 下载代码文件

根据Github上的说明，使用命令行下载模型文件和代码文件，利用学术加速会快一点

```bash
# 开启学术镜像，更快的clone代码 参考 https://www.autodl.com/docs/network_turbo/
source /etc/network_turbo

cd /root/autodl-tmp/
# 下载代码
git clone https://github.com/Kedreamix/Linly-Talker.git --depth 1

# 取消学术加速
unset http_proxy && unset https_proxy
```



### 3.3 下载模型文件

我制作一个脚本可以完成下述所有模型的下载，无需用户过多操作。这种方式适合网络稳定的情况，并且特别适合 Linux 用户。对于 Windows 用户，也可以使用 Git 来下载模型。如果网络环境不稳定，用户可以选择使用手动下载方法，或者尝试运行 Shell 脚本来完成下载。脚本具有以下功能。

1. **选择下载方式**: 用户可以选择从三种不同的源下载模型：ModelScope、Huggingface 或 Huggingface 镜像站点。
2. **下载模型**: 根据用户的选择，执行相应的下载命令。
3. **移动模型文件**: 下载完成后，将模型文件移动到指定的目录。
4. **错误处理**: 在每一步操作中加入了错误检查，如果操作失败，脚本会输出错误信息并停止执行。

选择使用`modelscope`来下载会快一点，不需要开学术加速，记得首先需要先安装modelscope库

```sh
# 下载modelscope
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /root/autodl-tmp/Linly-Talker
sh scripts/download_models.sh
```

![下载文件](https://pic1.zhimg.com/v2-5f1edcc7f135797f130dbe1565e4e889.png)

等待一段时间下载完以后，脚本会自动移动到对应的目录

![自动移动目录](https://pic1.zhimg.com/v2-7ed4657a8b45ef529bc62c49ad11eaa2.png)

## 四、Linly-Talker项目

### 4.1 环境安装

进入代码路径，进行安装环境，由于选了镜像是含有pytorch的，所以只需要进行安装其他依赖即可，可能需要花一定的时间，建议直接使用安装好的镜像

```bash
cd /root/autodl-tmp/Linly-Talker

conda install ffmpeg==4.2.2 # ffmpeg==4.2.2

# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple
pip install -r requirements_webui.txt

# 安装有关musetalk依赖
pip install --no-cache-dir -U  openmim
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 

# 安装NeRF-based依赖，可能问题较多，可以先放弃
# 亲测需要有卡开机后再跑这个pytorch3d，需要一定的内存来编译
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 若pyaudio出现问题，可安装对应依赖
sudo apt-get update
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
pip install -r TFG/requirements_nerf.txt
```



### 4.2 有卡开机

进入autodl容器实例界面，执行关机操作，然后进行有卡开机，开机后打开jupyterLab。

查看配置

```bash
nvidia-smi
```

![有卡开机](https://pic1.zhimg.com/v2-c2b3e6ed2d39bb8a1e237b04b05e0480.png)



### 4.3 运行网页版对话webui

需要有卡模式开机，执行下边命令，这里面就跟代码是一模一样的了

```bash
cd /root/autodl-tmp/Linly-Talker
# 第一次运行可能会下载部分nltk，可以使用一下学术加速
source /etc/network_turbo
python webui.py
```

![运行网页版对话webui](https://pica.zhimg.com/v2-472c322a57dc9e30f5c86b253124de87.png)

### 4.4 端口映射

这可以直接打开autodl的自定义服务，默认是6006端口，我们已经设置了，所以直接使用即可

![端口映射](https://pic1.zhimg.com/v2-c25c84053dc971c8b8258ce8fdb3667e.png)

另外还有一种端口映射方式，是通过输入ssh账密实现的，步骤是一样的

> ssh端口映射工具：windows：[https://autodl-public.ks3-cn-beijing.ksyuncs.com/tool/AutoDL-SSH-Tools.zip](https://autodl-public.ks3-cn-beijing.ksyuncs.com/tool/AutoDL-SSH-Tools.zip)

### 4.5 体验Linly-Talker（成功）

点开网页，即可正确执行Linly-Talker，这一部分就跟视频一模一样了

![体验Linly-Talker](https://picx.zhimg.com/v2-1559a5e3af76198e494bab29c5574b2d.png)



![MuseTalk](https://picx.zhimg.com/v2-9b997ecb8d66250c9c228702f3f54ab3.png)



**！！！注意：不用了，一定要去控制台=》容器实例，把镜像实例关机，它是按时收费的，不关机会一直扣费的。**

**建议选北京区的，稍微便宜一些。可以晚上部署，网速快，便宜的GPU也充足。白天部署，北京区的GPU容易没有。**