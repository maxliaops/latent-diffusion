# 潜在扩散模型

[arXiv](https://arxiv.org/abs/2112.10752) | [BibTeX](#bibtex)

<p align="center">
<img src=assets/results.gif />
</p>

[**使用潜在扩散模型进行高分辨率图像合成**](https://arxiv.org/abs/2112.10752)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
\* 同等贡献

<p align="center">
<img src=assets/modelfigure.png />
</p>

## 新闻

### 2022 年 7 月
- 推理代码和模型权重可用于运行我们的[检索增强扩散模型](https://arxiv.org/abs/2204.11824)。请参见[此部分](#检索增强扩散模型)。
### 2022 年 4 月
- 感谢[Katherine Crowson](https://github.com/crowsonkb)，分类器自由引导获得了约 2 倍的速度提升，[PLMS 采样器](https://arxiv.org/abs/2202.09778)可用。也可参见[此 PR](https://github.com/CompVis/latent-diffusion/pull/51)。

- 我们的 1.45B[潜在扩散 LAION 模型](#文本到图像)已集成到[Huggingface Spaces 🤗](https://huggingface.co/spaces)中，使用[Gradio](https://github.com/gradio-app/gradio)。试试 Web 演示：[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/multimodalart/latentdiffusion)

- 还有更多预训练的 LDM： 
  - 一个在[LAION-400M](https://arxiv.org/abs/2111.02114)数据库上训练的 1.45B[模型](#文本到图像)。
  - 在 ImageNet 上的类条件模型，使用[分类器自由引导](https://openreview.net/pdf?id=qw8AKxfYbI)时达到 FID 3.6。可通过[Colab 笔记本](https://colab.research.google.com/github/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb) [![][colab]][colab-cin]。

## Requirements
一个合适的[conda](https://conda.io/)环境名为`ldm`，可以通过下面命令创建和激活

```
conda env create -f environment.yaml
conda activate ldm
```

# 预训练模型
通过我们的[模型ZOO](#模型ZOO)可以获得所有可用的检查点的通用列表。如果您在工作中使用任何这些模型，我们总是很高兴收到[引用](#bibtex)。

## 检索增强扩散模型
![rdm-figure](assets/rdm-preview.jpg)
我们包括推理代码来运行我们的检索增强扩散模型（RDMs），如[https://arxiv.org/abs/2204.11824](https://arxiv.org/abs/2204.11824)中所述。

要开始使用，请在您的`ldm`环境中安装额外所需的 Python 包
```shell script
pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
pip install git+https://github.com/arogozhnikov/einops.git
```
并下载训练的权重（初步检查点）：

```bash
mkdir -p models/rdm/rdm768x768/
wget -O models/rdm/rdm768x768/model.ckpt https://ommer-lab.com/files/rdm/model.ckpt
```
由于这些模型是以一组 CLIP 图像嵌入为条件的，我们的 RDMs 支持不同的推理模式，如下所述。
#### 仅带有文本提示的 RDM（不需要明确检索）
由于 CLIP 提供了一个共享的图像/文本特征空间，并且 RDM 在训练期间学会覆盖给定示例的附近区域，我们可以直接采用给定提示的 CLIP 文本嵌入并对其进行条件处理。
通过运行
```
python scripts/knn2img.py  --prompt "一只快乐的熊在读报纸，油画在画布上"
```

#### 带有文本到图像检索的 RDM

为了能够运行基于文本提示和从该提示检索到的图像的 RDM，您还需要下载相应的检索数据库。
我们提供了从[Openimages-](https://storage.googleapis.com/openimages/web/index.html)和[ArtBench-](https://github.com/liaopeiyuan/artbench)数据集提取的两个不同的数据库。
交换数据库会导致模型的不同能力，如下图所示，尽管在两种情况下使用的学习权重是相同的。

下载包含从[Openimages](https://storage.googleapis.com/openimages/web/index.html)（~11GB）和[ArtBench](https://github.com/liaopeiyuan/artbench)提取的 CLIP 图像嵌入的检索数据库：
```bash
mkdir -p data/rdm/retrieval_databases
wget -O data/rdm/retrieval_databases/artbench.zip https://ommer-lab.com/files/rdm/artbench_databases.zip
wget -O data/rdm/retrieval_databases/openimages.zip https://ommer-lab.com/files/rdm/openimages_database.zip
unzip data/rdm/retrieval_databases/artbench.zip -d data/rdm/retrieval_databases/
unzip data/rdm/retrieval_databases/openimages.zip -d data/rdm/retrieval_databases/
```
我们还为 ArtBench 提供了训练的[ScaNN](https://github.com/google-research/google-research/tree/master/scann)搜索索引。通过下载并提取
```bash
mkdir -p data/rdm/searchers
wget -O data/rdm/searchers/artbench.zip https://ommer-lab.com/files/rdm/artbench_searchers.zip
unzip data/rdm/searchers/artbench.zip -d data/rdm/searchers
```

由于 OpenImages 的索引较大（~21GB），我们提供了一个脚本用于在采样期间创建并保存它。但是，请注意，
没有这个索引，使用 OpenImages 数据库进行采样将是不可能的。通过运行脚本
```bash
python scripts/train_searcher.py
```

基于文本引导的带有视觉近邻的检索采样可以通过 
```
python scripts/knn2img.py  --prompt "一个快乐的菠萝" --use_neighbors --knn <邻居数量> 
```
请注意，最大支持的邻居数量为 20。 
数据库可以通过 cmd 参数 ``--database`` 更改，它可以是 `[openimages, artbench-art_nouveau, artbench-baroque, artbench-expressionism, artbench-impressionism, artbench-post_impressionism, artbench-realism, artbench-renaissance, artbench-romanticism, artbench-surrealism, artbench-ukiyo_e]`。
对于使用 `--database openimages`，必须在之前运行上述脚本（`scripts/train_searcher.py`）。
由于其相对较小的尺寸，artbench 数据库最适合创建更抽象的概念，并且对于详细的文本控制效果不佳。 

#### 即将推出
- 更好的模型
- 更多分辨率
- 图像到图像检索

## 文本到图像
![text2img-figure](assets/txt2img-preview.png) 

下载预训练权重（5.7GB）
```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```
并用以下方式进行采样
```
python scripts/txt2img.py --prompt "一个正在弹吉他的病毒怪物，油画布上的油彩" --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50
```
这将在指定的输出位置（默认：`outputs/txt2img-samples`）逐个保存每个样本以及大小为 `n_iter` x `n_samples` 的网格。

质量、采样速度和多样性最好通过 `scale`、`ddim_steps` 和 `ddim_eta` 参数进行控制。通常，`scale` 值越高会产生更好的样本，但输出多样性会降低。

此外，增加 `ddim_steps` 通常也会提供更高质量的样本，但对于值 > 250，回报会逐渐减少。

快速采样（即较低的 `ddim_steps` 值）同时保持良好的质量可以通过使用 `--ddim_eta 0.0` 来实现。更快的采样（即甚至更低的 `ddim_steps` 值）同时保持良好的质量可以通过使用 `--ddim_eta 0.0` 和 `--plms`（参见 [流形上的伪数值扩散模型方法](https://arxiv.org/abs/2202.09778)）来实现。

#### 超过 256²

对于某些输入，仅以卷积方式在比其训练时更大的特征上运行模型有时会产生有趣的结果。要尝试一下，请调整 `H` 和 `W` 参数（它们将被整数除以 8 以计算相应的潜在大小），例如运行

```
python scripts/txt2img.py --prompt "日落在山脉后面，矢量图像" --ddim_eta 1.0 --n_samples 1 --n_iter 1 --H 384 --W 1024 --scale 5.0  
```
以创建大小为 384x1024 的样本。然而，请注意，与 256x256 设置相比，可控性有所降低。

下面的示例是使用上述命令生成的。 
![text2img-figure-conv](assets/txt2img-convsample.png)



## 修复
![inpainting](assets/inpainting.png)

下载预训练权重
```
wget -O models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```

并用以下方式进行采样
```
python scripts/inpaint.py --indir data/inpainting_examples/ --outdir outputs/inpainting_results
```
`indir` 应包含图像 `*.png` 和掩码 `<image_fname>_mask.png`，如 `data/inpainting_examples` 中提供的示例。

## 基于类别的 ImageNet

通过 [notebook](scripts/latent_imagenet_diffusion.ipynb) [![][colab]][colab-cin] 可用。
![class-conditional](assets/birdhouse.png)

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[colab-cin]: <https://colab.research.google.com/github/CompVis/latent-diffusion/blob/main/scripts/latent_imagenet_diffusion.ipynb>


## 无条件模型

我们还提供了一个用于从无条件 LDM（例如 LSUN、FFHQ 等）进行采样的脚本。通过以下方式启动它

```shell 脚本
CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/sample_diffusion.py -r models/ldm/<model_spec>/model.ckpt -l <logdir> -n <\#samples> --batch_size <batch_size> -c <\#ddim steps> -e <\#eta> 
```

# 训练你自己的 LDM

## 数据准备

### 人脸
对于下载 CelebA-HQ 和 FFHQ 数据集，请按照[taming-transformers](https://github.com/CompVis/taming-transformers#celeba-hq) 存储库中所述的步骤进行。

### LSUN 
LSUN 数据集可以通过此处提供的脚本方便地下载[点击这里](https://github.com/fyu/lsun)。我们对训练和验证图像进行了自定义分割，并在[这里](https://ommer-lab.com/files/lsun.zip)提供了相应的文件名。下载后，将它们解压缩到 `./data/lsun`。床/猫/教堂子集也应放置/符号链接到 `./data/lsun/bedrooms` / `./data/lsun/cats` / `./data/lsun/churches`。

### ImageNet
该代码将首次尝试通过[学术Torrents](http://academictorrents.com/)下载并准备 ImageNet。然而，由于 ImageNet 相当大，这需要大量的磁盘空间和时间。如果您的磁盘上已经有 ImageNet，您可以通过将数据放入 `${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/`（默认是 `~/.cache/autoencoders/data/ILSVRC2012_{split}/data/`）来加快速度，其中 `{split}` 是 `train` / `validation` 之一。它应该具有以下结构：

```
${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├──...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├──...
├──...
```
如果您没有提取数据，您也可以将 `ILSVRC2012_img_train.tar` / `ILSVRC2012_img_val.tar`（或它们的符号链接）放入 `${XDG_CACHE}/autoencoders/data/ILSVRC2012_train/` / `${XDG_CACHE}/autoencoders/data/ILSVRC2012_validation/`，然后将其提取到上述结构中，而无需再次下载它。请注意，这只会在既没有 `${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` 文件夹也没有 `${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/.ready` 文件的情况下发生。如果您想要强制再次运行数据集准备，请删除它们。

## 模型训练

日志和已训练模型的检查点被保存到 `logs/<START_DATE_AND_TIME>_<config_spec>`。

### 训练自动编码器模型

在 `configs/autoencoder` 中提供了用于在 ImageNet 上训练 KL 正则化自动编码器的配置。可以通过运行以下命令开始训练：
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/autoencoder/<config_spec>.yaml -t --gpus 0,
```
其中 `config_spec` 是{`autoencoder_kl_8x8x64`(f=32, d=64)，`autoencoder_kl_16x16x16`(f=16, d=16)，`autoencoder_kl_32x32x4`(f=8, d=4)，`autoencoder_kl_64x64x3`(f=4, d=3)} 之一。

对于训练 VQ 正则化模型，请参见[taming-transformers](https://github.com/CompVis/taming-transformers) 存储库。

### 训练 LDM

在 `configs/latent-diffusion/` 中，我们为在 LSUN-、CelebA-HQ、FFHQ 和 ImageNet 数据集上训练 LDM 提供了配置。可以通过运行以下命令开始训练：

```shell 脚本
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t --gpus 0,
```

其中 `<config_spec>` 是{`celebahq-ldm-vq-4`(f=4, VQ 正则化自动编码器，空间大小 64x64x3)，`ffhq-ldm-vq-4`(f=4, VQ 正则化自动编码器，空间大小 64x64x3)，`lsun_bedrooms-ldm-vq-4`(f=4, VQ 正则化自动编码器，空间大小 64x64x3)，`lsun_churches-ldm-vq-4`(f=8, KL 正则化自动编码器，空间大小 32x32x4)，`cin-ldm-vq-8`(f=8, VQ 正则化自动编码器，空间大小 32x32x4)} 之一。

# 模型ZOO

## 预训练自动编码模型
![rec2](assets/reconstruction2.png)

所有模型都训练到收敛（rFID 不再有实质性的改进）。
| Model                   | rFID vs val | train steps           |PSNR           | PSIM          | Link                                                                                                                                                  | Comments              
|-------------------------|------------|----------------|----------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| f=4, VQ (Z=8192, d=3)   | 0.58       | 533066 | 27.43  +/- 4.26 | 0.53 +/- 0.21 |     https://ommer-lab.com/files/latent-diffusion/vq-f4.zip                   |  |
| f=4, VQ (Z=8192, d=3)   | 1.06       | 658131 | 25.21 +/-  4.17 | 0.72 +/- 0.26 | https://heibox.uni-heidelberg.de/f/9c6681f64bb94338a069/?dl=1  | no attention          |
| f=8, VQ (Z=16384, d=4)  | 1.14       | 971043 | 23.07 +/- 3.99 | 1.17 +/- 0.36 |       https://ommer-lab.com/files/latent-diffusion/vq-f8.zip                     |                       |
| f=8, VQ (Z=256, d=4)    | 1.49       | 1608649 | 22.35 +/- 3.81 | 1.26 +/- 0.37 |   https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip |  
| f=16, VQ (Z=16384, d=8) | 5.15       | 1101166 | 20.83 +/- 3.61 | 1.73 +/- 0.43 |             https://heibox.uni-heidelberg.de/f/0e42b04e2e904890a9b6/?dl=1                        |                       |
|                         |            |  |                |               |                                                                                                                                                    |                       |
| f=4, KL                 | 0.27       | 176991 | 27.53 +/- 4.54 | 0.55 +/- 0.24 |     https://ommer-lab.com/files/latent-diffusion/kl-f4.zip                                   |                       |
| f=8, KL                 | 0.90       | 246803 | 24.19 +/- 4.19 | 1.02 +/- 0.35 |             https://ommer-lab.com/files/latent-diffusion/kl-f8.zip                            |                       |
| f=16, KL     (d=16)     | 0.87       | 442998 | 24.08 +/- 4.22 | 1.07 +/- 0.36 |      https://ommer-lab.com/files/latent-diffusion/kl-f16.zip                                  |                       |
 | f=32, KL     (d=64)     | 2.04       | 406763 | 22.27 +/- 3.93 | 1.41 +/- 0.40 |             https://ommer-lab.com/files/latent-diffusion/kl-f32.zip                       

### 获取模型

运行以下脚本下载并提取所有可用的预训练自动编码模型。 
```shell script
bash scripts/download_first_stages.sh
```
然后可以在 `models/first_stage_models/<model_spec>` 中找到第一阶段模型。

## 预训练 LDMs
| Datset                          |   Task    | Model        | FID           | IS              | Prec | Recall | Link                                                                                                                                                                                   | Comments                                        
|---------------------------------|------|--------------|---------------|-----------------|------|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| CelebA-HQ                       | 无条件图像合成    |  LDM-VQ-4 (200 DDIM steps, eta=0)| 5.11 (5.11)          | 3.29            | 0.72    | 0.49 |    https://ommer-lab.com/files/latent-diffusion/celeba.zip     |                                                 |  
| FFHQ                            | 无条件图像合成    |  LDM-VQ-4 (200 DDIM steps, eta=1)| 4.98 (4.98)  | 4.50 (4.50)   | 0.73 | 0.50 |              https://ommer-lab.com/files/latent-diffusion/ffhq.zip                                              |                                                 |
| LSUN-Churches                   | 无条件图像合成   |  LDM-KL-8 (400 DDIM steps, eta=0)| 4.02 (4.02) | 2.72 | 0.64 | 0.52 |         https://ommer-lab.com/files/latent-diffusion/lsun_churches.zip        |                                                 |  
| LSUN-Bedrooms                   | 无条件图像合成   |  LDM-VQ-4 (200 DDIM steps, eta=1)| 2.95 (3.0)          | 2.22 (2.23)| 0.66 | 0.48 | https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip |                                                 |  
| ImageNet                        | 类条件图像合成 | LDM-VQ-8 (200 DDIM steps, eta=1) | 7.77(7.76)* /15.82** | 201.56(209.52)* /78.82** | 0.84* / 0.65** | 0.35* / 0.63** |   https://ommer-lab.com/files/latent-diffusion/cin.zip                                                                   | *: w/ guiding, classifier_scale 10  **: w/o guiding, scores in bracket calculated with script provided by [ADM](https://github.com/openai/guided-diffusion) |   
| Conceptual Captions             |  文本条件图像合成 | LDM-VQ-f4 (100 DDIM steps, eta=0) | 16.79         | 13.89           | N/A | N/A |              https://ommer-lab.com/files/latent-diffusion/text2img.zip                                | finetuned from LAION                            |   
| OpenImages                      | 超分辨率   | LDM-VQ-4     | N/A            | N/A               | N/A    | N/A    |                                    https://ommer-lab.com/files/latent-diffusion/sr_bsr.zip                                    | BSR image degradation                           |
| OpenImages                      | 布局到图像合成    | LDM-VQ-4 (200 DDIM steps, eta=0) | 32.02         | 15.92           | N/A    | N/A    |                  https://ommer-lab.com/files/latent-diffusion/layout2img_model.zip                                           |                                                 | 
| Landscapes      |  语义图像合成   | LDM-VQ-4  | N/A             | N/A               | N/A    | N/A    |           https://ommer-lab.com/files/latent-diffusion/semantic_synthesis256.zip                                    |                                                 |
| Landscapes       |  语义图像合成   | LDM-VQ-4  | N/A             | N/A               | N/A    | N/A    |           https://ommer-lab.com/files/latent-diffusion/semantic_synthesis.zip                                    |             finetuned on resolution 512x512                                     |

### 获取模型

上面列出的 LDM 可以通过以下方式联合下载和提取
```shell script
bash scripts/download_models.sh
```
然后可以在 `models/ldm/<model_spec>` 中找到模型。

## 即将推出...

* 更多针对条件 LDM 的推理脚本。
* 在此期间，您可以使用我们的 Colab 笔记本 https://colab.research.google.com/drive/1xqzUi2iXQXDqXBHQGP9Mqt2YrYW6cx-J?usp=sharing 进行操作。

## 评论 

- 我们的扩散模型代码库在很大程度上建立在[OpenAI 的 ADM 代码库](https://github.com/openai/guided-diffusion)和[https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)之上。感谢开源！
- transformer编码器的实现来自[ x - transformers ](https://github.com/lucidrains/x-transformers) 由[ lucidrains ](https://github.com/lucidrains?tab=repositories) 。 



## 引用

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{https://doi.org/10.48550/arxiv.2204.11824,
  doi = {10.48550/ARXIV.2204.11824},
  url = {https://arxiv.org/abs/2204.11824},
  author = {Blattmann, Andreas and Rombach, Robin and Oktay, Kaan and Ommer, Björn},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Retrieval-Augmented Diffusion Models},
  publisher = {arXiv},
  year = {2022},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}


```
