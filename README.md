# VLPart：4090 复现与自有数据测试

## 1. 克隆这一个仓库

```bash
mkdir -p /data/yukun/VLPart
cd /data/yukun/VLPart
git clone https://github.com/yiyipupu/VLPart-reproduction.git source
mkdir -p cache/{pip,tmp,torch,matplotlib,conda} checkpoints datasets outputs weights env
cd source
export ROOT=/data/yukun/VLPart
export PIP_CACHE_DIR=$ROOT/cache/pip TMPDIR=$ROOT/cache/tmp
export TORCH_HOME=$ROOT/cache/torch XDG_CACHE_HOME=$ROOT/cache
export MPLCONFIGDIR=$ROOT/cache/matplotlib
```

仓库包含完整上游源码、configs、demo、tools 和 metadata；不包含权重、环境、缓存或私有数据。 

## 2. 配置 RTX 4090 环境（未实机验证）

4090 使用 Ada 架构，推荐候选环境为 Python 3.9、PyTorch 2.0.1/cu118、torchvision 0.15.2，以及 CUDA Toolkit 11.8。不要照搬本机 2080 Ti 的 cu117 环境。先运行 `nvidia-smi` 和 `nvcc --version`，确认驱动支持 CUDA 11.8，且 CUDA Toolkit 11.8 可用；驱动报告的 CUDA 版本不是已安装 Toolkit 版本。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
CONDA_PKGS_DIRS=$ROOT/cache/conda conda create -p "$ROOT/env/vlpart4090" python=3.9 pip -y
conda activate "$ROOT/env/vlpart4090"
python -m pip install pip==24.0 setuptools==59.5.0 wheel==0.38.4
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r reproduction/requirements-4090.txt
python -m pip install --no-build-isolation 'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33'
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@247691a1c292654867deb35cabb27f18d75a1363'
python -m pip check
python -c 'import torch; from detectron2 import _C; print(torch.__version__, torch.cuda.get_device_name()); print(torch.ones(1,device="cuda"))'
```

此配置只是一套固定版本的 4090 候选环境，尚未验证。完整本机环境见 `reproduction/local-2080ti.freeze.txt`，它不作为 4090 默认依赖文件。关键差异与验证记录见 `reproduction/CHANGES.md`。

## 3. 下载权重

```bash
wget -c https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth -O "$ROOT/weights/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth"
mkdir -p "$ROOT/weights/clip"
python - <<'CLIP'
import os,clip
clip.load('RN50',device='cpu',download_root=os.environ['ROOT']+'/weights/clip')
print('CLIP RN50 ready')
CLIP
```

默认模型改为论文完整方法的 SwinBase Cascade Mask R-CNN：LVIS/PACO/Pascal Part/PartImageNet 联合训练并加入 ImageNet 解析部件标注。它是加载官方已训练模型进行推理/评测，不是从零复现训练。所有模型权重统一放在 `weights/`，包括 CLIP RN50；`cache/` 仅存运行/安装缓存。Swin 骨干构建代码的初始化文件加载已被上游注释，完整 checkpoint 包含骨干参数，不另下载训练初始化权重。

内置词表 demo 使用仓库预计算文本嵌入；自定义词表评测需使用已下载的 CLIP RN50。官方 custom demo 的 CLIP 加载采用默认缓存路径，因此本文使用下面显式指定权重路径的内联评测命令，不修改上游加载代码。

[R50/SwinBase 全部权重及对应配置](MODEL_ZOO.md)：R50 是 ResNet50 Mask R-CNN；SwinBase 是更大的 Cascade Mask R-CNN。单数据集权重、联合 LVIS/PACO/其他部件数据权重、加入 ImageNet 标签（in）及解析伪部件标注（inparsed）的权重具有不同训练覆盖，必须匹配配置；不能只换权重文件名。当前未验证其他权重。

## 4. 自有数据格式

```text
/data/yukun/VLPart/datasets/custom/
  images/                  # 原始图片，可包含子目录
  annotations/val.json     # COCO 部件实例标注
  annotations/small.json   # 固定的小样本子集
```

JSON 包含 `images`（id、相对 file_name、width、height）、`categories`（唯一 id、英文 name，例如 `dog:head`）及 `annotations`（唯一 id、image_id、category_id、segmentation、bbox、area、iscrowd）。每个部件实例一条标注；bbox 为像素单位 xywh；segmentation 是 COCO 多边形或 RLE；area 为掩码面积。类别、对象/部件粒度、遮挡规则必须事先统一。无标注图片只能预测，不能报告 AP。类别 ID 可不连续，但文本嵌入必须按类别 ID 升序排列。

## 5. 转换并检查数据

不知道私有数据原始格式，不能声称已有通用转换器。已有 COCO 标注可直接准备为上述格式；语义标签图若没有实例信息，不能无损转换为实例分割真值。上游转换工具已完整保留：`tools/pascal_part_mat2json.py`（Pascal Part MAT）、`tools/partimagenet_format_json.py`（PartImageNet 专用路径变换）。后者不是通用 COCO 转换器。具体准备步骤见 [上游数据说明](datasets/README.md)。

以下命令检查标注、生成固定前 10 张图片子集；不生成新脚本文件：

```bash
python - <<'PY'
import json, os
from PIL import Image
from pycocotools.coco import COCO
root=os.environ['ROOT']+'/datasets/custom'
p=root+'/annotations/val.json'; d=json.load(open(p)); c=COCO(p)
assert len(c.imgs)==len(d['images']) and len(c.anns)==len(d['annotations'])
assert len(c.cats)==len(d['categories'])
assert len({x['name'] for x in d['categories']})==len(c.cats)
for i in d['images']:
    f=os.path.realpath(root+'/images/'+i['file_name'])
    assert os.path.commonpath([f,os.path.realpath(root+'/images')])==os.path.realpath(root+'/images')
    with Image.open(f) as im: assert im.size==(i['width'],i['height'])
for a in d['annotations']:
    assert a['image_id'] in c.imgs and a['category_id'] in c.cats
    m=c.annToMask(a); i=c.imgs[a['image_id']]
    assert m.shape==(i['height'],i['width']) and m.any()
    x,y,w,h=a['bbox']; assert x>=0 and y>=0 and w>0 and h>0
    assert x+w<=i['width']+1 and y+h<=i['height']+1
    assert abs(float(m.sum())-a['area'])<=max(1,0.01*float(m.sum()))
ids={i['id'] for i in sorted(d['images'],key=lambda x:x['id'])[:10]}
s=dict(d); s['images']=[i for i in d['images'] if i['id'] in ids]
s['annotations']=[a for a in d['annotations'] if a['image_id'] in ids]
json.dump(s,open(root+'/annotations/small.json','w'))
print('checked',len(d['images']),'images;',len(d['annotations']),'annotations')
PY
```

多边形的连续几何面积与栅格化面积可能有细微差异；检查失败要人工确认，不要直接篡改真值。还应抽样叠加原始标注核对语义、实例和粒度。该私有数据检查流程尚未对真实数据验证。

## 6. 小样本测试及全量评测

先验证公开图片推理：

```bash
wget -c https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg -O "$ROOT/datasets/dog.jpg"
CUDA_VISIBLE_DEVICES=0 python demo/demo.py --config-file configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.yaml --input "$ROOT/datasets/dog.jpg" --output "$ROOT/outputs/demo" --vocabulary pascal_part --confidence-threshold 0.7 --opts MODEL.WEIGHTS "$ROOT/weights/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth" VIS.BOX False
```

自有数据使用通用 COCOEvaluator，而不是硬编码 Pascal seen/novel 类别的评测器。以下内联命令注册自有类别、按 ID 升序生成官方同款 `a ` CLIP RN50 文本嵌入，并调用现有评测入口。保持官方配置的 800/1333 测试尺寸、100 detections/image、评测分数阈值；demo 的 0.7 只用于展示，不能用它过滤评测结果。

```bash
export SPLIT=small
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import os,json,numpy as np
from types import SimpleNamespace
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from vlpart.config import add_vlpart_config
from demo.predictor import reset_cls_test
import clip, torch
from train_net import do_test
root=os.environ['ROOT']; split=os.environ['SPLIT']
ann=f'{root}/datasets/custom/annotations/{split}.json'
d=json.load(open(ann)); names=[x['name'] for x in sorted(d['categories'],key=lambda x:x['id'])]
register_coco_instances('custom_parts',{},ann,f'{root}/datasets/custom/images')
clip_model,_=clip.load(root+'/weights/clip/RN50.pt',device='cpu')
texts=['a '+x.lower().replace(':',' ') for x in names]
with torch.no_grad():
    emb=torch.cat([clip_model.encode_text(clip.tokenize(texts[i:i+128])) for i in range(0,len(texts),128)]).float().T.contiguous()
del clip_model
cfg=get_cfg(); add_vlpart_config(cfg)
cfg.merge_from_file('configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.yaml')
cfg.DATASETS.TEST=('custom_parts',)
cfg.MODEL.WEIGHTS=f'{root}/weights/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth'
cfg.OUTPUT_DIR=f'{root}/outputs/custom_{split}'
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
model=build_model(cfg); DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
reset_cls_test(model,emb); model.eval()
np.save(cfg.OUTPUT_DIR+'/text_embeddings.npy',emb.numpy())
json.dump(names,open(cfg.OUTPUT_DIR+'/categories.json','w'))
print(do_test(cfg,model))
PY
```

小样本通过后，将 `export SPLIT=small` 改为 `export SPLIT=val`，原样重跑整个内联命令。不要用小样本 AP 冒充全量结果。该自定义评测命令尚未实测；需在 4090 上先完成小样本验证后再全量运行。它使用官方模型与评测函数，仅在运行时注册数据和替换词表。

## 7. 输入、输出与指标

输入是图片及预设或自定义英文部件词表；预测是类别、置信度、框与实例掩码。demo 保存可视化图片；`do_test` 保存 `outputs/custom_<split>/inference/custom_parts/` 下的预测与 COCO 评测结果。指标包括 bbox/segm AP、AP50、AP75、APs/m/l，标准 COCO AP 是 IoU 0.50:0.95，通常按百分数报告。报告数据版本、类别定义、权重、尺寸、阈值和每类 AP。任意私有数据不自动具备论文的 seen/novel 分组。

本机已验证：RTX 2080 Ti、R50 Pascal Part、公开 dog.jpg、11 个预测实例、日志预测耗时 2.04 秒（非基准测试），已人工查看可视化。存在重复/重叠预测，不能视为性能结论。完整 SwinBase 与 CLIP RN50 已下载，但尚未收到完整模型成功运行结果；不能把 R50 基线成功写成完整方法验证成功。4090 未验证；自定义词表评测及真实私有数据未验证。

## 8. 数据集、论文与上游

[论文 Going Denser with Open-Vocabulary Part Segmentation](https://arxiv.org/abs/2305.11173)；[上游 facebookresearch/VLPart](https://github.com/facebookresearch/VLPart)，固定 commit `e48903ab0cb38e4dd5be7ce7da2e36c855eff53f`。研究通过联合部件、对象、图像级数据及 DINO 对应关系生成的伪部件标注扩展泛化；普通 detector 推理无需重新执行 DINO 解析。涉及 Pascal Part、PartImageNet、PACO、VOC、COCO、LVIS、ImageNet。原文档见 [README.upstream.md](README.upstream.md)、[MODEL_ZOO.md](MODEL_ZOO.md)。保留 [MIT LICENSE](LICENSE) 及上游第三方许可说明；CLIP、Detic、DINO 等遵循各自许可证。

### 给师兄的操作顺序

1. 把本仓库链接发给师兄，按第 1–3 节配置 `/data/yukun/VLPart`、4090 环境和两份权重。
2. 新终端先执行下方激活命令，并重新设置第 1 节的缓存环境变量，再跑第 6 节公开图片 demo。
3. 确认退出码成功、终端预测实例数、输出图片实际存在并人工打开检查后，再准备自有数据。
4. 与师兄确认原始标注格式、对象/部件类别及实例定义；按第 4–5 节准备和检查 COCO 标注。
5. 先执行 small，再执行 val；保存环境版本、日志、预测和指标，回传实际结果。不要提前声称 4090 或真实数据已验证。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/yukun/VLPart/env/vlpart4090
export ROOT=/data/yukun/VLPart
cd "$ROOT/source"
export PIP_CACHE_DIR=$ROOT/cache/pip TMPDIR=$ROOT/cache/tmp
export TORCH_HOME=$ROOT/cache/torch XDG_CACHE_HOME=$ROOT/cache
export MPLCONFIGDIR=$ROOT/cache/matplotlib
```

如 Miniconda 安装位置不同，替换 `~/miniconda3`；不要引用 `/data/yipu/` 的环境。

## 9. 常见报错

- 沙箱内 GPU 不可见：在实际主机执行 nvidia-smi 和 CUDA 张量检查，不据此判断无显卡。
- Detectron2 undefined symbol：确保 torch/torchvision 匹配，用当前 torch 重新编译 Detectron2；不要混用预编译 wheel。
- 缺少 pkg_resources：使用固定 setuptools，CLIP 使用 `--no-build-isolation`。
- pycocotools/Matplotlib 数组转换报错：本机通过固定 numpy 1.23.5、Pillow 9.5.0、pycocotools 2.0.7、matplotlib 3.7.5 解决。
- CUDA OOM：先检查其他任务占用，不擅自降低分辨率。官方 AsyncPredictor 为多模型数据并行，不是单模型跨卡拆分。
- 类别维度不匹配：核对文本嵌入维度及排序；RN50 为 1024 维，不能直接用 ViT-B/32 的 512 维。
- 缺少 datasets/metadata：从仓库根目录运行；这些小型词表文件已提交。
