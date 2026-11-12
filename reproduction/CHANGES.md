# 修改与验证记录

- 上游 VLPart commit：e48903ab0cb38e4dd5be7ce7da2e36c855eff53f。
- 完整保留官方模型、demo、train_net.py、configs、tools 和 metadata；未改变网络或测试尺寸；未新增脚本。
- README.upstream.md 保留原 README；新 README 描述独立部署与内联数据检查/自定义评测命令。
- 新增 reproduction 文档及依赖清单；.gitignore 补充排除大文件目录。
- 2080 Ti 实测：Python 3.9.25、torch 1.13.1+cu117、torchvision 0.14.1+cu117；Detectron2 commit 247691a1c292654867deb35cabb27f18d75a1363，CUDA Toolkit 11.8 编译 sm75；CLIP commit a1d071733d7111c9c014f024669f959182114e33。
- 固定 timm 0.5.4、numpy 1.23.5、Pillow 9.5.0、matplotlib 3.7.5、pycocotools 2.0.7；后两项修复本机可视化失败。
- 本机公开 dog.jpg 官方 demo 成功，11 个部件实例；日志耗时 2.04 秒，非性能基准。
- 4090 环境候选改为 torch 2.0.1/cu118、torchvision 0.15.2、sm89；未实机验证，不复制 2080 Ti 环境作为默认。
- 自有数据转换/评测未验证。README 内联命令不构成已验证适配代码；任意原始私有标注仍须先准备为 COCO 部件实例格式。
- 没有上传权重、环境、缓存、输入图片或私有数据；提供下载命令。

## 完整方法交付修正
- 默认下载和推理/评测命令从 R50 Pascal Part 基线改为 SwinBase joint inparsed 完整方法配置。
- CLIP RN50 显式下载到 weights/clip，内联自定义评测直接加载该文件，避免重复下载到 cache。
- 增加环境激活、师兄交接顺序与验证边界；完整 SwinBase 推理尚未验证。
- 模型源码与官方配置不改，不新增脚本。
