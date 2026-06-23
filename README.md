# 写给饼子哥的部署文档

完整项目在 `FinalProject/` 子目录。

```bash
cd FinalProject
cat README.md            # ⭐ 主文档: 项目结构、跑实验、自研模型改进指南
```

TL;DR：
- `main` 和 `mps-perf` 两个分支内容基本一致（同样的模型 / 文档 / 消融脚本）
- 区别：`mps-perf` 多了一些 M5 Mac 的 MPS 优化（数据预加载、unified memory 路径），CUDA 用户用 `main` 就够
- 数据集在 `FinalProject/dataset/`，已就绪，不用下载
- 跑 `python scripts/train_line4b_lite.py` 做冒烟测试（~15 分钟）
- 详细目录、5 条 train_line、自研模型改进方向都在 FinalProject/README.md 里
