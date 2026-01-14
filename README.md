# InSPyReNet_comparison

This repository is a research fork of the official InSPyReNet codebase. It is
used as a baseline to compare against my own unsupervised segmentation method.
The goal here is to keep changes minimal and focused on evaluation and
inference workflows, not to re-implement the original project.

If you are looking for the official implementation and paper, see:
https://github.com/plemeri/InSPyReNet

## What is different from upstream

- This repo is for comparisons only; it is not the official release.
- Changes are kept minimal and documented for reproducibility.
- Dataset root can be configured via `CONSTANT.json` (see below).

## Dataset path configuration

Create a `CONSTANT.json` in the repo root:

```json
{
  "data_path": "/absolute/path/to/data_root"
}
```

When present, `data_path` is used to resolve dataset roots for training/testing
configs if the local `data/...` paths are not found.

## Usage

Use the original configs in `configs/` as usual. For example:

```bash
python run/Test.py -c configs/InSPyReNet_SwinB.yaml
```

If you want to compare on a specific dataset, update `Test.Dataset.sets` in the
config to match the dataset subfolder name under `data_path`.

## Citation

Please cite the original paper if you use this codebase:

```
@inproceedings{kim2022revisiting,
  title={Revisiting Image Pyramid Structure for High Resolution Salient Object Detection},
  author={Kim, Taehun and Kim, Kunhee and Lee, Joonyeong and Cha, Dongmin and Lee, Jiho and Kim, Daijin},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={108--124},
  year={2022}
}
```
