# This is modified repository of HoVer-Net.

# HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images

A multiple branch network that performs nuclear instance segmentation and classification within a single network. The network leverages the horizontal and vertical distances of nuclear pixels to their centres of mass to separate clustered cells. A dedicated up-sampling branch is used to classify the nuclear type for each segmented instance. <br />

Our paper:

[A Pragmatic Machine Learning Approach to Quantify Tumor Infiltrating Lymphocytes in Whole Slide Images](https://arxiv.org/abs/2202.06590)

Original HoVer-Net papers:

[Link to Medical Image Analysis paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub) 

[Link to arxiv](https://arxiv.org/abs/1812.06499v4)


## Set Up Environment using Docker

Use the following repository to set up environment:

[hover_build](https://github.com/uit-hdl/hover_build)


## Getting Started

Inside docker container

Preparation:
1. Edit `generate.sh` 
2. Run `generate.sh` for creating `config.yml`
3. Consider running  `misc/proc_consep_ann.py` and `misc/proc_pannuke_ann.py` once for dataset label preparation

Overall pipeline consits of running scripts consecutively.
1. (optional) `stain_norm.py` - Normalize dataset
2. `extract_patches.py` - Extract smaller patches for training
3. `train.py` - Train the model
4. `infer.py` - Perform inference
5. `process.py` - Perform post-processing
6. (optional) `compute_stats.py` - Evaluate results
7. (optional) `export_model.py` - Export model as (.pb) and as checkpoint


## Repository Structure

- `misc/info.py` contains predefined variables, specific to dataset.
- `config.py` is the configuration file.
- `src/` contains executable files used to run the model.
- `loader/`contains scripts for data loading and self implemented augmentation functions.
- `metrics/`contains evaluation code. 
- `misc/`contains util and data preparation scripts. 
- `model/` contains scripts that define the architecture of the segmentation models. 
- `opt/` contains scripts that define the model hyperparameters. 
- `postproc/` contains post processing utils. 
- `metrics/counts.py` is the file we used for counting TILs and Cancer cells for patients.


## Citation

If any part of this code is used, please give appropriate citation to original authors paper. <br />

BibTex entry: <br />
```
@article{graham2019hover,
  title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
  author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak, Jin Tae and Rajpoot, Nasir},
  journal={Medical Image Analysis},
  pages={101563},
  year={2019},
  publisher={Elsevier}
}
```


## Original authors

* [Quoc Dang Vu](https://github.com/vqdang)
* [Simon Graham](https://github.com/simongraham)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
