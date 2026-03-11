## Overview
Simple python-based interactive GUI to annotate videos with SAM2  


## Setup
 * Create a virtual environment of your choice and activate it: `conda create -n sam2_ann python=3.11 && conda activate sam2_ann`
 * Install `torch>=2.5.1` and `torchvision>=0.20.1` following the instructions from [here](https://pytorch.org/get-started/locally/)
 * Install the dependencies using `pip install -r requirements.txt`
 * Install SAM2 using `cd src/sam2 && pip install -e .`
 * Install SAM2 Predictor using `pip install -e ".[notebooks]"`
 * Place SAM2 (sam2.1_hiera_large) [checkpoints](https://github.com/facebookresearch/sam2/tree/main#model-description) at `src/sam2/checkpoints`
 * Maybe needed to fix could not load the Qt platform: Delete the libqxcb from "../anaconda3/envs/sam2_ann/lib/python3.11/site-packages/cv2/qt/plugins/platforms

## Instructions


## Citations
If you use this repo in your research, please cite our paper:

@inproceedings{sivakumar2025sg2vid,
  title={Sg2vid: Scene graphs enable fine-grained control for video synthesis},
  author={Sivakumar, Ssharvien Kumar and Frisch, Yannik and Ghazaei, Ghazal and Mukhopadhyay, Anirban},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={511--521},
  year={2025},
  organization={Springer}
}

@article{sivakumar2025sasvi,
  title={SASVi: segment any surgical video},
  author={Sivakumar, Ssharvien Kumar and Frisch, Yannik and Ranem, Amin and Mukhopadhyay, Anirban},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--11},
  year={2025},
  publisher={Springer}
}