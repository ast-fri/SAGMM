# Self-Adaptive Graph Mixture of Models (SAGMM)

## Table of Contents
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Requirements

- **CUDA**: 11.8
- **GCC**: 10.5.0
- **Python**: 3.10.18
- **Pytorch**: 2.1.2+cu118
- Complete dependency/packages list available in `environment.yml`

## Getting Started

### Step 1: Create Conda Environment
```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate the environment
conda activate sagmm
```
- The first line of the `yml` file sets the new environment's name (which is sagmm)

### Step 2: Verify Installation
```bash
python --version
nvcc --version
python -c "import torch; print(torch.__version__)"
```
- Use below commands to install pytorch:
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-sparse==0.6.18+pt21cu118 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-scatter==2.1.2+pt21cu118 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch_geometric==2.5.3
```
## Running the Project

### Step 1: Download Dataset
- [OGB Datasets](https://arxiv.org/pdf/2005.00687)
- [Deezer](https://arxiv.org/abs/2005.07959)
- [Yelpchi](https://ojs.aaai.org/index.php/ICWSM/article/view/14389) 
- [Pokec](https://snap.stanford.edu/data/soc-Pokec.html)
- The dataset will be downloaded in path given by `--data_dir` argument
### Step 2: Run the Script
In this section, we describe how to train the models for various experimental settings.
#### Node classification
```bash
cd node_classification/shell_scripts/
./run_<dataset>.sh
```
#### Link prediction
```bash
cd link_prediction/shell_scripts/
./run_<dataset>.sh
```

#### Graph classification/regression
```bash
cd graph_classification_regression
./graph_run.sh
```
For inference run `inference.py` or `inference_mini_batch.py` based on dataset with same set of parameters used for training
## License
This project is licensed under the CC-BY-NC License - see LICENSE file for complete details.

## Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@article{SAGMM-2025,
  title={Self-Adaptive Graph Mixture of Models},
  author={[Authors]},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

For questions or collaborations, please reach out through GitHub issues or contact the research team.

---

© 2025 Research Project | Self-Adaptive Graph Mixture of Models
