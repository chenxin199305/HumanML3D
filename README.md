# HumanML3D: 3D Human Motion-Language Dataset

HumanML3D is a 3D human motion-language dataset that originates from a combination of
[HumanAct12](https://github.com/EricGuo5513/action-to-motion) and
[Amass](https://github.com/EricGuo5513/action-to-motion) dataset.

It covers a broad range of human
actions such as daily activities (e.g., 'walking', 'jumping'),
sports (e.g., 'swimming', 'playing golf'),
acrobatics (e.g., 'cartwheel')
and artistry (e.g., 'dancing').

<div  align="center">    
  <img src="./document/dataset_showcase.png"  height = "500" alt="teaser_image" align=center />
</div>

## Motion Datasets

### AMASS Dataset

AMASS (Archive of Motion Capture as Surface Shapes) is a large-scale motion capture dataset that unifies a number of existing motion capture datasets into a common format. It provides a comprehensive collection of human motion data

Related content:

- https://github.com/nghorbani/amass

### KIT-ML Dataset

[KIT Motion-Language Dataset](https://motion-annotation.humanoids.kit.edu/dataset/) (KIT-ML) is also a related dataset that contains 3,911 motions and 6,278 descriptions. We processed KIT-ML dataset following the same procedures of HumanML3D dataset,
and provide the access in this repository. However, if you would like to use KIT-ML dataset, please remember to cite the original paper.

If this dataset is usefule in your projects, we will apprecite your star on this codebase. 😆😆

## Other Works on HumanML3D

:ok_woman: [T2M](https://ericguo5513.github.io/text-to-motion) - The first work on HumanML3D that learns to generate 3D motion from textual descriptions, with *temporal VAE*.  
:running: [TM2T](https://ericguo5513.github.io/TM2T) - Learns the mutual mapping between texts and motions through the discrete motion token.  
:dancer: [TM2D](https://garfield-kh.github.io/TM2D/) - Generates dance motions with text instruction.  
:honeybee: [MoMask](https://ericguo5513.github.io/momask/) - New-level text2motion generation using residual VQ and generative masked modeling.

## How to Obtain the Data

For KIT-ML dataset, you could directly download [[Here]](https://drive.google.com/drive/folders/1D3bf2G2o4Hv-Ale26YW18r1Wrh7oIAwK?usp=sharing). Due to the distribution policy of AMASS dataset, we are not allowed to distribute the data directly. We
provide a series of script that could reproduce our HumanML3D dataset from AMASS dataset.

You need to clone this repository and install the virtual environment.

<!-- ### [2021/01/12] Updates: add evaluation related files & scripts   -->

**[2022/12/15] Update**: Installing matplotlib=3.3.4 could prevent small deviation of the generated data from reference data. See [Issue](https://github.com/EricGuo5513/HumanML3D/issues/21#issue-1498109924)

### Python Virtual Environment

Conda environment is needed at first, then run the following command to install required packages. Lastly, you need to run the scripts in order to obtain HumanML3D dataset.

```sh
# Create conda environment
conda create -n humanml3d python=3.7.10 -y
conda activate humanml3d

# Install required packages
bash install_env.sh

# Run script to generate HumanML3D dataset
python pose_data_generator.py
```

> [!NOTE]
>
> This repository is refined by the author, may be slightly different from the original one.
> If you want to look into the original codebase, please visit [here](https://github.com/EricGuo5513/HumanML3D)