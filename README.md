# FTC-LSTM: Fourier Transform Convluitonal Long Short Term Memory Neural Network

by Siyu Chen, Lin Deng and Jun Zhao.

[[Paper page](https://ieeexplore.ieee.org/document/10509652)] [[Github](https://github.com/siyuChen540/inpainting_nn/edit/v2.0)]

# Abstract
Chlorophyll a (Chl a) concentration, a vital indicator of water quality and crucial for assessing the health of marine ecosystems, presents significant challenges for satellite remote sensing due to various interferences, such as cloud cover, sun-glints, and adjacency effects. These impediments limit our understanding of marine ecosystems and hinder sustainable management practices. This study proposes a novel approach to overcome these challenges: the Fourier transform convolutional long short-term memory (FTC-LSTM) framework. Integrating Fourier transform convolution (FTC) and long short-term memory (LSTM) layers, the FTC-LSTM model aims to estimate cloud-free Chl a, improving the accuracy and robustness of the inpainting process. Evaluation of the FTC-LSTM model across the South China Sea (SCS), along with two other state-of-the-art deep learning models [data-interpolating convolutional auto-encoder (DINCAE) and convolutional long short-term memory neural network (Conv-LSTM)], reveals its consistently superior performance across regions with distinct characteristics. Notably, the FTC-LSTM model achieved the highest scores with impressive values: 0.95 for determination coefficient ( R2 ), 47.57 for peak signal-to-noise ratio (PSNR), 0.99 for structural similarity index measure (SSIM), and 0.01 for root mean square error (RMSE). The temporal analysis demonstrates the model’s ability to accurately capture the temporal variability of Chl a in the SCS. Furthermore, a comparison of spatial patterns indicates that the FTC-LSTM model excels in reliably reconstructing Chl a distributions within the SCS, outperforming other models, particularly in tropical and subtropical regions significantly impacted by clouds.

# Module Architecture
<p align="center">
  <img src="https://github.com/siyuChen540/inpainting_nn/blob/v2.0/assert/dataprocess-block-layer-network2.png" />
</p>

# Environment setup
  1. Conda
    ```shell
    % Install conda for Linux, for other OS download miniconda at https://docs.conda.io/en/latest/miniconda.html
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    $HOME/miniconda/bin/conda init bash

    conda create -n ftc-lstm-env python=3.11
    conda activate ftc-lstm-env
    pip install -r requirements.txt
    ```

  2. Docker
    ```shell
    docker build -t ftc-lstm:latest.
    docker run -it --rm -v $(pwd):/workspace ftc-lstm:latest
    ```

# Interface <a name="interface"></a>
  Run
    ```shell
    python main.py
    ```

# Citation
  ```BibTeX
  @ARTICLE{10509652,
    author={Chen, Siyu and Deng, Lin and Zhao, Jun},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={Enhanced Reconstruction of Satellite-Derived Monthly Chlorophyll a Concentration With Fourier Transform Convolutional-LSTM}, 
    year={2024},
    volume={62},
    number={},
    pages={1-14},
    keywords={Image reconstruction;Remote sensing;Clouds;Biological system modeling;Long short term memory;Data models;Convolution;Deep learning;Fourier transform convolutional long short-term memory (FTC-LSTM);gap-filling;remote sensing;South China Sea (SCS)},
    doi={10.1109/TGRS.2024.3394399}}
  ```