# NeuralPCI

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neuralpci-spatio-temporal-neural-field-for-3d/3d-point-cloud-interpolation-on-dhb-dataset)](https://paperswithcode.com/sota/3d-point-cloud-interpolation-on-dhb-dataset?p=neuralpci-spatio-temporal-neural-field-for-3d)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neuralpci-spatio-temporal-neural-field-for-3d/3d-point-cloud-interpolation-on-nl-drive)](https://paperswithcode.com/sota/3d-point-cloud-interpolation-on-nl-drive?p=neuralpci-spatio-temporal-neural-field-for-3d)

**NeuralPCI: Spatio-temporal Neural Field for 3D Point Cloud Multi-frame Non-linear Interpolation**   
[Zehan Zheng](https://dyfcalid.github.io/)\*, Danni Wu\*, Ruisi Lu, [Fan Lu](https://fanlu97.github.io/), [Guang Chen](https://ispc-group.github.io/)†, Changjun Jiang.   
(\* Equal contribution, † Corresponding author)  
**CVPR 2023**  

**[[Paper (arXiv)]](https://arxiv.org/abs/2303.15126) | [[Paper (CVPR)]](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_NeuralPCI_Spatio-Temporal_Neural_Field_for_3D_Point_Cloud_Multi-Frame_Non-Linear_CVPR_2023_paper.html) | [[Project Page]](https://dyfcalid.github.io/NeuralPCI)**  | [Video] | [Talk] | [Slides] | [Poster]    


| Indoor Scenario  | Outdoor Scenario |
| ------------- | ------------- |
| <video src="https://user-images.githubusercontent.com/51731102/228475246-e0f2d3c8-adad-41d5-a474-c05a2945cb20.mp4">  | <video src="https://user-images.githubusercontent.com/51731102/228474998-37c81904-061b-4b35-b70e-465c94a93ed8.mp4">|


<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#changelog">Changelog</a>
    </li>
    <li>
      <a href="#get-started">Get Started</a>
    </li>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#benchmark">Benchmark</a>
    </li>
    <li>
      <a href="#visualization">Visualization</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
    </li>
    <li>
      <a href="#license">License</a>
    </li>
  </ol>
</details>


## Changelog  
2023-5-21: We release the code of NeuralPCI along with the NL-Drive dataset.  
2023-3-27: We post the preprint paper on arXiv and release the project page.  
2023-2-28: This paper is accepted by **CVPR 2023** 🎉🎉.  

## Introduction

This repository is the PyTorch implementation for **NeuralPCI**.  

NeuralPCI is an end-to-end 4D spatio-temporal Neural field for 3D Point Cloud Interpolation, which implicitly integrates multi-frame information to handle nonlinear large motions for both indoor and outdoor scenarios.

<img src="img\overview.png" width=85%>

The 4D neural field is constructed by encoding the spatio-temporal coordinates of the multi-frame input point clouds via a coordinate-based multi-layer perceptron network. For each point cloud frame of the input, the interpolation time is set to the corresponding timestamps of four input frames for NeuralPCI to generate the corresponding point cloud. And then the neural field is optimized on runtime in a self-supervised manner without relying on ground truth. In the inference stage after optimization, NeuralPCI receives a reference point cloud and an arbitrary interpolation frame moment as input to generate the point cloud of the associated spatio-temporal location.



## Get Started
Please follow installation instructions to setup the environment.  
```
git clone https://github.com/ispc-lab/NeuralPCI.git
cd ./NeuralPCI/

conda create -n npci python=3.9
conda activate npci

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y

conda install -c open3d-admin open3d==0.9.0 -y
conda install -c conda-forge -c fvcore -c iopath fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y

conda install pyg -c pyg -y
conda install -c conda-forge shapely -y

# compile cuda CD&EMD
cd ./utils/CD/chamfer3D/
python setup.py install

cd ../../EMD/
python setup.py install
cp build/lib.linux-x86_64-cpython-39/emd_cuda.cpython-39-x86_64-linux-gnu.so .

# compile point cloud utils
cd ../point_cloud_query/
python setup.py install

cd ../pointnet2/
python setup.py install
```

## Dataset

### NL-Drive
<img src="img/NL_Drive.png" width=50%>  

Download link: [[OneDrive] ](https://tongjieducn-my.sharepoint.com/:f:/g/personal/zhengzehan_tongji_edu_cn/Ej4AiwgJWp1MsAFwtWcxIFkBPDwsCW_3bWSRlpYf4XZw-w) 

NL-Drive dataset is a challenging multi-frame interpolation dataset for autonomous driving scenarios. Based on the principle of hard-sample selection and the diversity of scenarios, NL-Drive dataset contains point cloud sequences with large nonlinear movements from three public large-scale autonomous driving datasets: KITTI, Argoverse and Nuscenes.  



### DHB
<img src="img/DHB.png" width=50%>  

Download link: [[Google Drive] ](https://drive.google.com/drive/folders/1Oaras1mV6DOICMPkCggPZvnBAtc4SKgH?usp=sharing) 

Dynamic Human Bodies dataset (DHB), containing 10 point cloud sequences from the MITAMA dataset and 4 from the 8IVFB dataset. The sequences in DHB record 3D human motions with large and non-rigid deformation in real world. The overall dataset contains more than 3000 point cloud frames. And each frame has 1024 points.  



## Benchmark  

  

## Visualization

<img src="img/DHB_vis.png" width=80%>

<img src="img/NL_Drive_vis.png" width=80%>

## Citation

If you find our code or paper useful, please cite
```bibtex
@inproceedings{zheng2023neuralpci,
  title     = {NeuralPCI: Spatio-temporal Neural Field for 3D Point Cloud Multi-frame Non-linear Interpolation},
  author    = {Zheng, Zehan and Wu, Danni and Lu, Ruisi and Lu, Fan and Chen, Guang and Jiang, Changjun},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
  }
```

  

## Acknowledgements

  

## License

  All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
