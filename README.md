# 3DLaneNAS: Neural Architecture Search for Accurate and Light-Weight 3D Lane Detection
PyTorch implementation of [3DLaneNAS](https://link.springer.com/chapter/10.1007/978-3-031-15919-0_34): an evolutionary NAS approach for Accurate and Light-Weight 3D Lane Detection.

## Paper Abstract
Lane detection is one of the most fundamental tasks for autonomous driving. It plays a crucial role in the lateral control and the precise localization of autonomous vehicles. Monocular 3D lane detection methods provide state-of-the-art results for estimating the position of lanes in 3D world coordinates using only the information obtained from the front-view camera. Recent advances in Neural Architecture Search (NAS) facilitate automated optimization of various computer vision tasks. NAS can automatically optimize monocular 3D lane detection methods to enhance the extraction and combination of visual features, consequently reducing computation loads and increasing accuracy. This paper proposes 3DLaneNAS, a multi-objective method that enhances the accuracy of monocular 3D lane detection for both short- and long-distance scenarios while at the same time providing a fair amount of hardware acceleration. 3DLaneNAS utilizes a new multi-objective energy function to optimize the architecture of feature extraction and feature fusion modules simultaneously. Moreover, a transfer learning mechanism is used to improve the convergence of the search process. Experimental results reveal that 3DLaneNAS yields a minimum of 5.2% higher accuracy and ≈1.33× lower latency over competing methods on the synthetic-3D-lanes dataset.

## Get started
1. Clone the repository
    ```
    git clone https://github.com/HERO-MDH/3DLaneNAS.git
    ```
    We call this directory as `$RESA_ROOT`

2. Create an environment and activate it (We've used conda. but it is optional)

    ```Shell
    conda create -n lanenas python=3.9 -y
    conda activate lanenas
    ```

3. Install dependencies

    ```Shell
    # Install pytorch firstly, the cudatoolkit version should be same in your system. (you can also use pip to install pytorch and torchvision)
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
      
    # Install kornia and einops
    pip install kornia
    pip install einops

    # Install other dependencies
    pip install -r requirements.txt
    ```
## Dataset
Our method was tested using the 3D Lane Synthetic Dataset. For additional details about the database and required setups, please refer to the following link: [https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset]
## How to run 3DLaneNAS
3DLaneNAS employs three distinct search methods to find the optimal architecture: Simulated Annealing, Random Search, and Local Search. To use the platform with each of these search methods, follow the commands provided below.
```Shell
    python <SimulatedAnnealing.py, Random_Search.py, or local_search.py>
```
