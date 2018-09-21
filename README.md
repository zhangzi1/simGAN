# simGAN
A simple practice of CVPR 2017 best paper [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/abs/1612.07828).

source of the dataset could be found on [UnityEyes](http://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) and [MPIIGaze](http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz).

Here is how the dictory should look like:

    root
    ├── buffer
    ├── data
    │   ├── real
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   └── ...
    │   └── syn
    │       ├── 0.png
    │       ├── 1.png
    │       ├── 2.png
    │       └── ...
    ├── graphs
    ├── logs
    ├── samples
    ├── xxx.py
    ...
