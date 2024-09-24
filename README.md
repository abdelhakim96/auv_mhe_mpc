# Parametric MHE

This repository contains the code for the numerical example in the paper [MHE under parametric uncertainty - Robust state estimation without informative data](https://arxiv.org/abs/2312.14049). The implementation is based on Casadi and the solver ipopt.
The implementation was tested using Python3.10.12 with packages as included in the requirements.txt file.

## Reference

[Simon Muntwiler, Johannes Köhler, and Melanie N. Zeilinger. MHE under parametric uncertainty - Robust state estimation without informative data. arXiv preprint arXiv:2312.14049, 2023.](https://arxiv.org/abs/2312.14049)

## Abstract

In this paper, we study joint state and parameter estimation for general nonlinear systems with uncertain parameters and persistent process and measurement noise. In particular, we are interested in stability properties of the resulting state estimate in the absence of persistency of excitation (PE). With a simple academic example, we show that existing moving horizon estimation (MHE) approaches for joint state and parameter estimation as well as classical adaptive observers can result in diverging state estimates in the absence of PE, even if the noise is small. We propose a novel MHE formulation involving a regularization based on a constant prior estimate of the unknown system parameters. Only assuming the existence of a stable state estimator, we prove that the proposed MHE results in practically robustly stable state estimates irrespective of PE. We discuss the relation of the proposed MHE formulation to state-of-the-art results from MHE, adaptive estimation, and functional estimation. The properties of the proposed MHE approach are illustrated with a numerical example of a car with unknown tire friction parameters. 

# Setup Instructions
* Clone the repository:

        git clone git@gitlab.ethz.ch:ics/parametric-mhe.git

* Setup a virtual environment with requirements as stated in requirements.txt:

        python3.10 -m venv env_name
        source env_name/bin/activate
        cd parametric-mhe
        pip install -r requirements.txt
        
* Open the ipython notebook:

        jupyter notebook parametric-mhe.ipynb

