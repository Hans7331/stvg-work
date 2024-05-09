# Size-Modulated Deformable Attention in Spatio-Temporal Video Grounding Pipelines

This repository contains the code implementation for the paper titled "Size-Modulated Deformable Attention in Spatio-Temporal Video Grounding Pipelines".

## Overview

The code in this repository is based on the work done in [the github](https://github.com/jy0205/STCAT) repository of [Embracing Consistency: A One-Stage Approach for Spatio-Temporal Video Grounding](https://arxiv.org/abs/2209.13306). Our work extends and modifies the concepts introduced in the STCAT paper to incorporate size-modulated deformable attention mechanisms.

## Usage

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Hans7331/stcat-code.git
   cd stcat-code

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

### Running the Code

To run the basic experiment implemented in this repository, use the following command:
   ```bash
   python train_net.py --config-file "experiments/HC-STVG/e2e_STCAT_R101_HCSTVG.yaml"
   python train_net.py --config-file "experiments/VidSTG/e2e_STCAT_R101_VidSTG.yaml"
   ```   
## Note

The code for other experiments is currently being updated and will be added to the repository soon. Stay tuned for updates!



   
