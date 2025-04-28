# **A Skeleton-based Geometric Deep Neural Network for Alzheimer’s Disease Mice Behavioral Analysis**


## Abstract
<div style="text-align: justify"> 
Alzheimer’s disease (AD) is a progressive and irreversible brain disorder that remains incurable. Research has shown a strong link between gait and cognition, with AD significantly affecting gait and behavioral patterns. While most preclinical studies analyze gait using mice pawprints, this approach is prone to inconsistencies due to variations in pawprints' correction by different experimenters. In contrast, skeleton-based behavioral analysis provides a more consistent and cleaner representation. In this work, we collect a new mice dataset and propose a novel skeleton-based geometric deep attention network for disease classification using the mice's behavioral information from this dataset. We begin our analysis by extracting posture data as skeleton landmark sequences which are then processed by our proposed network for the classification. Our proposed approach demonstrates promising results, making it particularly relevant for preclinical gait research and we conduct an ablation study on our proposed approach to demonstrate its effectiveness.
</div>

## Sample Video and Pose Tracking 
We use the the DeepLabCut (DLC) animal pose tracking toolbox to extract the mice posture from our video dataset. For all instructions on installation and usage of the DLC toolboox, visit their Github page [Here](https://github.com/DeepLabCut/DeepLabCut/tree/main)

<img src="samples/combine.gif" style="width:100%; height:auto;">

## Packages and Dependencies
- For packages and dependencies, first create an enviroment using Python, activate the enviroment and run "pip install -r requirements.txt"

## Data Preparation 
- In the 'data_preparation' folder contains the files for preparing the pose skeleton files before feeding them to the geometric deep framework. Our preparation scripts follow this order: <br>  <br>
'create_project.m'  ---> 'extract_video_frames.m' ---> 'get_arena_coordinates.m' ---> 'get_obj_1_coordinates.m'  ---> 'get_obj_2_coordinates.m' ---> 'normalize_data.m' ---> 'normalize_obj_1_coord.m' ---> 'normalize_obj_2_coord.m' ---> 'extract_skeleton_within.m' ---> 'extract_skeleton_beyond.m' ---> 'extract_skeleton_all.m'
 
## Data Loading 
- In the 'data_loader' folder contains the files for loading as binary or triple classification tasks.

## Training
- For the network training, use the file 'train_LOOCV_b' or 'train_LOOCV_m' for binary or multi-class respectively. For the binary classification, run the 'train_LOOCV_b.py' file in 'geometric_models/with_attention' for attention-based or in the 'geometric_models/without_attention' for non-attention based. Similar procedure applies for the multi-class classification task (this time, the 'train_LOOCV_m.py' file in the corresponding folder)

## Angular velocity
- Use the 'compute_ang_velocity' file to calculate the angular velocities

## How to Cite
- If any part of this work has been useful to you, do not forget to cite. An example bibtex citation is ...

## License
This code and models are available for non-commercial scientific research purposes as defined in the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.

