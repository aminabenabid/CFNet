## CFNet: Cross-scale fusion network for medical image segmentation 

## Accepted at Journal of King Saud University - Computer and Information Sciences [Link](https://www.sciencedirect.com/science/article/pii/S131915782400212X)

### Abstract
Learning multi-scale feature representations is essential for medical image segmentation. Most existing frameworks
are based on U-shape architecture in which the high-resolution representation is recovered progressively by connecting
different levels of the decoder with the low-resolution representation from the encoder. However, intrinsic defects
in complementary feature fusion inhibit the U-shape from aggregating efficient global and discriminative features along
object boundaries. While Transformer can help model the global features, their computation complexity limits the application
in real-time medical scenarios. To address these issues, we propose a Cross-scale Fusion Network (CFNet), combining a 
cross-scale attention module and pyramidal module to fuse multi-stage/global context information. Specifically, we first 
utilize large kernel convolution to design the basic building block capable of extracting global and local information. 
Then, we propose a Bidirectional Atrous Spatial Pyramid Pooling (BiASPP), which employs atrous convolution in the bidirectional 
paths to capture various shapes and sizes of brain tumors. Furthermore, we introduce a cross-stage attention mechanism to reduce
redundant information when merging features from two stages with different semantics. Extensive evaluation was performed on five
medical image segmentation datasets: a 3D volumetric dataset, namely Brats benchmarks. CFNet-L achieves 85.74% and 90.98% dice score
for Enhanced Tumor and Whole Tumor on Brats2018, respectively. Furthermore, our largest model CFNet-L outperformed other
methods on 2D medical image. It achieved 71.95%, 82.79%, and 80.79% SE for STARE, DRIVE, and CHASEDB1, respectively.
### Requirements  
The main package and version of the python environment are as follows
```
pytorch | torchvision | matplotlib | opencv | pandas 
```  
The above environment is successful when running the code of the project. In addition, it is well known that pytorch has very good compatibility (version>=1.0). Thus, __I suggest you try to use the existing pytorch environment firstly.__  
    
The current version has problems reading the `.tif` format image in the DRIVE dataset on Windows OS. __It is recommended that you use Linux for training and testing__

---  
## Usage 
### 0) Download Project 

Running```git clone https://github.com/aminabenabid/CFNet.git```  
The project structure and intention are as follows : 
```
VesselSeg-Pytorch			# Source code		
    ├── config.py		 	# Configuration information
    ├── lib			            # Function library
    │   ├── common.py
    │   ├── dataset.py		        # Dataset class to load training data
    │   ├── datasetV2.py		        # Dataset class to load training data with lower memory
    │   ├── extract_patches.py		# Extract training and test samples
    │   ├── help_functions.py		# 
    │   ├── __init__.py
    │   ├── logger.py 		        # To create log
    │   ├── losses
    │   ├── metrics.py		        # Evaluation metrics
    │   └── pre_processing.py		# Data preprocessing
    ├── models		        # All models are created in this folder
    │   ├── denseunet.py
    │   ├── __init__.py
    │   ├── LadderNet.py
    │   ├── nn
    │   └── UNetFamily.py
    ├── prepare_dataset	        # Prepare the dataset (organize the image path of the dataset)
    │   ├── chasedb1.py
    │   ├── data_path_list		  # image path of dataset
    │   ├── drive.py
    │   └── stare.py
    ├── tools			     # some tools
    │   ├── ablation_plot.py
    │   ├── ablation_plot_with_detail.py
    │   ├── merge_k-flod_plot.py
    │   └── visualization
    ├── function.py			        # Creating dataloader, training and validation functions 
    ├── test.py			            # Test file
    └── train.py			          # Train file
```

### Training model
```
CUDA_VISIBLE_DEVICES=1 python train.py --save UNet_vessel_seg --batch_size 64
```
### Testing model
```
CUDA_VISIBLE_DEVICES=1 python test.py --save UNet_vessel_seg  
```  

## Acknowledgement
* Our main code datasets preprocessing is modified based on [VesselSeg-Pytorch](https://github.com/lee-zq/VesselSeg-Pytorch).
