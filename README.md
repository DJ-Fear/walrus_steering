# Walrus Steering: Interpretability and Activation Steering for Physics Models

This repository is a fork of the main Walrus physics model which is designed to enable interpretability work, especially activation steering on Walrus.

Within you will find tools firstly for generating "delta" concept vectors (activations vectors representing specific physical features learned by the model), and secondly for injecting concept vectors back into the model during inference for steering model predictions.



## Installation

1. **Clone and Setup Environment**
   
   Ensure you have a Python 3.10+ environment.
   ```bash
   # Clone the repository
   git clone <repo_url> walrus_steering
   cd walrus_steering
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   
   Install the required packages, including `the_well`.
   ```bash
   # Install dependencies for walrus_steering
   pip install -e .

   # Clone the_well into the base walrus_steering directory
   git clone <repo_url> the_well

   # Install dependencies for the_well
   pip install -e ./the_well
   ```

3. **Point to Model Checkpoint**
   
   Modify the checkpoint path in `start.py`:
   ```python
   "+checkpoint.load_checkpoint_path='path/to/your/checkpoint/'"
   ```

## Workflow Overview

The steering workflow consists of three main stages:

1. **Tensor Creation**: Generate the steering concept vector.
2. **Configuration**: Select the target simulation files, steering strength ($\alpha$), and steering injection method.
3. **Execution**: Run the rollout generation to produce videos.

### 1. Saving Raw Feature/Control Activations Tensors

First use start.py to generate rollout videos for a list of files which represent a system in conditions which display the physical feature you are interested in, and also the same system in another regime/alternative contitions which do not display the feature (e.g., Shear Flow system displaying Vortex flow and displaying Laminar flow.). Start.py will run call train.py to run the model over the files and generate a rollout video for each file.

You must ensure that (1) no steering is being applied by commenting out the lower config options, (2) layers_to_hook is set to the model layer which you wish save, and (3) save_activations is set to True.

```python
# Add the file(s) you want to process to the run_list (when steering is inactive the first 3 elements are unused).
run_list = [
    ["null", "null", "null", "shear_flow", "shear_flow_Reynolds_1e5_Schmidt_2e-1"],
]
```

```python
# Ensure configs are set up to save activations and not perform steering.
    # ... 
    "save_activations=True", # Save activations
    "layers_to_hook=['blocks.39']", # Example layer
    ]) # Ensure configs below this commented out
    # ...
```
This will generate .pickle files in the experiments/activations/ directory, containing the raw activation data for the specified layers and files.

### 2. Creating Steering Vectors

Use `tensor_creator.py` to create the concept vectors used for steering.

This script calculates the delta between two sets of activations – typically a "feature" set (e.g., Vortex) and a "control" set (e.g., Laminar flow).


1. Open `tensor_creator.py`.
2. Define your list of one or more`feature_files` and `control_files` (these will be the files which you saved in the previous step).
3. Run the script which will generate a single `.pickle` file in `experiments/activations/`.

The output pickle file contains the concept vector that will be injected during steering.

### 3. Configuring the Experiment

The main model and steering configuration happens in `start.py`. This file controls which dataset is loaded, which file is used as the initial condition, and how the steering is applied.

The behavior of the steering is highly sensitive to the configuration chosen in `start.py`.

Key settings to modify in `main()`:

*   **`run_list`**: A list of experiments to run sequentially.
    ```python
    # Format: [injection_strength, inject_sign, inject_type, dataset, filename]
    run_list = [
        [0.4, "pos", "pad", "shear_flow", "shear_flow_Reynolds_1e5_Schmidt_2e-1"],
        [0.1, "neg", "drop", "active_matter", "active_matter_L_10.0_zeta_5.0_alpha_-1.0"],
    ]
    ```
*   **`layers_to_hook`**: Used to specify one or more layers (e.g. `blocks.39`). This dictates both the layer(s) from which activations will be saved and the layer(s) at which the steering intervention will take place.

*   **`inject_tensor_path`**: The path to the concept vector.

*   **`inject_spatial_type`**: Used to specify how spatial the spatial dimensions of the overall steering tensor are to be handled. See below for details.

*   **`inject_sign`**: Sets the sign of the steering intervention, can be either `pos` (positive) or `neg` (negative).

*   **`inject_strength`**: This defines the stength of the steering intervention (the steering coeficient alpha in the paper). When too high (> 1.0) this often leads to numerical instability.

*   **`save_activations`**: For extracting raw activations from the layer(s) specified in layers_to_hook, if a file name is provided this will enable activations saving.

*   **`save_raw_predictions`**: Save raw model outputs to experiments/raw_predictions/ in .npy format.

*   **`short_validation_only`**: Ensure that unnecessary validation steps are not performed (you will probably always want this set to true).

*   **`image_validation`**: Save validation images in experiments/visuals (e.g. power spectra plots)

*   **`video_validation`**: Save rollout videos in experiments/visuals.

#### Steering Method (`inject_spatial_type`)
When saving the original feature and control activations they have a [Time, Batch, Channel, Height, Width, Depth] format. Of these, the delta calculation preserves the [C, H, W, D] dimensions. The size of C is always constant, but the spatial H, W and D dimensions can vary in size depending on the dataset/domain. The steering method therefore tells the model how to handle the spatial dimensions of the full steering tensor. There are four options:

*   **`none`**: No modification of spatial dimensions.
*   **`drop`**: Average the channel dimensions over the spatial dimensions in order to entirely drop the spatial dimensions. This type of steering tends to produce results which appear more "natural" and less "forced".
*   **`pad`**: Keeps the spatial dimensions. If there is any disparity in dimension size between the steering tensor and the target tensor, the steering tensor will be padded with zeros (where too large) or cropped (where too small).
*   **`interpol`**: Keeps the spatial dimensions. If there is any disparity in dimension size, the steering tensor will be filled with trilinear interpolation (where too large) or cropped (where too small).

### Running with SLURM

If you are using a SLURM cluster, you may wish to use the provided run script.

```bash
./run.sh
```

### Files Required:
*   Checkpoints:
    * `miles_rio_checkpoint_last/` (new)
    * `.../` (old)
*   Steering Tensor: [`experiments/activations/newTensor:(18vortex_group)-(10laminar_group)[+].pickle`]
*   Dataset: The Well's `shear_flow` dataset.


## General Notes

Steering results depend heavily on configuration settings, the feature/concept files used to create the concept vector and also the target datafile.

#### Regime Distance

Steering works best when the target datafile is in a regime which is not too far from the regime or physical feature which it is being steered towards. If a shear flow datafile is deep within the laminar regime then it will be difficult to steer it towards the vortex regime.


#### Checkpoints

We have observed that using older checkpoints often yield *more* striking steering results, especially for vortex steering. We suspect this is because the learned physics in earlier checkpoints is less strict, and therefore more permissible of steering which leads to unphysical results.


## Example Results: Vortex Steering

We have provided a concept vector to get you started. This is called `newTensor:(18vortex_group)-(10laminar_group)[+].pickle` and was created using all the shear flow datafiles (18 in the vortex regime, and 10 in the laminar regime).

To illustrate the variety of steering results which can be achieved with different concept vectors, target datafiles, and steering settings we include 6 examples below.

#### 1. Dropped Spatial Dimensions ($\alpha=–0.5$)
**Datafile:** `shear_flow_Reynolds_1e5_Schmidt_2e-1`
**Concept:** `(18vortex_group)-(10laminar_group)`
**Injection Type:** Drop | **Sign:** Negative | **Strength:** 0.5

[![Negative Drop Example](assets/shear_flow_Reynolds_1e5_Schmidt_2e-1[–drop@0.5][(18vortex_group)-(10laminar_group)].png)](assets/shear_flow_Reynolds_1e5_Schmidt_2e-1[–drop@0.5][(18vortex_group)-(10laminar_group)].png)

#### 2. Padded Spatial Dimensions ($\alpha=–0.4$)
**Datafile:** `shear_flow_Reynolds_1e5_Schmidt_2e-1`
**Concept:** `(18vortex_group)-(10laminar_group)`
**Injection Type:** Pad | **Sign:** Negative | **Strength:** 0.4

[![Negative Pad Example](assets/shear_flow_Reynolds_1e5_Schmidt_2e-1[–pad@0.4][(18vortex_group)-(10laminar_group)].png)](assets/shear_flow_Reynolds_1e5_Schmidt_2e-1[–pad@0.4][(18vortex_group)-(10laminar_group)].png)

#### 3. Dropped Spatial Dimensions  ($\alpha=+0.7$)
**Datafile:** `shear_flow_Reynolds_1e5_Schmidt_2e-1`
**Concept:** `(18vortex_group)-(10laminar_group)`
**Injection Type:** Drop | **Sign:** Positive | **Strength:** 0.7

[![Positive Drop Example](assets/shear_flow_Reynolds_1e5_Schmidt_2e-1[+drop@0.7][(18vortex_group)-(10laminar_group)].png)](assets/shear_flow_Reynolds_1e5_Schmidt_2e-1[+drop@0.7][(18vortex_group)-(10laminar_group)].png)

#### 4. Padded Spatial Dimensions ($\alpha=+0.5$)
**Datafile:** `shear_flow_Reynolds_1e5_Schmidt_2e-1`
**Concept:** `(18vortex_group)-(10laminar_group)`
**Injection Type:** Pad | **Sign:** Positive | **Strength:** 0.5

[![Positive Pad Example](assets/shear_flow_Reynolds_1e5_Schmidt_2e-1[+pad@0.5][(18vortex_group)-(10laminar_group)].png)](assets/shear_flow_Reynolds_1e5_Schmidt_2e-1[+pad@0.5][(18vortex_group)-(10laminar_group)].png)

#### 5. Dropped Spatial Dimensions ($\alpha=+0.6$)
**Datafile:** `shear_flow_Reynolds_1e5_Schmidt_2e-1`
**Concept:** `(2vortex_group)-(2laminar_group)`
**Injection Type:** Drop | **Sign:** Positive | **Strength:** 0.6

[![Positive Drop Example](assets/shear_flow_Reynolds_1e5_Schmidt_2e-1[+drop@0.6][(2vortex_group)-(2laminar_group)].png)](assets/shear_flow_Reynolds_1e5_Schmidt_2e-1[+drop@0.6][(2vortex_group)-(2laminar_group)].png)

#### 6. Padded Spatial Dimensions ($\alpha=+0.4$)
**Datafile:** `shear_flow_Reynolds_1e5_Schmidt_5e0`
**Concept:** `(1vortex_group)-(1laminar_group)`
**Injection Type:** Pad | **Sign:** Positive | **Strength:** 0.4

[![Positive Pad Example](assets/shear_flow_Reynolds_1e5_Schmidt_5e0[+pad@0.4][(1vortex_group)-(1laminar_group)].png)](assets/shear_flow_Reynolds_1e5_Schmidt_5e0[+pad@0.4][(1vortex_group)-(1laminar_group)].png)



