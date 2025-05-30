# SPDS800---Speciale-i-Data-Science

MS-pose/
├── eda/
├── videos/
├── evaluation/
├── runs/
├── wrappers/
│ ├── config_notebooks
│ │ ├── DeepLabCut
│ │ ├── Sleap
│ │ └── LightningPose
│ ├── wrapper_dlc.py
│ ├── wrapper_lightning.py
│ ├── wrapper_main.py
│ └── wrapper_sleap.py
└── README.md

**`videos/`**  
Contains the training videos and provides insight into what the training data looks like.

**`eda/`**  
Contains a notebook with the initial exploratory data analysis (EDA) conducted at the beginning of the project.

**`evaluation/`**  
Contains all files used for model evaluation and the calculations involved.

**`runs/`**  
Contains the data extracted from UCloud after training and inference runs for each model.

**`wrappers/`**  
The `wrappers` folder contains adapter scripts used for each pose estimation toolbox.  
For each toolbox, there is a corresponding configuration notebook located in `config_notebooks/`, which includes setup instructions and workflow examples:
- `DeepLabCut/`
- `Sleap/`
- `LightningPose/`

