# Knee-Lateral-Xray-Bone-Segmentation-Model-Program
A program containing steps to fine-tune a YOLO segmentation model, and applying said model to generate a mask image for each bone in the knee lateral X-ray image.

This program is a direct continuation of the Knee Region of Interest Detection project.

Knee Detection Project: https://github.com/Lu-Yik-Ho/Knee-Lateral-Xray-RoI-Detection-Model-Program

# Original Image:
<img width="636" height="636" alt="103854_C" src="https://github.com/user-attachments/assets/7d6a0eca-a3f9-4601-b963-3efd9fa4fdff" />

# Masks Produced:

Patella:

<img width="636" height="636" alt="103854_C_Patella_PMask" src="https://github.com/user-attachments/assets/df39ff5f-3e37-4798-921b-722d57188ac1" />

Femur:

<img width="636" height="636" alt="103854_C_Femur_PMask" src="https://github.com/user-attachments/assets/a6170a16-bedc-436d-ac10-a990a0af457d" />

Tibia:

<img width="636" height="636" alt="103854_C_Tibia_PMask" src="https://github.com/user-attachments/assets/ef53de16-5df8-4fa9-8fab-3f4e4194ec3d" />

# Files:
BoneSegmentationLateral_v1.1_main.ipynb: Main Jupyter Notebook Program.

BoneSegmentationLateral_TrainYOLOModel.py: Python file containing functions used by the main program.

BoneSegmentationLateral_ImageProcessing.py: Python file containing functions used by the main program.

BoneSegmentationLateral_v1.0_User Manual.docx: Word document containing a step-by-step guide in using the main program.

SquareCroppingModel.pt: A model fine-tuned to crop the region of interest from a knee lateral X-ray image.

SampleTrainedModel.pt: A model fine-tuned by me with this program.
