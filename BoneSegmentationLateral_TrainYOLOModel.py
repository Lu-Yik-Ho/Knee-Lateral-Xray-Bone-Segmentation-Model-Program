import torch
import pydicom # type: ignore
import os
import glob
import numpy as np
import json
import torch.utils.data
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import ultralytics # type: ignore
import cv2
os.environ['YOLO_VERBOSE'] = 'false'
from ultralytics import YOLO # type: ignore
from sklearn.model_selection import train_test_split
import yaml


def calculatePredictMask(result, label, mask_output_dir):
        
    """
    **Applies post-processing to a YOLO model predicted polygon mask, and returns updated information.**

    Post-processing includes removing small connections between areas and keeping the largest area.

    A folder (mask_output_dir) for storing image files while processing is required.

    Parameters:

        result (ultralytics.engine.results.Results): A model prediction result.

        label (str): The name of the label for this prediction.

        mask_output_dir (str): The directory for outputting prediction mask images.

    Returns:
    
        output (dict): A dictionary containing updated information of the prediction mask. See predict_masks below.
    """


    # Retrieve information for processing
    img_height, img_width = result.orig_shape
    file_name = os.path.splitext(result.path)[0].split(os.sep)[-1]
    mask_image_path = "{}_{}_PMask.png".format(os.path.join(mask_output_dir, file_name), label)

    # Create mask layer based on model predicted bounding points
    orig_mask = np.zeros(result.orig_shape)
    orig_contour = result.masks.xy[0].reshape(result.masks.xy[0].shape[0],1,2)
    orig_mask = cv2.drawContours(orig_mask, np.int32([orig_contour]), -1, (255, 255, 255), thickness= -1)

    # Processing to split connected areas
    orig_mask = cv2.erode(orig_mask, np.ones((10, 10), np.uint8)) 
    orig_mask = cv2.dilate(orig_mask, np.ones((10, 10), np.uint8)) 

    # Save and load to convert to OpenCV imread default format
    cv2.imwrite(mask_image_path, orig_mask)
    cv_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # Locate all contours
    _, thresh = cv2.threshold(cv_image, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Determine the contour with most bounding points
    contour_sizes = []

    for index, contour in enumerate(contours):
        #print("Contour {}:".format(index+1))
        #print(contour.shape[0])
        contour_sizes.append(contour.shape[0])

    #print("Largest Contour: Contour {}".format(contour_sizes.index(max(contour_sizes))+1))

    max_contour = contours[contour_sizes.index(max(contour_sizes))]

    # Calculate area of mask
    mask_area = cv2.contourArea(max_contour)

    #print("Area of Contour: {}".format(max_area))

    # Create new mask layer image based on largest contour
    new_mask = np.zeros(result.orig_shape)
    new_mask = cv2.drawContours(new_mask, [max_contour], -1, (255, 255, 255), thickness= -1)
    cv2.imwrite(mask_image_path, new_mask)

    # Return important values
    predict_mask = {
        "path": result.path,
        "label": label,
        "points": max_contour.reshape(max_contour.shape[0], 2),
        "mask_data": new_mask,
        "mask_path": mask_image_path,
        "mask_area": mask_area,
        "img_width": img_width, # Width of image
        "img_height": img_height, # Height of image
        
    }

    return predict_mask


def calculateTruthMask(result, json_dir, label_num, label, mask_output_dir):
        
    """
    **Collects information for the file storing the ground truth and returns compacted information.**

    Post-processing includes removing small connections between areas and keeping the largest area.

    A folder (mask_output_dir) for storing image files while processing is required.

    Parameters:

        result (ultralytics.engine.results.Results): A model prediction result.

        json_dir (str): The directory to your json files.

        label_num (int): The index of the label used. Used to refer to the correct shape.

        label (str): The name of the label for this prediction.
    
        mask_output_dir (str): The directory for outputting ground truth mask images.

    Returns:
    
        output (dict): A dictionary containing updated information of the ground truth mask. See predict_masks below.
    """

    # Find the corresponding ground truth json file 
    file_name = os.path.splitext(result.path)[0].split(os.sep)[-1]

    json_file = os.sep.join([json_dir, file_name + ".json"])
    mask_image_path = "{}_{}_TMask.png".format(os.path.join(mask_output_dir, file_name), label)

    with open(json_file) as f:
        json_data = json.load(f)

    img_width = json_data["imageWidth"]
    img_height = json_data["imageHeight"]


    points = json_data["shapes"][label_num]["points"][:]

    # Create mask layer based on model predicted bounding points
    orig_mask = np.zeros(result.orig_shape)

    orig_contour = []
    for pair in points:
        orig_contour.append([pair])
    mask_area = cv2.contourArea(np.int32(orig_contour))
    orig_mask = cv2.drawContours(orig_mask, np.int32([orig_contour]), -1, (255, 255, 255), thickness= -1)

    cv2.imwrite(mask_image_path, orig_mask)

    truth_mask = {
        "path": result.path,
        "label": json_data["shapes"][label_num]["label"],
        "points": points,
        "mask_data": orig_mask,
        "mask_path": mask_image_path,
        "mask_area": mask_area,
        "img_width": img_width, # Width of image
        "img_height": img_height # Height of image
    }

    return truth_mask


def createYaml(path, train, val, nc, names, output_dir):

    """
    **Creates a .yaml file based on given information.**

    The folders for labels are automatically detected, hence input is unnecessary.

    Parameters:

        path (str): The path to the YOLO dataset folder.

        train (str): The relative path from the YOLO dataset folder to the training images.

        val (str): The relative path from the YOLO dataset folder to the validation images.

        nc (int): The number of classes of bounding boxes.

        names ([str, ...]): The names of classes of bounding boxes.

        output_dir (str): The directory for storing output files.

    Returns:
        output: None
    """

    yamldata = {
    "path": path,
    "train": train,
    "val": val,
    "nc": nc,
    "names": names
    }
    # Write data to .yaml file
    with open(output_dir + "/data.yaml", 'w') as temp_file:
        yaml.dump(yamldata, temp_file, sort_keys=False)


def extraResultPlots(json_file):

    """
    **Generates additional plots based on provided .json file from trainModel().**

    Parameters:

        json_file (YOLO model): The path to your .json file.

    Returns:
        output: None
    """

    with open(json_file) as load_file:
        temp_json = json.load(load_file)
        print(temp_json)
        epoch_num = temp_json["Epoch Number"]
        temp_json.pop("Epoch Number")
        plt.figure(figsize=(32,40))
        for index, key in enumerate(temp_json.keys()):
            plt.subplot(5,5,index+1)
            plt.plot(epoch_num,temp_json[key],marker = "o")
            plt.title(key)
        print(len(temp_json))
    
    plt.show()


def image_split_by_json(json_file, img_dir, label_names, img_output_dir=None, txt_output_dir=None):

    """
    **Splits images per class in JSON files.**

    Should be used exclusively by YOLOTTSplit_splitClass().

    Parameters:

        json_file (str): The directory to your JSON file.

        img_dir (str): The directory to your images.

        label_names ([str, ...]): A list containing names for each class. The name should match the names used when labelling, in order according to their class ID.

        img_output_dir (str): The directory for outputting images.

        txt_output_dir (str): The directory for outputting text files.
            
    Returns:
    
        output: None

    """
    # Load json file
    with open(json_file) as f:
        json_data = json.load(f)

    # Find correspond image file
    img_path = os.path.join(img_dir,json_data["imagePath"])

    # Get common name for image and json
    file_name = os.path.splitext(json_file)[0].split(os.sep)[-1]

    # Image data for normalising
    full_width = json_data['imageWidth']
    full_height = json_data['imageHeight']

    for shape in json_data["shapes"]:
        
        # Identify label
        for name in label_names:
            if name == shape["label"]:
                label_id = label_names.index(name)
                print(label_id)

        points = np.array(shape["points"])

        # Save image
        if img_output_dir != None:

            img = Image.open(img_path)

            img_output_path = os.path.join(img_output_dir,file_name)

            with open(str(img_output_path) + "_{}_{}.png".format(label_id,shape["label"]), 'wb') as png_file:
                img.save(png_file, "PNG")
        
        # Save TXT with normalised coordinates
        if txt_output_dir != None:

            txt_info = "{}".format(label_id)

            for pair in points:
                txt_info += " {} {}".format(pair[0]/full_width,pair[1]/full_height)
                    
            txt_output_path = os.path.join(txt_output_dir,file_name)

            with open(str(txt_output_path) + "_{}_{}.txt".format(label_id,shape["label"]), 'w') as txt_file:
                txt_file.write("{}".format(txt_info))

    return None


def inferenceCheck(model, img_files, target_class = None, showImages = False):

    """
    **Passes inference images through the trained YOLO model for evaluation.**

    Should be combined with other functions to analyze results.

    Parameters:

        model (ultralytics.models.yolo.model.YOLO): Loaded model.

        img_files ([img_file, ...]): A list of inference images.

        showImages (bool): If true, all images with auto-generated bounding box with be shown in indivdual windows. Default: False

    Returns:
        output ([ultralytics.engine.results.Results, ...]): A list of result data from the model.
    """

    # Collects all inference images provided as PIL images and filter out all other files
    img_list = []

    for file in img_files:
        if file.endswith(".png"):
            img_list.append(Image.open(file))

    # Predicts masks with provided model for all images
    if target_class != None:
        results = model(img_list, classes = target_class)
    else:
        results = model(img_list)

    # Display all images in seperate windows if showImages is True
    if showImages:
        for result in results:
            result.show()        

    return results


def manual_mAP(IoU_list):

    """
    **Calculates mAP50 and mAP50-95 values from a list of intersection over union(IoU) values.**

    Parameters:
        IoU_list ([float, ...]]): A list of IoU floats.

    Returns:
        output (float_mAP50, float_mAP50_95): The mAP50 and mAP50_95 values.

    """
    # Assign rank values for each IoU values
    rank_list = []
    for IoU in IoU_list:
        rank = 0
        if IoU <= 0.50:
            rank = 0
        elif IoU <= 0.55:
            rank = 1
        elif IoU <= 0.60:
            rank = 2
        elif IoU <= 0.65:
            rank = 3
        elif IoU <= 0.70:
            rank = 4
        elif IoU <= 0.75:
            rank = 5
        elif IoU <= 0.80:
            rank = 6
        elif IoU <= 0.85:
            rank = 7
        elif IoU <= 0.85:
            rank = 8
        elif IoU <= 0.85:
            rank = 9
        else:
            rank = 10
        rank_list.append(rank)

    # Calculate mAP50 based on amount of non-zeros 
    mAP50 = (len(rank_list) - rank_list.count(0)) / len(rank_list)

    # Calculate mAP50_95 based on assigned rank values
    mAP50_95 = sum(rank_list) * 0.1 / len(rank_list)

    return mAP50, mAP50_95


def manual_score(predict_mask, truth_mask):
    
    """
    **Calculates dice coefficient and intersection over union values from the prediction mask and the ground truth mask.**

    Parameters:
        predict_mask (dict): Dictionary containing information about the prediction mask generated by calculatePredictMask().

        truth_mask (dict): Dictionary containing information about the ground truth mask generated by calculateTruthMask().

    Returns:
        output (float_dice, float_IoU): The dice coefficient and intersection over union values.

    """
    predict_data = predict_mask["mask_data"]
    predict_area = predict_mask["mask_area"]
    
    truth_data = truth_mask["mask_data"]
    truth_area = truth_mask["mask_area"]

    intersection_data = np.logical_and(predict_data, truth_data)
    intersection_area = np.sum(intersection_data)

    dice = 2 * intersection_area / (predict_area + truth_area)

    IoU = intersection_area / (predict_area + truth_area - intersection_area)

    return dice, IoU


def metric_tally(trainer):

    """
    **Collects information during model training and saves data as a .json file. Outputs progress occasionally in a different text file.**

    Should be called and used exclusively by trainModel().

    Parameters:
        trainer (ultralytics.models.yolo.segment.train.SegmentationTrainer): The trainer responsible for training the current model.

    Returns:
        output: None
    """

    global epoch_num_global, epoch_count, out_dir, result_dict, epochMod

    epoch_count += 1
    final_epoch = False

    if epoch_count == 1: # Initializing tally and output
        result_dict = {
        "Epoch Number": [],
        "Best Fitness": [],
        "Box Loss": [],
        "Lowest Box Loss": [],
        "Seg Loss": [],
        "Lowest Seg Loss": [],
        "Cls Loss": [],
        "Lowest Cls Loss": [],
        "Dfl Loss": [],
        "Lowest Dfl Loss": [],
        "Box Precision": [],
        "Highest Box Precision": [],
        "Box Recall": [],
        "Highest Box Recall": [],
        "Box mAP50": [],
        "Highest Box mAP50": [],
        "Box mAP50-95": [],
        "Highest Box mAP50-95": [],
        "Mask Precision": [],
        "Highest Mask Precision": [],
        "Mask Recall": [],
        "Highest Mask Recall": [],
        "Mask mAP50": [],
        "Highest Mask mAP50": [],
        "Mask mAP50-95": [],
        "Highest Mask mAP50-95": []
        }

        txt_file = open(out_dir + "/custom_output.txt", 'w')
        os.startfile(out_dir + "\\custom_output.txt")
        txt_startup = "Custom output file initiated.\nThis text file is stored at: " + out_dir + "/custom_output.txt\nWarning: Running trainNewModel() again will delete all contents in this file!\nIf you are viewing this file using Notepad, you may have to minimize and maximize the window to see live outputs.\n\n"
        txt_file.write(txt_startup)
    else:
        txt_file = open(out_dir + "/custom_output.txt", 'a')

        result_dict["Epoch Number"].append(epoch_count)
        result_dict["Best Fitness"].append(trainer.best_fitness) if trainer.best_fitness != None else result_dict["Best Fitness"].append(0)
        result_dict["Box Loss"].append(trainer.metrics["val/box_loss"])
        result_dict["Seg Loss"].append(trainer.metrics["val/seg_loss"])
        result_dict["Cls Loss"].append(trainer.metrics["val/cls_loss"])
        result_dict["Dfl Loss"].append(trainer.metrics["val/dfl_loss"])
        result_dict["Lowest Box Loss"].append(min(result_dict["Box Loss"]))
        result_dict["Lowest Seg Loss"].append(min(result_dict["Seg Loss"]))
        result_dict["Lowest Cls Loss"].append(min(result_dict["Cls Loss"]))
        result_dict["Lowest Dfl Loss"].append(min(result_dict["Dfl Loss"]))
        result_dict["Box Precision"].append(trainer.metrics["metrics/precision(B)"])
        result_dict["Box Recall"].append(trainer.metrics["metrics/recall(B)"])
        result_dict["Box mAP50"].append(trainer.metrics["metrics/mAP50(B)"])
        result_dict["Box mAP50-95"].append(trainer.metrics["metrics/mAP50-95(B)"])
        result_dict["Highest Box Precision"].append(max(result_dict["Box Precision"]))
        result_dict["Highest Box Recall"].append(max(result_dict["Box Recall"]))
        result_dict["Highest Box mAP50"].append(max(result_dict["Box mAP50"]))
        result_dict["Highest Box mAP50-95"].append(max(result_dict["Box mAP50-95"]))
        result_dict["Mask Precision"].append(trainer.metrics["metrics/precision(M)"])
        result_dict["Mask Recall"].append(trainer.metrics["metrics/recall(M)"])
        result_dict["Mask mAP50"].append(trainer.metrics["metrics/mAP50(M)"])
        result_dict["Mask mAP50-95"].append(trainer.metrics["metrics/mAP50-95(M)"])
        result_dict["Highest Mask Precision"].append(max(result_dict["Mask Precision"]))
        result_dict["Highest Mask Recall"].append(max(result_dict["Mask Recall"]))
        result_dict["Highest Mask mAP50"].append(max(result_dict["Mask mAP50"]))
        result_dict["Highest Mask mAP50-95"].append(max(result_dict["Mask mAP50-95"]))
         
    if epoch_count == epoch_num_global:
        final_epoch = True
        txt_file.write("Final Epoch Training Completed!\n")

    if epoch_count % epochMod == 0 or final_epoch == True:
       
       txt_info = "Epoch {} Completed.\n".format(epoch_count)

       for key in result_dict.keys():
            txt_info += key + ": {:.5f}\t".format(result_dict[key][epoch_count-2])

       txt_info += "\n\n"
       txt_file.write(txt_info)

       if final_epoch == True:
           txt_file.write("Training results saved at: {}\n".format(trainer.save_dir))
           txt_file.write("Best Checkpoint saved at: {}\n".format(trainer.best))
           txt_file.write("Device used for training: {}\n".format(trainer.device))
           with open(out_dir + "/output_data.json", "w") as json_file:
               json.dump(result_dict, json_file)
               txt_file.write("Results dictionary saved as: " + out_dir + "/output_data.json")
               

def metric_tally_silent(trainer):

    """
    **Collects information during model training and saves data as a .json fil without creating and outputting in a text file.**

    Should be called and used exclusively by trainModel().

    Parameters:
        trainer (ultralytics.models.yolo.segment.train.SegmentationTrainer): The trainer responsible for training the current model.

    Returns:
        output: None
    """
    global epoch_num_global, epoch_count, out_dir, result_dict

    epoch_count += 1

    if epoch_count == 1: # Initializing tally
        result_dict = {
        "Epoch Number": [],
        "Best Fitness": [],
        "Box Loss": [],
        "Lowest Box Loss": [],
        "Seg Loss": [],
        "Lowest Seg Loss": [],
        "Cls Loss": [],
        "Lowest Cls Loss": [],
        "Dfl Loss": [],
        "Lowest Dfl Loss": [],
        "Box Precision": [],
        "Highest Box Precision": [],
        "Box Recall": [],
        "Highest Box Recall": [],
        "Box mAP50": [],
        "Highest Box mAP50": [],
        "Box mAP50-95": [],
        "Highest Box mAP50-95": [],
        "Mask Precision": [],
        "Highest Mask Precision": [],
        "Mask Recall": [],
        "Highest Mask Recall": [],
        "Mask mAP50": [],
        "Highest Mask mAP50": [],
        "Mask mAP50-95": [],
        "Highest Mask mAP50-95": []
        }

    else:  # Saves data of each epoch into dictionaary. Ignore first epoch as most values are 0.
        result_dict["Epoch Number"].append(epoch_count)
        result_dict["Best Fitness"].append(trainer.best_fitness) if trainer.best_fitness != None else result_dict["Best Fitness"].append(0)
        result_dict["Box Loss"].append(trainer.metrics["val/box_loss"])
        result_dict["Seg Loss"].append(trainer.metrics["val/seg_loss"])
        result_dict["Cls Loss"].append(trainer.metrics["val/cls_loss"])
        result_dict["Dfl Loss"].append(trainer.metrics["val/dfl_loss"])
        result_dict["Lowest Box Loss"].append(min(result_dict["Box Loss"]))
        result_dict["Lowest Seg Loss"].append(min(result_dict["Seg Loss"]))
        result_dict["Lowest Cls Loss"].append(min(result_dict["Cls Loss"]))
        result_dict["Lowest Dfl Loss"].append(min(result_dict["Dfl Loss"]))
        result_dict["Box Precision"].append(trainer.metrics["metrics/precision(B)"])
        result_dict["Box Recall"].append(trainer.metrics["metrics/recall(B)"])
        result_dict["Box mAP50"].append(trainer.metrics["metrics/mAP50(B)"])
        result_dict["Box mAP50-95"].append(trainer.metrics["metrics/mAP50-95(B)"])
        result_dict["Highest Box Precision"].append(max(result_dict["Box Precision"]))
        result_dict["Highest Box Recall"].append(max(result_dict["Box Recall"]))
        result_dict["Highest Box mAP50"].append(max(result_dict["Box mAP50"]))
        result_dict["Highest Box mAP50-95"].append(max(result_dict["Box mAP50-95"]))
        result_dict["Mask Precision"].append(trainer.metrics["metrics/precision(M)"])
        result_dict["Mask Recall"].append(trainer.metrics["metrics/recall(M)"])
        result_dict["Mask mAP50"].append(trainer.metrics["metrics/mAP50(M)"])
        result_dict["Mask mAP50-95"].append(trainer.metrics["metrics/mAP50-95(M)"])
        result_dict["Highest Mask Precision"].append(max(result_dict["Mask Precision"]))
        result_dict["Highest Mask Recall"].append(max(result_dict["Mask Recall"]))
        result_dict["Highest Mask mAP50"].append(max(result_dict["Mask mAP50"]))
        result_dict["Highest Mask mAP50-95"].append(max(result_dict["Mask mAP50-95"]))
         
    if epoch_count == epoch_num_global:
        with open(out_dir + "/output_data.json", "w") as json_file:
            json.dump(result_dict, json_file)
            print("Results dictionary saved as: " + out_dir + "/output_data.json")

 
def predict_imgs(model, img_files, mask_output_dir, label_names):

    """
    **Predicts the masks for each class in an image and outputs mask images.**

    Parameters:

        model (ultralytics.models.yolo.model.YOLO): Loaded model.

        img_files ([img_file, ...]): A list of inference images.
        
        mask_output_dir (str): The directory for outputting prediction mask images.

        label_names ([str, ...]): A list containing names for each class. The name should match the names used when labelling, in order according to their class ID.
        
            
    Returns:
    
        output: None

    """

    for index, label in enumerate(label_names):
        results = inferenceCheck(model, img_files, index)

        for result in results:
            calculatePredictMask(result, label, mask_output_dir)

    return None


def segmentation_inference(model, img_dir, json_dir, mask_output_dir, truth_output_dir, label_names, plot_size = 10):

    """
    **Runs an inference check for a trained YOLO model.**

    This program first loads the model and runs the model with provided images in "prediction" mode. 
    Then, the image is plotted with the prediction along the ground truth as comparison.
    The dice coefficient and intersection over union value is also given.

    Parameters:

        model (class 'ultralytics.models.yolo.model.YOLO'): Loaded YOLO model.

        img_dir (str): The directory to your images.
            
        json_dir (str): The directory to your json files.

        mask_output_dir (str): The directory for outputting prediction mask images.
        
        truth_output_dir (str): The directory for outputting ground truth mask images.

        label_names ([str, ...]): A list containing names for each class. The name should match the names used when labelling, in order according to their class ID.

        plot_size (int): The length of each square plot in inches. Final size: width = plot_size * 3, height = plot_size = plot_size * num_of_prediction
            
    Returns:
    
        output: None

    """

    # Initialise processing list
    dice_list = []
    IoU_list = []
    results_storage = []
    results = []

    # Retrieve image files
    img_files = glob.glob("{}/**.png".format(img_dir))

    # Predict a class with model individually
    for index, label in enumerate(label_names):
        results_storage.append(inferenceCheck(model, img_files, index, False))

    # Transpose results to simplify plotting
    for i in range(len(results_storage[0])):
        for j in range(len(results_storage)):
            results.append(results_storage[j][i])

    # Initialize matploblib figure 
    plot_num = len(results)
    figure = plt.figure(figsize=(plot_size*3, plot_size*plot_num))
    figure.subplots(plot_num,3)

    # Run inference for every result obtained
    for index, result in enumerate(results):

        # Skips image if no detection
        if result.masks == None:
            print("No detection!")
            continue
            
        plot_index = 3 * index + 1

        # Obtain class label name and id 
        label_id = index % len(label_names)
        result_label = label_names[label_id]

        # First Subplot: Prediction
        subplot = plt.subplot(plot_num,3,plot_index)
        subplot.imshow(Image.open(result.path), cmap='gray', vmin=0, vmax=255, alpha = 0.3)

        # Obtain prediction information
        predict_mask = calculatePredictMask(result, result_label, mask_output_dir)

        # Apply formating
        plt.title("{}".format(predict_mask["path"]))    
        plt.xlabel("{}: Predicted {}".format(plot_index, predict_mask["label"]) )
        
        # Plot predicted mask
        predict_polygon = patches.Polygon(predict_mask["points"] ,closed = False, alpha = 0.3, color= "r")
        subplot.add_patch(predict_polygon)

        # Second Subplot: Ground Truth
        plot_index += 1
        subplot = plt.subplot(plot_num,3,plot_index)
        subplot.imshow(Image.open(result.path), cmap='gray', vmin=0, vmax=255, alpha = 0.3)

        # Obtain ground truth information
        truth_mask = calculateTruthMask(result, json_dir, label_id, result_label, truth_output_dir)

        # Apply formating
        plt.xlabel("{}: Ground Truth {}".format(plot_index, truth_mask["label"]) )
        
        # Plot ground truth mask
        truth_polygon = patches.Polygon(truth_mask["points"] ,closed = False, alpha = 0.3, color= "b")
        subplot.add_patch(truth_polygon)

        # Third Subplot: Comparison
        plot_index += 1
        subplot = plt.subplot(plot_num,3,plot_index)

        subplot.imshow(Image.open(result.path), cmap='gray', vmin=0, vmax=255, alpha = 0.3)

        # Plot masks for comparision
        predict_polygon = patches.Polygon(predict_mask["points"] ,closed = False, alpha = 0.3, color= "r")
        subplot.add_patch(predict_polygon)

        truth_polygon = patches.Polygon(truth_mask["points"] ,closed = False, alpha = 0.3, color= "b")
        subplot.add_patch(truth_polygon)

        # Calculate metrics
        dice, IoU = manual_score(predict_mask, truth_mask)
        dice_list.append(dice)
        IoU_list.append(IoU)

        plt.xlabel("{}: Comparison\nDice Coefficient: {}\nIntersection over Union (IoU): {}".format(plot_index,dice,IoU))


    plt.show()

    mAP50, mAP50_95 = manual_mAP(IoU_list)

    # Output statistics
    print("Inference check complete!")
    print("Number of Images: {}\n".format(len(results)))
    print("Highest IoU (Intersection over Union): {}".format(max(IoU_list)))
    print("Lowest IoU (Intersection over Union): {}".format(min(IoU_list)))
    print("Average IoU (Intersection over Union): {}".format(sum(IoU_list)/len(IoU_list)))
    print("mAP50: {}".format(mAP50))
    print("mAP50-95: {}".format(mAP50_95))
    print("Highest Dice Coefficient: {}".format(max(dice_list)))
    print("Lowest Dice Coefficient: {}".format(min(dice_list)))
    print("Average Dice Coefficient: {}".format(sum(dice_list)/len(dice_list)))


def trainModel(dataset_dir, model_path = None, epoch_num = 100, simple_output = True, s_out_per_epoch = 10):
        
    """
    **Trains a model based on provided YOLO Dataset.**

    The model will be trained with (epoch_num) epochs. Leave model_path to train a brand new "yolo11n-seg.pt" model.

    You can choose to open a simpler output system recorded on a newly generated text file. Use s_out_per_epoch to control the amount of outputs.

    Parameters:
        dataset_dir (str): The target YOLO dataset. This directory should include images, labels and a .yaml file.

        model_path (str): The target model to be trained. The model should be in ".pt" format. Input None to train new "yolo11n-seg.pt" model. Default: None

        epoch_num (int): Number of epochs used for training. Default: 100

        simple_output (bool): Whether to create and open a text file to output statistics. Default: True

        s_out_per_epoch (int): Number of epochs before an output in the simple output system. Ignored if simple_output = False. Default: 10

    Returns:
        output: None
    """

    global out_dir, epochMod, epoch_count, epoch_num_global

    epoch_count = 0
    epochMod = s_out_per_epoch 
    out_dir = dataset_dir
    epoch_num_global = epoch_num

    if model_path == None:
        model = YOLO("yolo11n-seg.pt", verbose = False)
    else:
        model = YOLO(str(model_path), verbose = False)

    if simple_output:
        model.add_callback("on_train_epoch_end", metric_tally)
    else:
        model.add_callback("on_train_epoch_end", metric_tally_silent)

    model.train(data = dataset_dir + "/data.yaml", epochs=epoch_num, verbose = False)

    print("Training complete!")


def YOLOTTSplit_splitClass(json_files, img_dir, target_dir, label_names, test_section = 0.20):

    """
    **Splits images and corresponding text to training and testing sets, and automatically compiles files into YOLO required format.**

    Please make sure all images and their corresponding text files have the same filename!

    Parameters:

        json_files ([str, ...]): A list of directories to each of your JSON files.

        img_dir (str): The directory to your images.

        target_dir (str): The directory for outputting files.

        label_names ([str, ...]): A list containing names for each class. The name should match the names used when labelling, in order according to their class ID.
        
        test_section (float): The percentage of images and labels to be used for testing/validating. Default: 0.20
            
    Returns:
    
        output: None

    """
    # Splits JSON files to training and testing groups
    train, test, _, _ = train_test_split(json_files, json_files, test_size=test_section, random_state=42)

    for json_file in train:

        # Create directories
        img_output_dir = os.sep.join([target_dir,"images","train"])
        os.makedirs(img_output_dir, exist_ok=True)

        txt_output_dir = os.sep.join([target_dir,"labels","train"])
        os.makedirs(txt_output_dir, exist_ok=True)

        image_split_by_json(json_file, img_dir, label_names, img_output_dir, txt_output_dir)

    for json_file in test:

        # Create directories
        img_output_dir = os.sep.join([target_dir,"images","val"])
        os.makedirs(img_output_dir, exist_ok=True)

        txt_output_dir = os.sep.join([target_dir,"labels","val"])
        os.makedirs(txt_output_dir, exist_ok=True)

        image_split_by_json(json_file, img_dir, label_names, img_output_dir, txt_output_dir)

    return None