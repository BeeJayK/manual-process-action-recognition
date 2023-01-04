#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:03:40 2021

@author: marlinberger

helper functions in the context of the training of inference 2 (motion
classification)
"""

# python packages
import csv
import cv2
import logging
import mediapipe
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import ImageFont
from tensorflow import keras



def check_import():
    """use this function to check if the import worked
    """
    logging.debug("\nyou made it in\n")


def name_constants(keyword):
    """returns the corresponding file- or foldername to a keyword. This way,
    they are defined at a fixed place, but are accessable all over the project

    Args:
        keyword (str): used to access the corresponding name

    Returns:
        name (str): (file-)name of anything
    """
    # folders
    # --------
    # names of the folders used to shift and create the train-ready data
    RAW_VIDEO_DIR_NAME = "00_Raw_Inputs"
    RENAMED_INPUTS_DIR_NAME = "01_Renamed_Inputs"
    LABEL_TABBLE_DIR_NAME = "02_Label_Tables"
    TRAIN_DATA_DIR_NAME = "03_Train_Data"
    TRAIN_DATA_PACKS_DIR_NAME = "04_Train_Data_Packs"
    EVALUATE_DIR_NAME = "05_Evaluate"
    ANALYZE_DIR_NAME = "06_Analyze"
    ARCHITECTURE_DIR_NAME = "99_Architectures"
    # save folder for models
    MODEL_SAVE_DIR_NAME = "_saves"
    # subfolder in "05_Evaluate"
    EV_BASE = "01_Processed_Videos"
    EV_LABELS = "02_Label_Tables"
    EV_MODEL = "03_Model"
    EV_RESULTS = "04_Results"
    # subfolder in "06_Analyze"
    AN_MODEL = "01_Models"

    # files
    # --------
    # name of the _config file for neural nets
    CONFIG_FILE_NAME = "_config.csv"

    # suffix's
    # --------
    # Ending for the table-frame-motionclass tables
    TABLES_SUFFIX = ".tab.txt"

    # serach through just created local variables by keyword
    name = locals()[keyword]
    return (name)


def get_folderpaths_in_dir(path_to_dir):
    """return all paths from folders inside the given directory

    Args:
        path_to_dir (PosixPath): path to the dir, from which the folders shall
                                 be returned

    Returns:
        paths (list(PosixPath)): paths to the folders inside the given directory
    """
    paths = [x for x in path_to_dir.iterdir() if x.is_dir()]
    return (paths)


def mediapipe_hands_paramsets(paramset_n):
    """save packs of parameters, with which the mediapipe_hands solution
    is driben, here. This way, different versions can easaly be developed and
    used across several devices and architectures

    Args:
        paramset_n (int): index of the param set to return

    Return:
        paramset (dict): params to use with the mediapip-hands solution
    """

    # initial params to use
    if paramset_n == 0:
        paramset = {
            "static_image_mode": True,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5
        }

    # NOTE: if "max_num_hands" is set to something else than 2, it will break
    # 		the current structure of how the mediapipe results are saved and
    # 		send across the stack (this applies to convert_mediapipe_output_v2()
    # 		function. convert_mediapipe_output() can handle anything but needs
    # 		different input pipelines for the tensorflow models)
    paramset["max_num_hands"] = 2

    return (paramset)


def mediapipe_hands_process(image, paramset):
    """routine to process straigt images (NOT recorded by a selfie cam) with
    the mediapipes hands solution.

    Args:
        image (numpy.ndarray): the picture, probably loaded with cv2, at least
                               in the training context
        paramset (dict): the params to be used, bundled in a dict. For shape,
                         look at the function of this script:
                         mediapipe_hands_paramsets()

    Returns:
        results (mediapipe.[.].SolutionOutputs): The results from mediapipe
                                                 hands
    """
    with mediapipe.solutions.hands.Hands(
        static_image_mode=paramset["static_image_mode"],
        max_num_hands=paramset["max_num_hands"],
        min_detection_confidence=paramset["min_detection_confidence"],
        min_tracking_confidence=paramset["min_tracking_confidence"],
    ) as hands:
        # BGR 2 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # flip on horizontal, as mediapipe expects the output from a
        # front camera (this way, "handedness" works smoothly)
        image = cv2.flip(image, 1)
        # set flag
        image.flags.writeable = False
        # Get the mediapipe output
        results = hands.process(image)
    return (results)


def convert_mediapipe_output_v2(mediapipe_result, ts_ms=0, ts_img=0):
    """convert the mediapipe result to the shape in which it will be sent across
    the mqtt stack. It is also used in the context of creating the training
    data, to ensure all formats are exactly the same.
    The structure is designed to be extantable without breaking older structures
    and looks like this, but has a different shape than the v1-function. It is
    structured as a 1-line-table. This way, processing is SUPER efficient, while
    the readability suffers, BUT, these files wont be read anyway. And if they
    shall be read - during debugging etc. - they are still readable, just a
    little bit less nice
    This approach assumes that the "max_num_hands" argument for the mediapipe
    solution is set to 2.

    Args:
        mediapipe_result (mediapipe.process): the output from the mediapipe
                                              analysis
        ts_ms (int): timestamp [datetime], of when the photo command was given
        ts_img (int): timestamp [datetime], of when the photo was taken

    Returns:
        structured_output (dict): the structures output
    """
    # extract coordinates and mediapipe's hand prediction
    coordinates = mediapipe_result.multi_hand_landmarks
    hands = mediapipe_result.multi_handedness

    # initialise the hand-representation storage that will be passed along the
    # pipeline in every aspect
    structured_output = dict()

    # apply timestamp
    structured_output["timestamp_photo_command"] = ts_ms
    structured_output["timestamp_image_taken"] = ts_img
    structured_output["no_hands_detected_flag"] = 0

    # initialise hand-coordinate columns
    for hand_num in [0, 1]:
        # insert the mediapipes prediction wether it is a left or right hand
        # initialise both with -1, which signals no hand got detected (nan's
        # are thus exclusive to coordinates, which has some benefits for
        # imputation layers)
        structured_output[f"{hand_num}_pred"] = -1
        structured_output[f"{hand_num}_pred_conf"] = -1
        for hand_point in range(21):
            structured_output[f"{hand_num}_{hand_point}_x"] = np.nan
            structured_output[f"{hand_num}_{hand_point}_y"] = np.nan
            structured_output[f"{hand_num}_{hand_point}_z"] = np.nan

    # itterate through mediapipe results and assign them. if there are no, this
    # part is skipped and the handcoordinates are already filled withs "False"
    if isinstance(coordinates, list):
        for hand_num, landmarks in enumerate(coordinates):
            # assign the hand prediction. left == 0, right == 1
            hand_masked = assignments(
                hands[hand_num].classification[0].label.lower()
            )
            # safe the handedness results
            structured_output[f"{hand_num}_pred"] = hand_masked
            structured_output[f"{hand_num}_pred_conf"] = (
                hands[hand_num].classification[0].score)

            # fill up the hand points
            for hand_point, coordinate in enumerate(landmarks.landmark):
                structured_output[f"{hand_num}_{hand_point}_x"] = coordinate.x
                structured_output[f"{hand_num}_{hand_point}_y"] = coordinate.y
                structured_output[f"{hand_num}_{hand_point}_z"] = coordinate.z
    else:
        # signal that no hand was detected
        structured_output["no_hands_detected_flag"] = 1

    return (structured_output)


def write_coordinates_to_csv(filepath, mediapipe_result):
    """helper function, to save the mediapipe-output as csv file.

    Args:
        filepath (PosixPath): where to safe the file. includes the filename
        structured_output (dict): the processed output from the mediapipe
                                  analysis, in the structure that is used
                                  across the project stack (see function
                                  convert_mediapipe_output() above)
    """
    # open the given file, write it key by key on the first dictionary level
    with open(str(filepath) + ".csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mediapipe_result.items():
            writer.writerow([key, value])


def write_coordinates_to_tfrecord(filepath, mediapipe_result):
    """helper function, to save the mediapipe-output as tfrecord file. This
    will speed up the reading during training

    Args:
        filepath (PosixPath): where to safe the file. includes the filename
        structured_output (dict): the processed output from the mediapipe
                                  analysis, in the structure that is used
                                  across the project stack (see function
                                  convert_mediapipe_output() above)
    """
    # TODO(MB): implement if it otherwise leads to a bottleneck
    pass


def sort_paths_by_first_num(paths):
    """sort given paths by a number, that comes first in the file's name
    and are delimited by an underline from the rest of the file's name

    Args:
        paths (list(PosixPath)): a list of paths

    Returns:
        sorted_paths (list(PosixPath)): the same list of paths, but sorted by
                                        number in the beginning
    """
    sorted_paths = sorted(paths, key=lambda x: int(x.name.split("_")[0]))
    return (sorted_paths)


def get_motionclass_vector(tablepath, tot_frame_num):
    """return an assignment for every frame with it's motion class, which
    can be used to loop through all picture and have theire motion class
    appearand

    Args:
        tablepath (PosixPath): path where to find the table
        tot_frame_num (int): how many frames the inspected video contains

    Returns:
        motionclass_vector (list): len equals the given input pictures, each
                                   each position contains the motion class for
                                   this frame
    """
    # open the label table
    table_lines = open_labels_table(tablepath)

    # open storage for the value pairs
    label_frame_assignment = []
    for value_pair in table_lines:
        # extract each startframe and motionclass
        startframe, motionclass = value_pair.split(",")
        # store them as bundled integers
        label_frame_assignment.append((int(startframe), int(motionclass)))

    # the label vector, containing one entry per picture
    motionclass_vector = []
    for idx, (startframe, motionclass) in enumerate(label_frame_assignment):
        # determine for how many pictures the motionclass appears.
        # distinguish the way this is determined for the last itteration
        if idx < len(label_frame_assignment) - 1:
            class_present_for = (label_frame_assignment[idx + 1][0] -
                                 startframe)
        else:
            class_present_for = tot_frame_num - startframe
        # build the motionclass vector by appending the motion class for
        # so many times, like frames they appear
        motionclass_vector.extend([motionclass for _ in range(
            class_present_for)])

    return (motionclass_vector)


def get_video_writer(video_safe_path, Eval_Specs, example_pic):
    """return an object that can be used to create a video under the given path

    Args:
        safepath (PosixPath): the path to the video that shall be created
        Eval_Specs (obj): object with all the information about the choosen
                          evaluation folder
        example_pic (np.arr): a representative image for the video that's about
                              to be created

    Returns:
        OutputWriter (cv2 object): the initialised video writer
    """
    # calculate the output framerate NOTE: if train-fps differs from
    # evaluate-fps, this needs to be considered somewhere and probably the
    # calculation here needs to be performed differently
    fps = Eval_Specs.ORIGINAL_FPS / Eval_Specs.FPS_REDUCTION_FAC

    # set the output shape
    dims = example_pic.shape[1], example_pic.shape[0]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    OutputWriter = cv2.VideoWriter(str(video_safe_path), fourcc, fps, dims)
    return (OutputWriter)


class Eval_Specs():
    """an object of this class will contain information about the folder
    structure, model used etc., which are needed at different places and are
    therefore commulated inside this object
    """

    def __init__(self, MAIN_FOLDER, MODE):
        """initialise the eval-spec object

        Args:
            MAIN_FOLDER (str): the name of the evaluation folder to be evaluated
            mode (str): in ["VisT", "Analyzer"]. Indicates what's done
        """
        # get paths and names
        self.MAIN_FOLDER = MAIN_FOLDER
        self.MODE = MODE
        self.CURR_EVAL_MAIN_PATH = self.get_curr_eval_path()

        # get the names and paths of all models
        self.TRAIN_RUN_NAMES, self.TRAIN_RUN_PATHS = self.get_train_run_infos()

        # as for the VisT-Routine only one model is appearent, everything can
        # be initialised imidiatly
        if self.MODE == "VisT":
            # initialise video specific stuff
            self.PATH_TO_VIDEOS_MAIN = Path(
                f"{self.CURR_EVAL_MAIN_PATH}/" \
                f"{name_constants('EV_BASE')}"
            )
            self.VIDEO_PATHS = get_folderpaths_in_dir(
                self.PATH_TO_VIDEOS_MAIN
            )
            self.INPUT_VIDEOS_N = len(self.VIDEO_PATHS)
            # get the label tables, which is also only needed for the video-
            # overlay-routine
            self.LABEL_TABLE_PATHS = get_filepaths_in_dir(
                self.CURR_EVAL_MAIN_PATH / name_constants("EV_LABELS")
            )
            # intialise the model specific stuff
            self.setup_on_run(self.TRAIN_RUN_PATHS[0])
        
        elif self.MODE == "Analyzer":
            # if the Analyzer routine uses this object, it will re-initialize
            # this object on evey run/model on it's own, by calling the object's
            # function self.setup_on_run() from a loop that itterates through
            # all detected model paths
            pass

    def get_curr_eval_path(self):
        """return the path to the folder from which everything starts

        Returns:
            CURR_EVAL_MAIN_PATH (PosixPath): path to the current eval dir
        """
        # get the name of the folder in which the outer eval/analyze-folders lie
        if self.MODE == "VisT":
            TOPLEVEL_FOLDER = name_constants("EVALUATE_DIR_NAME")
        elif self.MODE == "Analyzer":
            TOPLEVEL_FOLDER = name_constants("ANALYZE_DIR_NAME")
        
        # get the path of the main eval/analyzation folder
        CURR_EVAL_MAIN_PATH = Path(
            f"{TOPLEVEL_FOLDER}/" \
            f"{self.MAIN_FOLDER}"
        )

        return (CURR_EVAL_MAIN_PATH)

    def get_train_run_infos(self):
        """extract the name of the run, whichs model got placed in the choosen
        evaluation folder to be used

        Returns:
            TRAIN_RUN_NAMES (list(str)): the names of the runs from the models
                                         used
            TRAIN_RUN_PATHS list((PosixPath)): paths to the folders of the
                                               models used
        """
        # get the name of the folder in which models are to be found
        if self.MODE == "VisT":
            MODEL_SUB_PATH = name_constants("EV_MODEL")
        elif self.MODE == "Analyzer":
            MODEL_SUB_PATH = name_constants("AN_MODEL")
        
        # get the path where the models are layed
        MODEL_FOLDER_PATH = Path(
            f"{self.CURR_EVAL_MAIN_PATH}/" \
            f"{MODEL_SUB_PATH}"
        )

        # get the paths and names of the models (== runs) in there
        TRAIN_RUN_PATHS = get_folderpaths_in_dir(MODEL_FOLDER_PATH)
        TRAIN_RUN_NAMES = [path.name for path in TRAIN_RUN_PATHS]

        return (TRAIN_RUN_NAMES, TRAIN_RUN_PATHS)

    def setup_on_run(self, run_path):
        """this function set's up this object to analyze a specific run, based
        on a run's folder. This folder will contain the h5-model and a config
        file with all information needed to initialise data etc.

        Args:
            run_path (PosixPath): path to the desired run's folder
        """
        # assign the runs name
        self.RUN_NAME = run_path.name
        # get the models configs
        self.PATH_TO_MODEL_CONFIG = self.get_model_config_path(run_path)
        # load the config dict from the run
        self.model_config = simple_model_config_df_2_dict_converter(
            df_from_model_config(
                self.PATH_TO_MODEL_CONFIG
            )
        )
        # get the sequence len and fps reduction fac of the model used
        self.SEQ_LEN = int(
            self.model_config["sequence_len"]
        )
        self.FPS_REDUCTION_FAC = int(
            self.model_config["fps_reduce_fac"]
        )
        # get fps of the input video from the config file
        # NOTE: the framerate of the videodata used for training needs to be
        #       the same as for the evaluated here. Otherwise, the dynamic
        #       calculation of the output framerate (in the motion_tracking_
        #       helpers.py Script) will lead to slower or fastened output videos
        self.ORIGINAL_FPS = float(
            self.model_config["input_framerate"]
        )

        # load the trained model
        self.model = self.get_model(run_path)

    def get_model_config_path(self, run_path):
        """return path to the config file of the model used

        Args:
            run_path (PosixPath): path to the desired run's folder
        
        Returns:
            PATH_TO_MODEL_CONFIG (PosixPath): the path to the file
        """
        CONFIG_FILE_NAME = name_constants("CONFIG_FILE_NAME")
        PATH_TO_MODEL_CONFIG = Path(
            f"{run_path}/" \
            f"{CONFIG_FILE_NAME}"
        )
        return (PATH_TO_MODEL_CONFIG)

    def get_model(self, run_path):
        """load the trained model which shall be used for evaluation/shall
        be evaluated

        Args:
            run_path (PosixPath): path to the desired run's folder

        Returns:
            model (tf.model): the trained model
        """
        run_cont = get_filepaths_in_dir(run_path)
        # NOTE: this could also get saftey checked, e.g. if there is only one
        # 		h5 model in etc.
        model_path = [path for path in run_cont if ".h5" in str(path)][0]
        # load the model
        # NOTE: the custom_layers.py script needs to be imported while this
        #       script runs, otherwise the loader cannot decode the custom
        #       layers used
        model = keras.models.load_model(model_path)
        return (model)


if __name__ == "__main__":
    pass
