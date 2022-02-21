# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:00:47 2022

@author: The code is owned by https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/main/Evaluation/EvaluateSpotting.py

"""

######################################################################################################
#The code is originally owned by https://github.com/SilvioGiancola/SoccerNetv2-DevKit/blob/main/Evaluation/EvaluateSpotting.py
######################################################################################################


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from SoccerNet.Evaluation.ActionSpotting import evaluate

if __name__ == '__main__':

    # Load the arguments
    parser = ArgumentParser(description='Evaluation for Action Spotting', 
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path', required=True,
                        type=str, help='Path to the SoccerNet-V2 dataset folder (or zipped file) with labels')
    parser.add_argument('--Predictions_path', required=True,
                        type=str, help='Path to the predictions folder (or zipped file) with prediction')
    parser.add_argument('--Prediction_file', required=False, type=str,
                        help='Name of the prediction files as stored in folder (or zipped file) [None=try to infer it]', default=None)
    parser.add_argument('--split', required=False, type=str,
                        help='Set on which to evaluate the performances', default="test")
    parser.add_argument('--version', required=False, type=int,
                        help='Version of SoccerNet [1,2]', default=2)
    parser.add_argument('--metric', required=False, type=str,
                        help='tight (tight average mAP 2022) or loose (average mAP 2021)', default="tight")

    args = parser.parse_args()

    # a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown 
    results = evaluate(SoccerNet_path=args.SoccerNet_path, Predictions_path=args.Predictions_path,
                       split=args.split, version=args.version, prediction_file=args.Prediction_file, metric=args.metric)

    print("tight Average mAP: ", results["a_mAP"])
    print("tight Average mAP per class: ", results["a_mAP_per_class"])
    print("tight Average mAP visible: ", results["a_mAP_visible"])
    print("tight Average mAP visible per class: ", results["a_mAP_per_class_visible"])
    print("tight Average mAP unshown: ", results["a_mAP_unshown"])
    print("tight Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])