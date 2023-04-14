# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Union

import numpy as np
import trackeval
from mmengine.logging import MMLogger


class MOTEvaluator:
    allowed_metrics = ['HOTA', 'CLEAR', 'Identity']

    def __init__(self,
                 tracker_dir: str,
                 gts_dir: str,
                 metric: Union[str, List[str]] = ['HOTA', 'CLEAR',
                                                  'Identity']):
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        for metric in metrics:
            if metric not in self.allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')
        self.metrics = metrics

        self.tracker_dir = tracker_dir
        self.gts_dir = gts_dir
        self.benchmark = 'MOT17'

    def compute_metrics(self) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()

        eval_results = dict()
        eval_config = trackeval.Evaluator.get_default_eval_config()

        # need to split out the tracker name
        # caused by the implementation of TrackEval
        trackers_dir, tracker_name = self.tracker_dir.rsplit(osp.sep, 1)
        dataset_config = self.get_dataset_cfg(self.gts_dir, trackers_dir,
                                              tracker_name)

        evaluator = trackeval.Evaluator(eval_config)
        dataset = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics = [
            getattr(trackeval.metrics,
                    metric)(dict(METRICS=[metric], THRESHOLD=0.5))
            for metric in self.metrics
        ]

        output_res, _ = evaluator.evaluate(dataset, metrics)
        output_res = output_res['MotChallenge2DBox'][tracker_name][
            'COMBINED_SEQ']['pedestrian']

        if 'HOTA' in self.metrics:
            logger.info('Evaluating HOTA Metrics...')
            eval_results['HOTA'] = np.average(output_res['HOTA']['HOTA'])
            eval_results['AssA'] = np.average(output_res['HOTA']['AssA'])
            eval_results['DetA'] = np.average(output_res['HOTA']['DetA'])

        if 'CLEAR' in self.metrics:
            logger.info('Evaluating CLEAR Metrics...')
            eval_results['MOTA'] = np.average(output_res['CLEAR']['MOTA'])
            eval_results['MOTP'] = np.average(output_res['CLEAR']['MOTP'])
            eval_results['IDSW'] = np.average(output_res['CLEAR']['IDSW'])
            eval_results['TP'] = np.average(output_res['CLEAR']['CLR_TP'])
            eval_results['FP'] = np.average(output_res['CLEAR']['CLR_FP'])
            eval_results['FN'] = np.average(output_res['CLEAR']['CLR_FN'])
            eval_results['Frag'] = np.average(output_res['CLEAR']['Frag'])
            eval_results['MT'] = np.average(output_res['CLEAR']['MT'])
            eval_results['ML'] = np.average(output_res['CLEAR']['ML'])

        if 'Identity' in self.metrics:
            logger.info('Evaluating Identity Metrics...')
            eval_results['IDF1'] = np.average(output_res['Identity']['IDF1'])
            eval_results['IDTP'] = np.average(output_res['Identity']['IDTP'])
            eval_results['IDFN'] = np.average(output_res['Identity']['IDFN'])
            eval_results['IDFP'] = np.average(output_res['Identity']['IDFP'])
            eval_results['IDP'] = np.average(output_res['Identity']['IDP'])
            eval_results['IDR'] = np.average(output_res['Identity']['IDR'])

        return eval_results

    def get_dataset_cfg(self, gt_folder: str, tracker_folder: str,
                        tracker_name: str):
        """Get default configs for trackeval.datasets.MotChallenge2DBox.

        Args:
            gt_folder (str): the name of the GT folder
            tracker_folder (str): the name of the tracker folder

        Returns:
            Dataset Configs for MotChallenge2DBox.
        """
        dataset_config = dict(
            # Location of GT data
            GT_FOLDER=gt_folder,
            # Trackers location
            TRACKERS_FOLDER=tracker_folder,
            # Where to save eval results
            # (if None, same as TRACKERS_FOLDER)
            OUTPUT_FOLDER=None,
            TRACKERS_TO_EVAL=[tracker_name],
            # Option values: ['pedestrian']
            CLASSES_TO_EVAL=['pedestrian'],
            # Option Values: 'MOT15', 'MOT16', 'MOT17', 'MOT20', 'DanceTrack'
            BENCHMARK=self.benchmark,
            # Option Values: 'train', 'test'
            SPLIT_TO_EVAL='val' if self.benchmark == 'DanceTrack' else 'train',
            # Whether tracker input files are zipped
            INPUT_AS_ZIP=False,
            # Whether to print current config
            PRINT_CONFIG=True,
            # Whether to perform preprocessing
            # (never done for MOT15)
            DO_PREPROC=False if self.benchmark == 'MOT15' else True,
            # Tracker files are in
            # TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            TRACKER_SUB_FOLDER='',
            # Output files are saved in
            # OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            OUTPUT_SUB_FOLDER='',
            # Names of trackers to display
            # (if None: TRACKERS_TO_EVAL)
            TRACKER_DISPLAY_NAMES=None,
            # Where seqmaps are found
            # (if None: GT_FOLDER/seqmaps)
            SEQMAP_FOLDER=None,
            # Directly specify seqmap file
            # (if none use seqmap_folder/benchmark-split_to_eval)
            SEQMAP_FILE=None,
            # If not None, specify sequences to eval
            # and their number of timesteps
            SEQ_INFO=None,
            # '{gt_folder}/{seq}.txt'
            GT_LOC_FORMAT='{gt_folder}/{seq}/gt.txt',
            # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
            # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
            # If True, the middle 'benchmark-split' folder is skipped for both.
            SKIP_SPLIT_FOL=True,
        )

        return dataset_config
