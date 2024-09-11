from dataclasses import dataclass
from data_reader import DDAFeature
import deepnovo_config
import logging
import math
import numpy as np
import os
logger = logging.getLogger(__name__)
from deepnovo_config import args
spectrum_name = os.path.splitext(os.path.basename(args.spectrum))[0]
@dataclass
class BeamSearchedSequence:
    sequence: list  # list of aa id
    position_score: list
    score: float  # average by length score


class DenovoWriter(object):
    def __init__(self, denovo_output_file):
        self.output_handle = open(denovo_output_file, 'w')
        header_list = ["sequence",
                       "score",
                       "aa_scores",
                       "spectrum_id"]
        header_row = "\t".join(header_list)
        print(header_row, file=self.output_handle, end='\n')

    def close(self):
        self.output_handle.close()

    def write(self, dda_original_feature: DDAFeature, searched_sequence: BeamSearchedSequence):
        if searched_sequence.sequence:
            sequence = ','.join([deepnovo_config.vocab_reverse[aa_id] for
                                           aa_id in searched_sequence.sequence])
            score = "{:.2f}".format(np.average([(math.exp(x)) for x in searched_sequence.position_score]))
            aa_scores = ','.join(['{0:.2f}'.format(math.exp(x)) for x in searched_sequence.position_score])
            spectrum_id = f"{spectrum_name}:{dda_original_feature.scan}"
        else:
            sequence = ""
            score = ""
            aa_scores = ""
            spectrum_id = ""
        predicted_row = "\t".join([sequence,
                                   score,
                                   aa_scores,
                                   spectrum_id])
        print(predicted_row, file=self.output_handle, end="\n")

    def __del__(self):
        self.close()
