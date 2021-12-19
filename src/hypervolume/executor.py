"""Implements the executor of the hypervolume calculator"""

import argparse
import logging
import os

import pandas as pd

from src.hypervolume.calculator import HypervolumeCalculator

logging.getLogger().setLevel(logging.INFO)


class HypervolumeCalculatorApp:

    """Implements the command line application for the hypervolume calculation"""

    def _parse_args(self):
        """Parses and returns the arguments"""
        parser = argparse.ArgumentParser(description="hypervolume calculator app")
        parser.add_argument(
            "-i", "--input-dir", help="sets the path to the directory that contains the input files")
        parser.add_argument(
            "-f", "--input-file-type", help="sets the input file type, 'csv' or 'txt'")
        parser.add_argument(
            "-o", "--output-file-path", help="sets the path to the generated results csv file")
        parser.add_argument(
            "-s", "--model-sense", help="-1 for maximization problem, 1 for minimization problem")
        return parser.parse_args()

    def run(self):
        """Runs the command line application"""
        args = self._parse_args()
        input_dir = args.input_dir
        file_type = args.input_file_type
        model_sense = int(args.model_sense)

        instance_index_2_hyper_volume = {}

        hypervolume_calculator = HypervolumeCalculator()

        for file_path in os.listdir(input_dir):

            if file_type == 'csv':
                df = pd.read_csv(os.path.join(input_dir, file_path))

                # format input from CBSA
                df = df.drop('z_0', axis=1)
                df = df[['z_1', 'z_2', 'connected']]
                instance_index = file_path.split('_')[-2].split('dat')[0]
                
            elif file_type == 'txt':
                df = pd.read_csv(os.path.join(input_dir, file_path), sep=' ', header=None)
                
                # format input from Boland et al. (2015)
                df = df.iloc[:, :3]
                instance_index = file_path.split('out')[0]

            else:
                raise ValueError("unsupported input file type")
            
            hyper_volume = hypervolume_calculator.run(df, model_sense=model_sense)
            instance_index_2_hyper_volume[instance_index] = hyper_volume
   
        hyper_volume_df = pd.DataFrame(
            instance_index_2_hyper_volume.items(), columns=["instance", "hypervolume"])
        
        hyper_volume_df = hyper_volume_df.sort_values(by=["instance"], ascending=True)
        
        print(hyper_volume_df)

        output_file_path = args.output_file_path
        hyper_volume_df.to_csv(os.path.join(output_file_path, "results.csv"))
