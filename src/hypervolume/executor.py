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
        description = """
        A hypervolume calculator app. Takes a nondominated set and returns the dominated area of the unit square that 
        represents the scaled objective function space. The input file names should be either instance_number.csv 
        instance_number.txt, as in from Boland et al. (2015) or as generated from our momilp solver app.

        The file must contain 3 columns: First two columns show objective values of points and the last column takes 
        binary values, 1 if the successive points are connected, 0 otherwise.
        """
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            "-i", "--input-dir", help="sets the path to the directory that contains the input files")
        parser.add_argument(
            "-f", "--input-file-type", help="sets the input file type, 'csv' or 'txt'")
        parser.add_argument(
            "-o", "--output-file-path", help="sets the path to the generated results csv file")
        parser.add_argument(
            "-s", "--model-sense", help="-1 for maximization problem, 1 for minimization problem")
        parser.add_argument(
            "-b", "--boland-nd-set", help="sets if the input file is taken from Boland et al. (2015)", 
            action="store_true")
        parser.add_argument(
            "-m", "--momilp-solver-nd-set", help="sets if the input file is generated from our momilp solver app", 
            action="store_true")
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

                # format input from momilp solver app
                if args.momilp_solver_nd_set:
                    df = df.drop('z_0', axis=1)
                    df = df[['z_1', 'z_2', 'connected']]
                    instance_index = int(file_path.split('_')[-2].split('dat')[0])
                else:
                    instance_index = int(file_path.split('.csv')[0])
                
            elif file_type == 'txt':
                df = pd.read_csv(os.path.join(input_dir, file_path), sep=' ', header=None)
                
                # format input from Boland et al. (2015)
                if args.boland_nd_set:
                    df = df.iloc[:, :3]
                    instance_index = int(file_path.split('out')[0])
                else:
                    instance_index = int(file_path.split('.txt')[0])

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
