"""Implements instance generator factory"""

import logging
import os

from src.instance_generator.instance import GurobiMomilpInstance, MomilpFileInstanceData, MomilpRandomInstanceData, \
    MomilpInstanceParameterSet


class InstanceType:

    """Represents an instance type"""

    GENERAL_MOMILP = "general_momilp"
    KNAPSACK = "knapsack"


class InstanceCreator:

    """Implements instance creator"""

    _DATA_FILE_DIR_CONFIGURATION_NAME = "data_file_dir"
    _INSTANCE_NAME_FORMAT = \
        "momilp_{num_objs}obj_{num_constraints}con_{num_integer_vars}int_{num_binary_vars}bin_{instance_name}.lp"
    _RANDOM_INSTANCE_NAME_FORMAT = "ins_{instance_number}"

    @staticmethod
    def _create_general_momilp_instances(output_dir, data_file_dir=None, num_instances=0, **params):
        """Creates general momilp instances"""
        if data_file_dir:
            data_files = [os.path.join(data_file_dir, f) for f in os.listdir(data_file_dir) if f.endswith(".txt")]
            logging.info(
                "creating '%d' instances at '%s' directory from the data files in the '%s' directory..." % (
                    len(data_files), output_dir, data_file_dir))
            for file in data_files:
                instance_name = str(file).split("/")[-1].split(".")[0]
                param_2_value = MomilpInstanceParameterSet(**params).to_dict()
                data = MomilpFileInstanceData(file, param_2_value)
                instance = GurobiMomilpInstance(data, param_2_value)
                instance_file_name = InstanceCreator._create_instance_file_name(instance_name, **param_2_value)
                path = os.path.join(output_dir, instance_file_name)
                instance.write(path)
            return
        logging.info("creating '%d' random instances at '%s' directory..." % (num_instances, output_dir))
        for i in range(num_instances):
            instance_name = InstanceCreator._RANDOM_INSTANCE_NAME_FORMAT.format(instance_number=i+1)
            param_2_value = MomilpInstanceParameterSet(**params).to_dict()
            data = MomilpRandomInstanceData(param_2_value, np_rand_num_generator_seed=i)
            instance = GurobiMomilpInstance(data, param_2_value)
            instance_file_name = InstanceCreator._create_instance_file_name(instance_name, **param_2_value)
            path = os.path.join(output_dir, instance_file_name)
            instance.write(path)

    @staticmethod
    def _create_instance_file_name(instance_name, **param_2_value):
        param_2_value["instance_name"] = instance_name
        return InstanceCreator._INSTANCE_NAME_FORMAT.format(**param_2_value)

    @staticmethod
    def create(instance_type, output_dir, data_file_dir=None, num_instances=None, **params):
        """Creates 'num_instances' many instances of the specified type in the output directory"""
        assert instance_type == InstanceType.GENERAL_MOMILP, "currently only general momilp instances can be generated"
        InstanceCreator._create_general_momilp_instances(
            output_dir, data_file_dir=data_file_dir, num_instances=num_instances, **params)


