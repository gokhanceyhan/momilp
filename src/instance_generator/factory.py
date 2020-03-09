"""Implements instance generator factory"""

from src.instance_generator.instance import GurobiMomilpInstance, MomilpInstanceParameterSet


class InstanceType:

    """Represents an instance type"""

    GENERAL_MOMILP = "general_momilp"
    KNAPSACK = "knapsack"


class InstanceCreator:

    """Implements instance creator"""

    @staticmethod
    def _create_general_momilp_instances(output_dir, num_instances, **params):
        """Creates general momilp instances"""
        for i in range(num_instances):
            param_2_value = MomilpInstanceParameterSet(**params).to_dict()
            instance = GurobiMomilpInstance(param_2_value, instance_number=i+1, np_rand_num_generator_seed=i)
            instance.write(output_dir)

    @staticmethod
    def create(instance_type, output_dir, num_instances, **params):
        """Creates 'num_instances' many instances of the specified type in the output directory"""
        assert instance_type == InstanceType.GENERAL_MOMILP, "currently only general momilp instances can be generated"
        InstanceCreator._create_general_momilp_instances(output_dir, num_instances, **params)


