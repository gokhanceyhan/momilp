"""Implements an algorithm to calculate the dominated hypervolume by a nondominated set 

This is restricted to the bi-objective case. The nondominated set can include edges."""


class HypervolumeCalculator:

    """Implements the hypervolume calculation"""

    def _preprocess(self, df):
        
        # rename the columns
        df.columns = ["z1", "z2", "connected"]

        # sort in the ascending order of first 'z1', then 'z2'
        df = df.sort_values(by=["z1", "z2"])

        assert df['z1'].max() > df['z1'].min() and df['z2'].max() > df['z2'].min(), "the set contains dominated points"
        # apply min-max scaling
        df['z1'] = (df['z1'] - df['z1'].min()) / (df['z1'].max() - df['z1'].min())
        df['z2'] = (df['z2'] - df['z2'].min()) / (df['z2'].max() - df['z2'].min())

        # reindex
        df = df.reset_index(drop=True)

        return df


    def run(self, df, model_sense=-1):
        """Calculates and returns the hypervolume
        
        The hypervolume value represents the proportion of the 2D unit square dominated by the 
        standardized nondominated set.
        
        df: The dataframe must contain the extreme nondominated points and 
        must have 3 columns: First two columns show objective values of points and the last column takes 
        binary values, 1 if the successive points are connected, 0 otherwise.

        model_sense: -1 for maximization problem and 1 for minimization problem
        
        """
        if len(df) == 0:
            return 0
        if len(df) == 1:
            return 1
        
        df = self._preprocess(df)

        hyper_volume = 0.0

        for point_index, point in df.iterrows():
            if point_index == 0:
                continue
            
            previous_point = df.iloc[point_index - 1, :]
            hyper_volume += (point['z1'] - previous_point['z1']) * point['z2']

            if previous_point['connected'] == 0:
                continue
            
            hyper_volume += (point['z1'] - previous_point['z1']) * (previous_point['z2'] - point['z2']) / 2

        return hyper_volume if model_sense == -1 else 1 - hyper_volume
