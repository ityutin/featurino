import pandas as pd

from featurino.featurino import Featurino

# These features were made just as an example so don't bother yourself with the practical meaning. :)


class Lengths(Featurino):
    @property
    def _prefix(self) -> str:
        return 'lengths'

    def _build_features(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        custom_param = kwargs.get('some_custom_param')
        if custom_param is not None:
            print('{} will be used to calculate Lengths features'.format(custom_param))

        sepal_lengths = df['sepal_length']
        petal_lengths = df['petal_length']

        result_df = df[self._merge_on].copy()
        result_df['sepal_squared'] = sepal_lengths * sepal_lengths
        result_df['petal_squared'] = petal_lengths * petal_lengths

        return result_df


class Widths(Featurino):
    @property
    def _prefix(self) -> str:
        return 'widths'

    def _build_features(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        sepal_lengths = df['sepal_width']
        petal_lengths = df['petal_width']

        result_df = df[self._merge_on].copy()
        result_df['sepal_plus_petal'] = sepal_lengths + petal_lengths

        return result_df
