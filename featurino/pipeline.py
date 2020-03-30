import pandas as pd
from typing import Type

from featurino.featurino import Featurino


class FeaturinoPipeline:
    FORCE_RELOAD_KEY = 'force_reload'
    """
    Easy way to combine results from multiple Featurino objects.

    Init parameters
    ----------
    main_df: pd.DataFrame
        Raw dataframe that will be used to calculate features.
    args:
        args that will be passed to init method for each Featurino
    kwargs:
        kwargs that will be passed to init method for each Featurino
    """
    def __init__(self, main_df: pd.DataFrame, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs

        self._piped_types = {}
        # just a copy of initial dataframe
        self._start_df = main_df.copy()
        # we will accumulate all the features in this dataframe
        self._all_features_df = self._start_df
        self._force_reload = False

    def set_force_reload(self, value: bool):
        """
        Changes default value of _force_reload flag.
        We may want to set it once and use it for all Featurinos.
        """
        self._force_reload = value
        return self

    def pipe(self, featurino_type: Type[Featurino], *args, **kwargs):
        """
        Gets features from the provided Featurino and merges them to _all_features_df

        Parameters
        ----------
        featurino_type: Subclass type of custom Featurino
            Type of Featurino to instantiate and get features from.
        args:
            args to pass to the featurino's build_features method.
        kwargs:
            kwargs to pass to the featurino's build_features method.

        Returns
        -------
        Self, so you can pipe as many Featurinos as you want one by one.
        """
        if self._piped_types.get(featurino_type):
            # Once we calculated features for a featurino type - we don't want to do it again.
            raise ValueError("You've tried to pipe the featurino {} twice".format(featurino_type.__name__))
        else:
            featurino = featurino_type(*self._init_args, **self._init_kwargs)

        # if force_reload flag is not passed - we will use saved _force_reload flag.
        kwargs[FeaturinoPipeline.FORCE_RELOAD_KEY] = kwargs.get(FeaturinoPipeline.FORCE_RELOAD_KEY, self._force_reload)
        self._all_features_df = featurino.build_features(df=self._all_features_df, *args, **kwargs)
        return self

    def features_df(self) -> pd.DataFrame:
        result_df = self._all_features_df
        # once user requests all the features we reset the internal df
        # so the next time user calls pipe we don't have any feature duplicates
        self._all_features_df = self._start_df
        self._force_reload = False
        return result_df
