import pandas as pd
import os
from abc import abstractmethod
from typing import List

from featurino.df_cache import DfCache, CsvDfCache


class Featurino:
    """
    This lib was written during one of the tabular competitions.
    I found it very useful for fast and flexible feature engineering.

    Featurino encapsulates logic of feature calculations using
    a provided dataframe with raw data. Split your features into
    logical parts and make one subclass of Featurino for each of them.
    Then you will be able to combine them together whatever you like.
    Every featurino stores calculated features on disk
    for quick and easy access to them in the future.

    Init parameters
    ----------
    data_dir_path: str
        Where to save your features.
    merge_on: list of str
        What columns to get from the raw df to uniquely represent feature rows.
        These columns will be used to merge features of multiple Featurinos.
    force_reload:
        Whether you want to recalculate features despite the already existing cached file.
    """

    def __init__(self,
                 data_dir_path: str,
                 merge_on: List[str],
                 force_reload: bool = False,
                 df_cache: DfCache = CsvDfCache()):
        self._data_dir_path = data_dir_path
        self._merge_on = merge_on
        self._force_reload = force_reload
        self._df_cache = df_cache
        # df with already calculated features
        self._cached_df = None

    @property
    @abstractmethod
    def _prefix(self) -> str:
        """
        Provide a prefix for columns to uniquely represent
        features from this Featurino subclass.

        Final column name will look like this:
        %prefix%__%feature_name%

        Returns
        -------
        String prefix for your features.
        """
        pass

    def build_features(self,
                       df: pd.DataFrame,
                       *args,
                       **kwargs) -> pd.DataFrame:
        """
        Takes data in main_df and returns features calculated out of it.

        Parameters
        ----------
        df: pd.DataFrame
            Raw data for your problem
        args:
            custom arguments you may pass to your subclass
        kwargs
            custom named arguments you may pass to your subclass

        Returns
        -------
        Dataframe with features for your problem.
        """
        force_reload = kwargs.get('force_reload', self._force_reload)

        can_load_from_disk = self._is_cache_on_disk
        must_build = force_reload or not can_load_from_disk

        if must_build:
            result_df = self._build_features(df=df, *args, **kwargs)
            result_df = self._prefix_df_cols(df=result_df)
            self._save_to_disk(df=result_df)
            self._log(text='Features has been calculated.')
        elif self._is_cache_in_memory:
            result_df = self._cached_df
            self._log(text='Using in-memory features.')
        else:
            result_df = self._load_from_disk()
            self._log(text='Loaded from disk.')

        self._cached_df = result_df

        result_df = df.merge(result_df, on=self._merge_on)

        return result_df

    def _log(self, text: str):
        print('{}: {}'.format(type(self).__name__, text))

    @abstractmethod
    def _build_features(self,
                        df: pd.DataFrame,
                        *args,
                        **kwargs) -> pd.DataFrame:
        """
        Provide feature calculation logic in a subclass.
        Note: It must contain _merge_on columns.
        """
        pass

    def _prefix_df_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds prefixes to all the cols except the those that are used for merge.
        """
        no_merge_col = [col for col in df.columns if not col in self._merge_on]
        prefixed_columns = ['{}__'.format(self._prefix) + col for col in no_merge_col]
        cols_rename_map = dict(zip(no_merge_col, prefixed_columns))
        result_df = df.rename(columns=cols_rename_map)

        return result_df

    def _load_from_disk(self):
        assert self._is_cache_on_disk, "Can't find cache file on disk"
        result_df = self._df_cache.load(path=self._cache_path)
        return result_df

    def _save_to_disk(self, df: pd.DataFrame):
        assert Featurino._check_df(df=df), "The provided dataframe is empty or None, build your features first."

        features_dir = os.path.dirname(os.path.normpath(self._cache_path))
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)

        self._df_cache.save(df=df, path=self._cache_path)

    @property
    def _is_cache_in_memory(self) -> bool:
        return self._check_df(df=self._cached_df)

    @property
    def _is_cache_on_disk(self) -> bool:
        result = os.path.isfile(self._cache_path)
        return result

    @staticmethod
    def _check_df(df: pd.DataFrame):
        return df is not None and not df.empty

    @property
    def _cache_path(self) -> str:
        return '{}/{}_features_cache.csv'.format(self._data_dir_path, self._prefix)
