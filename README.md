# ⚛️ featurino
Lib for fast&amp;easy manual feature engineering

## Why

This tiny lib was created during one of the data competitions and helped to make feature engineering quicker and more reliable. It's a good fit for small projects as well.

## How

It helps you to split features into logical parts and then cache them on disk. Then you will be able to experiment with different combinations of the features without need to rerun your heavy calculations. All the features will be prefixed thus easily distinguishable. 

## Example

Note: See demo/Featurino_demo.ipynb for more details.

```
from featurino.featurino import Featurino
```

You should create Featurino subclass for every group of features.


```
for example 

class DateFeatures(Featurino):
    pass

class TrendsFeatures(Featurino):
    pass
    
class PurchasesFeatures(Featurino):
    pass
```

You initialize each Featurino subclass with the next params:

```
def __init__(self,
             data_dir_path: str, # where to save cached features
             merge_on: List[str], # which columns will be used to merge features between multiple Featurino
             force_reload: bool = False, # whether you want to recalculate the features
             df_cache: DfCache = CsvDfCache()): # logic how to save and load features 
```

Each subclass must override two internal methods

```
@property
def _prefix(self) -> str:
    # all your features returned from _build_features method 
    # will be prefixed with the string your provide here
    return %_features_prefix_%

def _build_features(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    # here you take df as input. It may look like a skeleton for your features dataset
    # e.g. containing all the information required to build features on top of ( ids, dates etc )
    # also you can provide any parameters you like using args and kwargs.
    # the return dataframe must contain merge_on columns you passed in init method. 
    pass
```  

When all the features are implemented, you should use FeaturinoPipeline to combine features whatever you like.

```
pipeline = FeaturinoPipeline(main_df=data, 
                             merge_on=['id'], 
                             data_dir_path='features')
```
To calculate features you need to provide a type of Featurino subclass and then pass any parameters that your Featurino subclass expects to use in _build_features method.
```
pipeline.pipe(DateFeatures, date_param1=param1, date_param2=param2) \
        .pipe(TrendsFeatures, trend_param1=trend_param1) \
        .pipe(PurchasesFeatures) \
        .features_df()
```
If you just want to get features from cache - use only types.
```
pipeline.pipe(DateFeatures) \
        .pipe(TrendsFeatures) \
        .pipe(PurchasesFeatures) \
        .features_df()
```
For the hands-on experience see demo/Featurino_demo.ipynb.

### Prerequisites

Pandas is required for using the lib. 

The lib was written using Python 3.7, yet I currently don't know the minimum Python version required.  

```
pandas
python 3.7 ( the true minimum version is not yet confirmed ) 
``` 