import pandas as pd
from pyspark.sql import DataFrame
from pyspark.mllib.stat import Statistics

def spark_corr_matrix(df: DataFrame, method: str="pearson", dropna: bool=True) -> pd.DataFrame:
    """
    Computes correlation matrix between all columns in a spark DataFrame
    :param df: Spark DataFrame
    :param method: Correlation method (default pearson correlation)
    :param dropna: Drop nans before calculating correlations
    :return: Correlation matirx as a pandas DataFrame
    """
    if dropna:
        df = df.na.drop()

    col_names = df.columns
    features = df.rdd.map(lambda row: row[0:])

    corr_mat = Statistics.corr(features, method=method)
    corr_df = pd.DataFrame(corr_mat)

    corr_df.index, corr_df.columns = col_names, col_names

    return corr_df

import operator
import random
def balance_classes(_df, label):
    num_classes = _df.groupBy(label).agg(F.countDistinct(label)).count()

    counts = {}
    for i in range(0,num_classes):
        counts[i] = _df.filter(F.col(label)==i).count()

    min_class = min(counts.items(), key=operator.itemgetter(1))[0]
    min_count = counts[min_class]

    df_size = _df.count()

    # create a monotonically increasing id
    _df = _df.withColumn("idx", F.monotonically_increasing_id())

    u = []

    for i in range(0,num_classes):
        uniques = _df.filter(F.col(label)==i).select('idx').distinct().collect()
        vals = [uniques[i][0] for i in range(len(uniques))]
        random.shuffle(vals)
        _u = vals[:min_count]
        u+=_u

    _df = _df.where(F.col("idx").isin(u))
    _df = _df.drop('idx')

    return _df

import sys
def fill_backward(df, id_column, key_column, fill_column):

    # Fill null's with last *non null* value in the window
    ff = df.withColumn(
        'fill_bwd',
        F.first(fill_column, True) # True: fill with last non-null
        .over(
            Window.partitionBy(id_column)
            .orderBy(key_column)
            .rowsBetween(0, sys.maxsize))
        )

    # Drop the old column and rename the new column
    ff_out = ff.drop(fill_column).withColumnRenamed('fill_bwd', fill_column)

    return ff_out

def fill_forward(df, id_column, key_column, fill_column):

    # Fill null's with last *non null* value in the window
    ff = df.withColumn(
        'fill_fwd',
        F.last(fill_column, True) # True: fill with last non-null
        .over(
            Window.partitionBy(id_column)
            .orderBy(key_column)
            .rowsBetween(-sys.maxsize, 0))
        )

    # Drop the old column and rename the new column
    ff_out = ff.drop(fill_column).withColumnRenamed('fill_fwd', fill_column)

    return ff_out
