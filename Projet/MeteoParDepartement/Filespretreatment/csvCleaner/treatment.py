import pandas as pd

def deleteEmptyColumns(dataFrame: pd.DataFrame, seuil: float = 0.25) -> pd.DataFrame:
    """
    Supprime les colonnes :
    - entièrement vides (100% de NaN)
    - ou avec plus de `seuil` % de valeurs NaN.

    :param dataFrame: Le DataFrame à modifier.
    :param seuil: Seuil maximum de NaN autorisé (entre 0 et 1). Défaut : 0.25 (25%)
    :return: Le DataFrame nettoyé.
    """
    # 1. Delete 100% NaN columns
    df_cleaned = dataFrame.dropna(axis=1, how='all')

    # 2. Delete columns with a `seuil` of NaN values
    seuil_valide = df_cleaned.shape[0] * seuil
    df_final = df_cleaned.loc[:, df_cleaned.isna().sum() <= seuil_valide]
    
    # 3. Delete columns with only one unique value
    df_final = df_final.drop(columns=df_final.columns[df_final.nunique(dropna=False) <= 1], axis=1)
    
    return df_final

def deleteEmptyRows(dataFrame: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes qui contiennent au moins une valeur NaN.

    :param dataFrame: Le DataFrame à modifier.
    :return: Le DataFrame nettoyé.
    """
    return dataFrame.dropna(axis=0, how='any').reset_index(drop=True)

import pandas as pd

def dateReformatage(dataFrame: pd.DataFrame) -> pd.DataFrame:
    """
    Reformate les dates du DataFrame en format datetime Pandas.

    :param dataFrame: Le DataFrame à modifier.
    :return: Le DataFrame avec les dates reformattées.
    """
    dataFrame["AAAAMMJJHH"] = pd.to_datetime(
        dataFrame["AAAAMMJJHH"].astype(str),
        format='%Y%m%d%H',
        errors='coerce' 
    )
    return dataFrame
    