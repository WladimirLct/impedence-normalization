import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool

def fix_data(df):
   
    missing_values = df.isnull().sum()

    # Gérer les valeurs manquantes avec forward-fill et backward-fill
    if missing_values.any():
        df.sort_values(['Experiment', 'Well', 'Date', 'Frequency', 'Parameter'], inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
    else:
        print("No missing values found")

    # Pivot du DataFrame pour avoir les paramètres en colonnes
    df = df.pivot_table(
        index=['Experiment', 'Well', 'Date', 'Frequency'],
        columns='Parameter',
        values='Value'
    ).reset_index()

    return df

def process_frequency(args):
    
    frq, well, df_frq_well, fmt = args

    # Convertir la colonne "Date" en datetime
    df_frq_well["Date"] = pd.to_datetime(df_frq_well["Date"], format=fmt)

    # Trier par date
    df_frq_well = df_frq_well.sort_values("Date").reset_index(drop=True)

    # Calculer les valeurs max, min et la valeur à t0 pour AbsZ
    absz_max = df_frq_well["AbsZ"].max()
    absz_min = df_frq_well["AbsZ"].min()
    t0 = df_frq_well["Date"].iloc[0]
    absz_t0 = df_frq_well["AbsZ"].iloc[0]

    # Calculer les deltas de temps à partir de t0
    dt = df_frq_well["Date"] - t0

    # Créer une colonne d'index séquentiel
    df_frq_well["index"] = df_frq_well.index

    # Calculer le temps total en secondes
    total_seconds = dt.dt.total_seconds()

    # Calculer 'hours', 'minutes', 'seconds'
    df_frq_well["hours"] = total_seconds / 3600
    remaining_seconds = total_seconds % 3600
    df_frq_well["minutes"] = remaining_seconds / 60
    df_frq_well["seconds"] = remaining_seconds % 60

    # Calculer les colonnes normalisées
    absz = df_frq_well["AbsZ"].values
    df_frq_well["AbsZ_t0"] = absz - absz_t0
    df_frq_well["AbsZ_max"] = absz / absz_max
    df_frq_well["AbsZ_min_max"] = (absz - absz_min) / (absz_max - absz_min)

    return df_frq_well


def normalize_columns(df, save_path):
   
    df = fix_data(df)
    fmt = '%Y%m%d_%H-%M-%S'
    df["Date"] = df["Date"].dt.strftime(fmt)

    # Réinitialiser l'index pour des indices uniques
    df = df.reset_index(drop=True)
    df['original_index'] = df.index

    # Grouper par 'Frequency' et 'Well'
    freq_well_groups = df.groupby(['Frequency', 'Well'])

    # Préparer une liste de tuples pour le multiprocessing
    freq_well_dfs = [(frq, well, group.copy(), fmt) for (frq, well), group in freq_well_groups]

    # Traitement en parallèle
    with Pool() as pool:
        dfs = pool.map(process_frequency, freq_well_dfs)

    # Concaténer les DataFrames traités et réordonner selon l'index original
    df_processed = pd.concat(dfs)
    df_processed = df_processed.sort_values('original_index')
    df_processed = df_processed.drop(columns=['original_index'])

    # Reconvertir la colonne "Date" en datetime
    df_processed["Date"] = pd.to_datetime(df_processed["Date"], format=fmt)

    # Sauvegarder le DataFrame traité
    df_processed.to_csv(save_path, index=False)

    return df_processed


def compute_custom_norm_treated_control(df, treated_wells, control_wells):
    """
    Compute the custom normalization using treated and control wells.
    For each combination of 'hours' and 'Frequency', the function computes
    the mean AbsZ for the treated and control groups and returns the ratio.
    
    Parameters:
        df (pd.DataFrame): Processed DataFrame containing at least columns:
                           'Well', 'hours', 'Frequency', 'AbsZ'
        treated_wells (list): List of well identifiers for the treated group.
        control_wells (list): List of well identifiers for the control group.
    
    Returns:
        pd.DataFrame: A DataFrame with the computed column 'Znorm', along with
                      grouping variables. Returns None if any group is empty.
    """
    # Filter the DataFrame for treated and control groups
    df_treated = df[df['Well'].isin(treated_wells)]
    df_control = df[df['Well'].isin(control_wells)]
    
    # Check that both groups contain data
    if df_treated.empty or df_control.empty:
        return None

    # Group by 'hours' and 'Frequency' and compute the mean AbsZ values
    treated_mean = df_treated.groupby(['hours', 'Frequency'])['AbsZ'].mean().reset_index()
    control_mean = df_control.groupby(['hours', 'Frequency'])['AbsZ'].mean().reset_index()

    # Merge the two groups on the grouping variables
    merged = pd.merge(treated_mean, control_mean, on=['hours', 'Frequency'], 
                      suffixes=('_treated', '_control'))
    # Compute the normalization value as the ratio
    merged['Znorm'] = merged['AbsZ_treated'] / merged['AbsZ_control']
    # Optionally, add a column to label the normalization
    merged['Well'] = "Treated/Control"
    return merged


import pandas as pd

def custom_normalization(df, norm_method, groups, t0=0):
    """
    Calcule la normalisation personnalisée basée sur les valeurs à t0.

    Pour "CustomNorm1":
      Znorm = (moyenne de AbsZ pour les puits traités à t0) / (moyenne de AbsZ pour les puits de contrôle à t0)

    Pour "CustomNorm2":
      Znorm = (moyenne de AbsZ pour les puits infectés à t0 - moyenne de AbsZ pour les puits de contrôle à t0) /
              (moyenne de AbsZ pour les puits infectés à t0 - moyenne de AbsZ pour les puits traités à t0)

    Args:
      df         : DataFrame contenant les mesures. Doit inclure les colonnes 'hours', 'Frequency', 'Well' et 'AbsZ'.
      norm_method: Méthode de normalisation à appliquer ("CustomNorm1" ou "CustomNorm2").
      groups     : Dictionnaire indiquant, pour chaque type ("infected", "treated", "control"), 
                   la liste des puits à utiliser, par exemple :
                   {
                     "infected": ["Well1", "Well2"],
                     "treated": ["Well3"],
                     "control": ["Well4"]
                   }
      t0         : Temps de référence (par défaut 0).

    Returns:
      Un DataFrame contenant ['hours', 'Frequency', 'Znorm'] calculé sur les données à t0,
      ou None si le calcul ne peut être effectué (par exemple, si un groupe requis est manquant ou vide).
    """
    # Filtrer les données pour t0 (supposons que t0 correspond à hours == 0)
    df_t0 = df[df['hours'] == t0]
    if df_t0.empty:
        return None

    if norm_method == "CustomNorm1":
        # Vérifier que les groupes "treated" et "control" sont définis
        if not groups.get("treated") or not groups.get("control"):
            return None

        df_treated = df_t0[df_t0['Well'].isin(groups["treated"])]
        df_control = df_t0[df_t0['Well'].isin(groups["control"])]

        if df_treated.empty or df_control.empty:
            return None

        treated_mean = df_treated["AbsZ"].mean()
        control_mean = df_control["AbsZ"].mean()

        if control_mean == 0:
            return None

        Znorm = treated_mean / control_mean
        result = df_t0[['hours', 'Frequency']].drop_duplicates().copy()
        result['Znorm'] = Znorm
        return result

    elif norm_method == "CustomNorm2":
        # Vérifier que les trois groupes sont définis
        if not groups.get("infected") or not groups.get("treated") or not groups.get("control"):
            return None

        df_infected = df_t0[df_t0['Well'].isin(groups["infected"])]
        df_treated  = df_t0[df_t0['Well'].isin(groups["treated"])]
        df_control  = df_t0[df_t0['Well'].isin(groups["control"])]

        if df_infected.empty or df_treated.empty or df_control.empty:
            return None

        infected_mean = df_infected["AbsZ"].mean()
        treated_mean  = df_treated["AbsZ"].mean()
        control_mean  = df_control["AbsZ"].mean()

        if infected_mean == treated_mean:
            return None

        Znorm = (infected_mean - control_mean) / (infected_mean - treated_mean)
        result = df_t0[['hours', 'Frequency']].drop_duplicates().copy()
        result['Znorm'] = Znorm
        return result

    else:
        return None

