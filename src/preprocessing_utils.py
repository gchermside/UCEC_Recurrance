import config

import pandas as pd
import numpy as np
import joblib
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.utils import resample
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def assign_labels(clinical_df):
    '''given the clinical dataframe, returns the corresposnding labels, 
    assigning 1 for recurrance, 0 for no recurrance, 
    and None if the patient has no recurrence information. 
    Currently uses NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT to identify recurrance.
    If NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT is NaN, uses DSF_STATUS to save the label.'''
    labels = []
    for _, row in clinical_df.iterrows():
        if row['NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT'] == 'Yes':
            labels.append(1)
        elif row['NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT'] == 'No':
            labels.append(0)
        elif pd.isna(row['NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT']):
            if row['DFS_STATUS'] == '1:Recurred/Progressed':
                labels.append(1)
            elif row['DFS_STATUS'] == '0:DiseaseFree':
                labels.append(0)
            else:
                labels.append(None)
    return pd.Series(labels, index=clinical_df.index)


def drop_patients_missing_data(clinical_df, mrna_df, labels):
    '''Drops patients from both dataframes that are not present in the other dataframe. 
    Drops patients who are missing labeling data used to define recurrence.
    Returns the cleaned dataframes and labels.'''
    # Find patient IDs not shared between the two dataframes:
    clinical_not_in_mrna = set(clinical_df.index) - set(mrna_df.index)
    mrna_not_in_clinical = set(mrna_df.index) - set(clinical_df.index)
    # There are 2 patients ('TCGA-EY-A1GJ', 'TCGA-AP-A0LQ') in the clinical data that are not in the mRNA data.
    clinical_df = clinical_df.drop(index=clinical_not_in_mrna)
    mrna_df = mrna_df.drop(index=mrna_not_in_clinical)
    labels = labels.drop(index=clinical_not_in_mrna)
    labels = labels.drop(index=mrna_not_in_clinical)
    assert clinical_df.shape[0] == mrna_df.shape[0] == labels.shape[0], "Dataframes have different number of patients after cleaning"

    # Now drop patients missing labeling data used to define recurrence:
    patients_no_label = labels[labels.isna()].index
    clinical_df = clinical_df.drop(index=patients_no_label)
    mrna_df = mrna_df.drop(index=patients_no_label)
    labels = labels.drop(index=patients_no_label)
    assert not labels.isna().any(), "Found unlabeled patient after cleaning"

    return clinical_df, mrna_df, labels

class ClinicalPreprocessor:
    def __init__(self, cols_to_remove, categorical_cols, max_null_frac=0.3, uniform_thresh=0.99):
        self.cols_to_remove = cols_to_remove
        self.categorical_cols = categorical_cols
        self.max_null_frac = max_null_frac
        self.uniform_thresh = uniform_thresh
        
        # Saved state after fit
        self.removed_cols_ = []
        self.columns_ = None  # final column order
        self.num_fill_values_ = {}
        self.cat_fill_values_ = {}
    
    def _drop_highly_uniform_columns(self, df):
        """Identifies highly uniform columns (> threshold same value)."""
        cols_to_drop = []
        for col in df.columns:
            non_na_values = df[col].dropna()
            if not non_na_values.empty:
                top_freq = non_na_values.value_counts(normalize=True).iloc[0]
                if top_freq > self.uniform_thresh:
                    cols_to_drop.append(col)
        return cols_to_drop
    
    def fit(self, df):
        # --- Step 1. Drop specified columns
        removed = [c for c in self.cols_to_remove if c in df.columns]
        
        # --- Step 2. Drop columns with too many nulls
        thresh = len(df) * (1 - self.max_null_frac)
        high_null_cols = [c for c in df.columns if df[c].isna().sum() > len(df) - thresh]
        removed.extend(high_null_cols)
        
        # --- Step 3. Drop highly uniform columns
        uniform_cols = self._drop_highly_uniform_columns(df)
        removed.extend(uniform_cols)

        # --- Step 4. Drop all identified columns
        df = df.drop(columns=removed, errors="ignore")
        
        # --- Step 5. Fill NaNs
        # Numerical → median
        numeric_cols = df.select_dtypes(include=['number']).columns
        self.num_fill_values_ = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(self.num_fill_values_)
        
        # Categorical → mode
        cat_cols = [c for c in self.categorical_cols if c in df.columns]
        self.cat_fill_values_ = {c: df[c].mode().iloc[0] for c in cat_cols if not df[c].dropna().empty}
        for c, mode_val in self.cat_fill_values_.items():
            df[c] = df[c].fillna(mode_val)
        
        # --- Step 6. One-hot encode categorical
        df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        # Save results
        self.removed_cols_ = removed
        self.columns_ = df_enc.columns.tolist()
        
        return self
    
    def transform(self, df):
        # Drop removed cols
        df = df.drop(columns=[c for c in self.removed_cols_ if c in df.columns], errors="ignore")
        
        # --- Fill NaNs using training fill values
        numeric_cols = df.select_dtypes(include=['number']).columns
        for c in numeric_cols:
            if c in self.num_fill_values_:
                df[c] = df[c].fillna(self.num_fill_values_[c])
        
        cat_cols = [c for c in self.categorical_cols if c in df.columns]
        for c in cat_cols:
            if c in self.cat_fill_values_:
                df[c] = df[c].fillna(self.cat_fill_values_[c])
        
        # One-hot encode
        df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        # Reindex to training columns (fill missing with 0)
        df_enc = df_enc.reindex(columns=self.columns_, fill_value=0)
        
        return df_enc


class MrnaPreprocessor:
    def __init__(self,
                 max_null_frac=0.3,
                 uniform_thresh=0.99,
                 corr_thresh=0.9,
                 var_thresh=1e-5,
                 re_run_pruning=True, # this is so that when I'm testing stability selection I can skip pruning (takes a while to run)
                 literature_genes=set(),
                 use_stability_selection=True,
                 n_boots=100,
                 fpr_alpha=0.05,
                 stability_threshold=0.8,
                 random_state=42):

        self.max_null_frac = max_null_frac
        self.uniform_thresh = uniform_thresh
        self.corr_thresh = corr_thresh
        self.var_thresh = var_thresh
        self.re_run_pruning = re_run_pruning
        self.literature_genes = literature_genes

        # Stability selection params
        self.use_stability_selection = use_stability_selection
        self.n_boots = n_boots
        self.fpr_alpha = fpr_alpha
        self.stability_threshold = stability_threshold
        self.random_state = random_state

        # Saved state after fit
        self.removed_cols_ = []
        self.medians_ = {}
        self.columns_ = None
        self.selection_freq_ = None

    def _drop_highly_uniform_columns(self, df):
        """Identify and drop highly uniform columns (> threshold)."""
        cols_to_drop = []
        for col in df.columns:
            non_na_values = df[col].dropna()
            if not non_na_values.empty:
                top_freq = non_na_values.value_counts(normalize=True).iloc[0]
                if top_freq > self.uniform_thresh:
                    cols_to_drop.append(col)
        return df.drop(columns=cols_to_drop), cols_to_drop

    def _prune_correlated_features(self, df):
        """Prune correlated features above correlation threshold."""
        corr_matrix = df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)

        high_corr_map = {
            gene: set(corr_matrix.index[corr_matrix.loc[gene] >= self.corr_thresh])
            for gene in corr_matrix.columns
        }

        genes_to_keep = set(corr_matrix.columns)
        genes_to_remove = set()

        while True:
            correlated_genes = {g: nbrs for g, nbrs in high_corr_map.items() if nbrs & genes_to_keep}
            if not correlated_genes:
                break

            degrees = {g: len(nbrs & genes_to_keep) for g, nbrs in correlated_genes.items() if g in genes_to_keep}
            if not degrees:
                break

            worst_gene = max(degrees, key=lambda g: degrees[g])

            if worst_gene in self.literature_genes:
                neighbors = correlated_genes[worst_gene] & genes_to_keep
                non_lit_neighbors = [n for n in neighbors if n not in self.literature_genes]
                if non_lit_neighbors:
                    worst_gene = min(non_lit_neighbors, key=lambda n: df[n].var())
                else:
                    break
            else:
                ties = [g for g, d in degrees.items() if d == degrees[worst_gene]]
                if len(ties) > 1:
                    worst_gene = min(ties, key=lambda g: df[g].var())
            
            genes_to_remove.add(worst_gene)
            genes_to_keep.remove(worst_gene)

        return df[list(genes_to_keep)], genes_to_remove

    def _stability_feature_selection(self, X, y):
        """Bootstrap stability-based feature selection (Jessie’s approach)."""
        np.random.seed(self.random_state)
        feature_counts = pd.Series(0, index=X.columns)

        for i in range(self.n_boots):
            X_boot, y_boot = resample(
                X, y,
                stratify=y,
                n_samples=len(y),
                replace=True,
                random_state=self.random_state+i
            )
            selector = SelectFpr(score_func=f_classif, alpha=self.fpr_alpha)
            selector.fit(X_boot, y_boot)
            selected = X_boot.columns[selector.get_support()]
            feature_counts[selected] += 1

        selection_freq = feature_counts / self.n_boots
        selected_features = selection_freq[selection_freq >= self.stability_threshold].index.tolist()

        print(f"Stability selection: kept {len(selected_features)} / {X.shape[1]} features "
              f"({self.stability_threshold*100:.0f}% stability threshold)")

        self.selection_freq_ = selection_freq
        return X[selected_features], list(set(X.columns) - set(selected_features))

    def fit(self, df, y=None):
        removed = []

        # Step 1. Drop columns with too many nulls
        high_null_cols = [c for c in df.columns if df[c].isna().sum() > len(df) * self.max_null_frac]
        removed.extend(high_null_cols)
        df_temp = df.drop(columns=high_null_cols, errors="ignore")
        print(f"Dropped {len(high_null_cols)} columns with >{self.max_null_frac*100}% nulls")

        # Step 2. Drop highly uniform columns
        df_temp, uniform_cols = self._drop_highly_uniform_columns(df_temp)
        removed.extend(uniform_cols)
        print(f"Dropped {len(uniform_cols)} highly uniform columns")

        # Step 3. Fill NaNs with median
        self.medians_ = df_temp.median().to_dict()
        df_temp = df_temp.fillna(self.medians_)

        # Step 4. Variance filter
        low_var_cols = [c for c in df_temp.columns if df_temp[c].var() < self.var_thresh]
        df_temp = df_temp.drop(columns=low_var_cols, errors="ignore")
        removed.extend(low_var_cols)
        print(f"Dropped {len(low_var_cols)} low variance columns (<{self.var_thresh})")

        # Step 5. Prune correlated features
        if self.re_run_pruning:
            df_temp, correlated_genes = self._prune_correlated_features(df_temp)
            joblib.dump(correlated_genes, "../new_data/correlated_genes_to_remove.pkl")
            removed.extend(correlated_genes)
            print(f"Dropped {len(correlated_genes)} correlated genes (>{self.corr_thresh} correlation)")
        else:
            correlated_genes = joblib.load("../new_data/correlated_genes_to_remove.pkl")
            df_temp = df_temp.drop(columns=correlated_genes, errors="ignore")
            removed.extend(correlated_genes)

        # Step 6. Stability-based selection
        if self.use_stability_selection:
            if y is None:
                raise ValueError("y labels required for stability-based feature selection")
            df_temp, dropped_stability = self._stability_feature_selection(df_temp, y)
            removed.extend(dropped_stability)

        # Save final state
        self.removed_cols_ = list(set(removed))
        self.columns_ = df_temp.columns.tolist()

        return self

    def transform(self, df):
        # Drop known removed cols
        df = df.drop(columns=[c for c in self.removed_cols_ if c in df.columns], errors="ignore")

        # Fill NaNs with median
        df = df.fillna(self.medians_)

        # Check column alignment
        missing = set(self.columns_) - set(df.columns)
        extra = set(df.columns) - set(self.columns_)
        if missing or extra:
            raise ValueError(
                f"Column mismatch! Missing: {missing}, Extra: {extra}, "
                f"{len(missing)} missing, {len(extra)} extra"
            )

        # Reorder df to match training column order
        df = df[self.columns_]

        return df

class MrnaPreprocessorWrapper(MrnaPreprocessor, BaseEstimator, TransformerMixin):
    pass

class ClinicalPreprocessorWrapper(ClinicalPreprocessor, BaseEstimator, TransformerMixin):
    pass
