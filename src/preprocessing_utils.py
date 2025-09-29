import pandas as pd
import numpy as np
import joblib
from sklearn.utils import resample
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import pandas as pd
import numpy as np

def generate_recurrence_labels(treatment_file, status_file, clinical_file):
    """
    Generates a pd.Series of recurrence labels for all patients.
    
    Label rules:
     1 (recurred): 
        * ANATOMIC_TREATMENT_SITE = "Local Recurrence" or "Distant Recurrence"
        * REGIMEN_INDICATION = "Recurrence"
        * STATUS = "Locoregional Recurrence"
        * NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT = "Yes"
     0 (no recurrence): 
        * NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT = "No"
        * AND no other columns show recurrence
     None (unknown/ambiguous): 
        * All other patients
        * Patients with conflicting signals (e.g., "No" in clinical but positive elsewhere)
    """
    
    # --- Load data ---
    df_treatment = pd.read_csv(treatment_file, sep="\t", comment="#", low_memory=False)
    df_status = pd.read_csv(status_file, sep="\t", comment="#", low_memory=False)
    df_clinical = pd.read_csv(clinical_file, sep="\t", comment="#", low_memory=False)
    
    # Ensure PATIENT_ID is a column
    if df_treatment.index.name == "PATIENT_ID":
        df_treatment = df_treatment.reset_index()
    if df_clinical.index.name == "PATIENT_ID":
        df_clinical = df_clinical.reset_index()
    
    # --- Set of patient IDs labeled as recurrence ---
    recur_patients = set()
    
    # From treatment file
    treatment_mask = df_treatment["ANATOMIC_TREATMENT_SITE"].isin(["Local Recurrence", "Distant Recurrence"])
    regimen_mask = df_treatment["REGIMEN_INDICATION"] == "Recurrence"
    recur_patients.update(df_treatment.loc[treatment_mask | regimen_mask, "PATIENT_ID"].unique())
    
    # From status file
    status_mask = df_status["STATUS"].astype(str).str.strip() == "Locoregional Recurrence"
    recur_patients.update(df_status.loc[status_mask, "PATIENT_ID"].unique())
    
    # From clinical file
    clinical_yes_mask = df_clinical["NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT"].astype(str).str.strip().str.lower() == "yes"
    recur_patients.update(df_clinical.loc[clinical_yes_mask, "PATIENT_ID"].unique())
    
    # --- Set of patients labeled as no recurrence ---
    clinical_no_mask = df_clinical["NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT"].astype(str).str.strip().str.lower() == "no"
    no_recur_patients = set(df_clinical.loc[clinical_no_mask, "PATIENT_ID"].unique())
    
    # --- Combine all patient IDs ---
    all_patients = set(df_clinical["PATIENT_ID"]) | set(df_treatment["PATIENT_ID"]) | set(df_status["PATIENT_ID"])
    
    # --- Assign labels ---
    labels = {}
    for pid in all_patients:
        if pid in recur_patients and pid in no_recur_patients:
            # conflict: one source says no, another says yes
            labels[pid] = None
        elif pid in recur_patients:
            labels[pid] = 1
        elif pid in no_recur_patients:
            labels[pid] = 0
        else:
            labels[pid] = None
    
    # Return as pd.Series
    label_series = pd.Series(labels, name="Recurrence_Label")
    label_series.index.name = "PATIENT_ID"
    
    return label_series


def drop_patients_missing_data(clinical_df, mrna_df, labels):
    """
    Drops patients not shared across clinical_df, mrna_df, and labels.
    Also drops patients missing labeling data (None or NaN).
    
    Returns:
        clinical_df_clean, mrna_df_clean, labels_clean
    """
    # Step 1: Find shared patient IDs (preserve order)
    shared_patients = clinical_df.index.intersection(mrna_df.index).intersection(labels.index)
    
    # Step 2: Subset all three to shared patients, in the same order
    clinical_df_clean = clinical_df.loc[shared_patients].copy()
    mrna_df_clean = mrna_df.loc[shared_patients].copy()
    labels_clean = labels.loc[shared_patients].copy()
    
    # Step 3: Drop patients with missing labels (None/NaN)
    valid_patients = labels_clean[labels_clean.notna()].index
    clinical_df_clean = clinical_df_clean.loc[valid_patients]
    mrna_df_clean = mrna_df_clean.loc[valid_patients]
    labels_clean = labels_clean.loc[valid_patients]
    
    # Step 4: Sanity checks
    assert clinical_df_clean.shape[0] == mrna_df_clean.shape[0] == labels_clean.shape[0], \
        "Dataframes have different number of patients after cleaning"
    assert not labels_clean.isna().any(), "Found unlabeled patient after cleaning"
    assert clinical_df_clean.index.equals(mrna_df_clean.index) and clinical_df_clean.index.equals(labels_clean.index), \
        "Indexes are not aligned"
    
    return clinical_df_clean, mrna_df_clean, labels_clean
    

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
    
    def _drop_highly_uniform_columns(self, X):
        """Identifies highly uniform columns (> threshold same value)."""
        cols_to_drop = []
        for col in X.columns:
            non_na_values = X[col].dropna()
            if not non_na_values.empty:
                top_freq = non_na_values.value_counts(normalize=True).iloc[0]
                if top_freq > self.uniform_thresh:
                    cols_to_drop.append(col)
        return cols_to_drop
    
    def fit(self, X, y=None):
        # --- Step 1. Drop specified columns
        removed = [c for c in self.cols_to_remove if c in X.columns]
        
        # --- Step 2. Drop columns with too many nulls
        thresh = len(X) * (1 - self.max_null_frac)
        high_null_cols = [c for c in X.columns if X[c].isna().sum() > len(X) - thresh]
        removed.extend(high_null_cols)
        
        # --- Step 3. Drop highly uniform columns
        uniform_cols = self._drop_highly_uniform_columns(X)
        removed.extend(uniform_cols)

        # --- Step 4. Drop all identified columns
        X = X.drop(columns=removed, errors="ignore")
        
        # --- Step 5. Fill NaNs
        # Numerical → median
        numeric_cols = X.select_dtypes(include=['number']).columns
        self.num_fill_values_ = X[numeric_cols].median()
        X[numeric_cols] = X[numeric_cols].fillna(self.num_fill_values_)
        
        # Categorical → mode
        cat_cols = [c for c in self.categorical_cols if c in X.columns]
        self.cat_fill_values_ = {c: X[c].mode().iloc[0] for c in cat_cols if not X[c].dropna().empty}
        for c, mode_val in self.cat_fill_values_.items():
            X[c] = X[c].fillna(mode_val)
        
        # --- Step 6. One-hot encode categorical
        X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        # Save results
        self.removed_cols_ = removed
        self.columns_ = X_enc.columns.tolist()
        
        return self
    
    def transform(self, X):
        # Drop removed cols
        X = X.drop(columns=[c for c in self.removed_cols_ if c in X.columns], errors="ignore")
        
        # --- Fill NaNs using training fill values
        numeric_cols = X.select_dtypes(include=['number']).columns
        for c in numeric_cols:
            if c in self.num_fill_values_:
                X[c] = X[c].fillna(self.num_fill_values_[c])
        
        cat_cols = [c for c in self.categorical_cols if c in X.columns]
        for c in cat_cols:
            if c in self.cat_fill_values_:
                X[c] = X[c].fillna(self.cat_fill_values_[c])
        
        # One-hot encode
        X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        # Reindex to training columns (fill missing with 0)
        X_enc = X_enc.reindex(columns=self.columns_, fill_value=0)
        
        return X_enc


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

    def _drop_highly_uniform_columns(self, X):
        """Identify and drop highly uniform columns (> threshold)."""
        cols_to_drop = []
        for col in X.columns:
            non_na_values = X[col].dropna()
            if not non_na_values.empty:
                top_freq = non_na_values.value_counts(normalize=True).iloc[0]
                if top_freq > self.uniform_thresh:
                    cols_to_drop.append(col)
        return X.drop(columns=cols_to_drop), cols_to_drop

    def _prune_correlated_features(self, X):
        """Prune correlated features above correlation threshold."""
        corr_matrix = X.corr().abs()
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
                    worst_gene = min(non_lit_neighbors, key=lambda n: X[n].var())
                else:
                    break
            else:
                ties = [g for g, d in degrees.items() if d == degrees[worst_gene]]
                if len(ties) > 1:
                    worst_gene = min(ties, key=lambda g: X[g].var())
            
            genes_to_remove.add(worst_gene)
            genes_to_keep.remove(worst_gene)

        return X[list(genes_to_keep)], genes_to_remove

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
              f"({self.stability_threshold*100:.0f}% stability threshold)"
              f"Used {self.n_boots} boots")

        self.selection_freq_ = selection_freq
        return X[selected_features], list(set(X.columns) - set(selected_features))

    def fit(self, X, y=None):
        removed = []

        # Step 1. Drop columns with too many nulls
        high_null_cols = [c for c in X.columns if X[c].isna().sum() > len(X) * self.max_null_frac]
        removed.extend(high_null_cols)
        X_temp = X.drop(columns=high_null_cols, errors="ignore")
        print(f"Dropped {len(high_null_cols)} columns with >{self.max_null_frac*100}% nulls")

        # Step 2. Drop highly uniform columns
        X_temp, uniform_cols = self._drop_highly_uniform_columns(X_temp)
        removed.extend(uniform_cols)
        print(f"Dropped {len(uniform_cols)} highly uniform columns")

        # Step 3. Fill NaNs with median
        self.medians_ = X_temp.median().to_dict()
        X_temp = X_temp.fillna(self.medians_)

        # Step 4. Variance filter
        low_var_cols = [c for c in X_temp.columns if X_temp[c].var() < self.var_thresh]
        X_temp = X_temp.drop(columns=low_var_cols, errors="ignore")
        removed.extend(low_var_cols)
        print(f"Dropped {len(low_var_cols)} low variance columns (<{self.var_thresh})")

        # Step 5. Prune correlated features
        if self.re_run_pruning:
            X_temp, correlated_genes = self._prune_correlated_features(X_temp)
            joblib.dump(correlated_genes, "../new_data/correlated_genes_to_remove.pkl")
            removed.extend(correlated_genes)
            print(f"Dropped {len(correlated_genes)} correlated genes (>{self.corr_thresh} correlation)")
        else:
            correlated_genes = joblib.load("../new_data/correlated_genes_to_remove.pkl")
            X_temp = X_temp.drop(columns=correlated_genes, errors="ignore")
            removed.extend(correlated_genes)

        # Step 6. Stability-based selection
        if self.use_stability_selection:
            if y is None:
                raise ValueError("y labels required for stability-based feature selection")
            X_temp, dropped_stability = self._stability_feature_selection(X_temp, y)
            removed.extend(dropped_stability)

        # Save final state
        self.removed_cols_ = list(set(removed))
        self.columns_ = X_temp.columns.tolist()

        return self

    def transform(self, X):
        # Drop known removed cols
        X = X.drop(columns=[c for c in self.removed_cols_ if c in X.columns], errors="ignore")
        print("dropping", len(self.removed_cols_), "columns total")

        # Fill NaNs with median
        X = X.fillna(self.medians_)

        # Check column alignment
        missing = set(self.columns_) - set(X.columns)
        extra = set(X.columns) - set(self.columns_)
        if missing or extra:
            raise ValueError(
                f"Column mismatch! Missing: {missing}, Extra: {extra}, "
                f"{len(missing)} missing, {len(extra)} extra"
            )

        # Reorder X to match training column order
        X = X[self.columns_]

        return X

class MrnaPreprocessorWrapper(MrnaPreprocessor, BaseEstimator, TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")  # make sure fit() was called
        return np.array(self.columns_)  # or self.cleaned_columns_ if you store them


class ClinicalPreprocessorWrapper(ClinicalPreprocessor, BaseEstimator, TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")  # make sure fit() was called
        return np.array(self.columns_)  # or self.cleaned_columns_ if you store them


