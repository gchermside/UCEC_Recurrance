# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder



# df = pd.read_csv("ucec_tcga_pan_can_atlas_2018\data_clinical_patient.txt", sep="\t", comment="#", low_memory=False)

# print("nulls", df.isnull().sum())

# MAX_NULL_VALS = 0.3

# og_columns = df.columns

# for col in df.columns:
#     print(f"number of unique values in {col}: {len(df[col].unique())}")  # Show only first 5 unique values
#     print("-" * 50)
#     # if there is only one value for every patient, remove the column
#     if len(df[col].dropna().unique()) == 1:
#         df.drop([col], axis=1, inplace=True)

# # remove the column is over MAX_NULL_VALS percent null values
# df.dropna(axis=1, thresh=len(df) * (1 - MAX_NULL_VALS))

# print("columns removed from clinical ", list(set(og_columns) - set(df.columns)))


# pair_counts = df.groupby(['NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT', 'DFS_STATUS']).size().reset_index(name='Count')

# # Print the pairings and the count
# print(pair_counts)

# # remove non-informational columns
# df.drop(columns=['PATIENT_ID', 'OTHER_PATIENT_ID'])

# # Fill numerical NaNs with median
# numerical_cols = df.select_dtypes(include=['number']).columns
# df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# # Fill categorical NaNs with the most frequent category
# categorical_cols = df.select_dtypes(include=['string']).columns
# # Fill categorical NaNs with the most frequent category (mode)
# for col in categorical_cols:
#     mode_value = df[col].mode()
#     if not mode_value.empty:
#         df[col] = df[col].fillna(mode_value.iloc[0])
#     else:
#         # If mode is empty, fill with a default value (e.g., 'Unknown' or 'NaN')
#         print("mode is none")
#         df[col] = df[col].fillna('Unknown')

# #df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# categorical_columns = ["HISTORY_NEOADJUVANT_TRTYN",
#                         "ETHNICITY",
#                         "HISTORY_NEOADJUVANT_TRTYN",
#                         "ICD_10", 
#                         "PRIOR_DX", 
#                         "RACE",
#                         "RADIATION_THERAPY", 
#                         "IN_PANCANPATHWAYS_FREEZE", 
#                         "PFS_STATUS", 
#                         "GENETIC_ANCESTRY_LABEL"] # do further research on what ICD_10 and ICD_O_3_SITE are

# # One-Hot Encode categorical columns (drop first to avoid redundancy)
# df = pd.get_dummies(df, columns=["HISTORY_NEOADJUVANT_TRTYN"], drop_first=True, dtype=float)
# # comment, right now I'm making eveery column be numerical, may change some to boolean if that would also work

# print("pringin df", df)

# # Define features (drop target variable and any non-relevant columns like patient IDs)

# # # Label Encode ordinal variables
# # encoder = LabelEncoder()
# # df['PATH_T_STAGE'] = encoder.fit_transform(df['PATH_T_STAGE'])










# # encoder = OneHotEncoder(drop="if_binary")  # drop='first' to avoid redundancy
# # encoder.fit(X)

# # encoded_cols = encoder.fit_transform(df[catagorical_columns])
# # df_encoded = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(catagorical_columns))

# # df = pd.concat([df, df_encoded], axis=1).drop(columns=['Cancer_Type'])

# # scaler = StandardScaler()
# # df_scaled = scaler.fit_transform(df)


# # Should remove (because they are post initial treatment)
# # DAYS_LAST_FOLLOWUP
# # this is probbaly fine? DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS
# # ??? Overall Survival Status → OS_STATUS
# # ??? Overall Survival (Months) → OS_MONTHS
# # Disease-specific Survival status → DSS_STATUS
# # Months of disease-specific survival → DSS_MONTHS

# # should remove because, yeah...
# # FORM_COMPLETION_DATE

# # should be used as the labels for the data:
# # DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS
# # PERSON_NEOPLASM_CANCER_STATUS
# # Disease Free Status → DFS_STATUS
# # Disease Free (Months) → DFS_MONTHS

# # DSS_STATUS and DSS_MONTHS

# # Progression Free Status → PFS_STATUS
# # Progress Free Survival (Months) → PFS_MONTHS



