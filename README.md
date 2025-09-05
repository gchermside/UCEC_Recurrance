## UCEC_Recurrence

### Data:
I'm using data from The Cancer Genome Atlas Program on Uterine Corpus Endometrial Carcinoma. This data set contains data for 529 samples of Uterine Corpus Endometrial Carcinoma, including both clinical and genomic data that I am using. The data can be found in the ucec_tcga_pan_can_atlas_2018 folder.

### Pre-processing:
Code can be found in preprocessing.ipynb file. Processed data can be found in the data folder.

Steps:
- clinical and mRNA data imported
- I rename rows of mRNA data that have the same Hugo symbol and Entrez_Gene_Id but different data.
- mRNA data is labeled with Hugo symbol or Entrez_Gene_Id if Hugo symbol is not found.
- patients not is both files are removed.
- Columns are removed where over 99% of the non-null values are the same.
- Columns are removed if over MAX_NULL_VALS percent null values
- extraneous columns are removed
- labels are assigned using NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT. If NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT is NaN DSF_STATUS is used instead. Patients with both traits missing are removed.
- for numerical traits, NaN values are filled in with the median value
- for categorical traits, NaN values are filled in with the mode
- One-Hot Encode categorical columns (drop first to avoid redundancy)

### Training Models:
- right now, this is a bit of a mess. I'm experiementing with ah few different types of models ot train. I've tried 

- xgboost (has overfitting problems).  doing worse than random. 

- logistic regression, kind of the best one so far. It's auc-roc is 0.69 on the testing data, but at least there isn't much overfitting. That being said I don't know how much room for improvment there is with logisitic regression

- random_forest: auc-roc of 0.5811. Extreme overfitting. 

- SVC_no_LASSO is working with an aur-roc of 0.6984 on the testing data. Adding LASSO makes it run a very long time so i have not tried it yet. 
-

### Training:
- somehow my data has lost a feature since some of my models were trained so they need to be re-trained. In the future I should keep this in mind when messing with the pre-processing. 
