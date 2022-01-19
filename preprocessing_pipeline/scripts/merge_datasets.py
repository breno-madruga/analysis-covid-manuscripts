##############################################################################################
############## Merging the COVID-19 manuscripts dataset with the 2020 InCites JCR ############
##############################################################################################

########################################################################
# Importing required libraries.
import csv, pandas as pd, numpy as np
########################################################################

########################################################################
# 1. Getting and preprocessing the datasets
########################################################################

########################################################################
# 1.1. Production data
########################################################################

# Importing the data.
df_data = pd.read_csv("preprocessing_pipeline/data/raw/manuscript_covid_processed.csv", delimiter=",", header=0, dtype=object)

# Changing the type of some columns of Production data.
df_data.citation_num = df_data.citation_num.astype(np.float32)
df_data.ref_count = df_data.ref_count.astype(np.float32)
df_data.publication_date = pd.to_datetime(df_data.publication_date, format="%Y-%m-%d")

# Converting from the "str" type to the "list" type of some columns of Production data.
df_data.replace({np.nan: None}, inplace=True)
df_data.auth_keywords = df_data.auth_keywords.apply(lambda x: eval(x) if x else None)
df_data.index_terms = df_data.index_terms.apply(lambda x: eval(x) if x else None)
df_data.affiliations = df_data.affiliations.apply(lambda x: eval(x) if x else None)
df_data.subject_areas = df_data.subject_areas.apply(lambda x: eval(x) if x else None)
df_data.authors = df_data.authors.apply(lambda x: eval(x) if x else None)
df_data.author_affil = df_data.author_affil.apply(lambda x: eval(x) if x else None)
df_data.references = df_data.references.apply(lambda x: eval(x) if x else None)

########################################################################
# 1.2. InCites Journal Citation Reports (Web of Science)
########################################################################

# Importing the impact factor data.
df_jcr = pd.read_csv("preprocessing_pipeline/data/raw/jcr_2020_processed.csv", delimiter=",", header=0)

# Changing the invalid values to "None".
df_jcr.replace({np.nan: None}, inplace=True)

########################################################################
# 2. Merging the production dataframe with JCR dataset
########################################################################

# Merging in order to get the "impact factor" for each journal article.
columns = df_jcr.columns.tolist()
columns.remove("issn")
columns.remove("e_issn")
df_data = df_data.reindex(columns=[*df_data.columns.tolist(), *columns])
df_data.loc[df_data.issn.notnull(), columns] = df_data.issn[
    df_data.issn.notnull()].apply(lambda x: df_jcr.loc[
        df_jcr.issn.isin(x.split()) | df_jcr.e_issn.isin(x.split()), columns].iloc[0] \
            if df_jcr.issn.isin(x.split()).any() or df_jcr.e_issn.isin(x.split()).any() \
                else pd.Series(dict(zip(columns, [None] * len(columns)))))

# Updating the "impact factor" to the journal articles without it.
df_data.loc[df_data.impact_factor_2020.isnull() & df_data.source_type.isin(["j", "d"]),
    "impact_factor_2020"] = 0

# Updating the "label" to the journal articles without it.
df_data.loc[df_data.label.isnull() & df_data.source_type.isin(["j", "d"]), "label"] = "E"

# Changing the invalid values to "None".
df_data.replace({np.nan: None}, inplace=True)

########################################################################
# 2. Saving the data
########################################################################

# Saving the final dataset.
df_data.to_csv("preprocessing_pipeline/data/prepared/final_manuscript_covid.csv", index=False, quoting=csv.QUOTE_ALL)