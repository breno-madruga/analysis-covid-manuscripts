##############################################################################################
############ Preprocessing the COVID-19 manuscripts and 2020 InCites JCR datasets ############
##############################################################################################

########################################################################
# Importing required libraries.
import csv, pandas as pd, numpy as np
########################################################################

########################################################################
# 1. Defining the required functions
########################################################################

# Defining the function to normalize the country from the "author_affil" and "affiliations" features.
def normalize_countries(row):
    if row.affiliations:
        for pos, affil in enumerate(row.affiliations):
            if not affil["country"] and affil["affiliation"]:
                if df_countries.country[[str(c).lower().strip() in affil["affiliation"].lower().strip()
                                         for c in df_countries.country]].size > 0:
                    row.affiliations[pos]["country"] = df_countries.country[
                        [str(c).lower().strip() in affil["affiliation"].lower().strip()
                         for c in df_countries.country]].iloc[0]
                elif df_countries.country[[str(c) in affil["affiliation"]
                                           for c in df_countries.acronym]].size > 0:
                    row.affiliations[pos]["country"] = df_countries.country[
                        [str(c) in affil["affiliation"] for c in df_countries.acronym]
                    ].iloc[0]
            elif affil["country"]:
                temp = df_countries.country[df_countries.acronym.isin(df_countries.acronym[
                            [str(c).lower().strip() in affil["country"].lower().strip()
                             for c in df_countries.country]].values)]
                if temp.size > 0:
                    row.affiliations[pos]["country"] = temp.iloc[0]

    if row.author_affil:
        for pos, author in enumerate(row.author_affil):
            if not author["country"] and author["affiliation"]:
                if df_countries.country[[str(c).lower().strip() in author["affiliation"].lower().strip()
                                         for c in df_countries.country]].size > 0:
                    row.author_affil[pos]["country"] = df_countries.country[
                        [str(c).lower().strip() in author["affiliation"].lower().strip()
                         for c in df_countries.country]].iloc[0]
                elif df_countries.country[[str(c) in author["affiliation"]
                                           for c in df_countries.acronym]].size > 0:
                    row.author_affil[pos]["country"] = df_countries.country[
                        [str(c) in author["affiliation"] for c in df_countries.acronym]
                    ].iloc[0]
            elif author["country"]:
                    temp = df_countries.country[df_countries.acronym.isin(df_countries.acronym[
                            [str(c).lower().strip() in author["country"].lower().strip()
                             for c in df_countries.country]].values)]
                    if temp.size > 0:
                        row.author_affil[pos]["country"] = temp.iloc[0]

    return row

########################################################################
# 2. Getting and preprocessing the datasets
########################################################################

########################################################################
# 2.1. Countries data
########################################################################

# Importing the countries data.
df_countries = pd.read_csv("suplementary_data/countries.csv", delimiter=";", header=0, index_col=None)

########################################################################
# 2.2. Production data
########################################################################

# Importing the data.
df_data = pd.read_csv("preprocessing_pipeline/data/raw/final_covid_19.csv", delimiter=",", header=0, dtype=object)

# Changing the type of some columns of Production data.
df_data.citation_num = df_data.citation_num.astype(np.float32)
df_data.ref_count = df_data.ref_count.astype(np.float32)
df_data.publication_date = pd.to_datetime(df_data.publication_date, format="%Y-%m-%d")

# Removing unnecessary columns of Production data.
df_data.drop(axis=1, columns="doi", inplace=True)

# Converting from the "str" type to the "list" type of some columns of Production data.
df_data.replace({np.nan: None}, inplace=True)
df_data.auth_keywords = df_data.auth_keywords.apply(lambda x: eval(x) if x else None)
df_data.index_terms = df_data.index_terms.apply(lambda x: eval(x) if x else None)
df_data.affiliations = df_data.affiliations.apply(lambda x: eval(x) if x else None)
df_data.subject_areas = df_data.subject_areas.apply(lambda x: eval(x) if x else None)
df_data.authors = df_data.authors.apply(lambda x: eval(x) if x else None)
df_data.author_affil = df_data.author_affil.apply(lambda x: eval(x) if x else None)
df_data.references = df_data.references.apply(lambda x: eval(x) if x else None)

# Applying the "normalize_countries" function to the data.
df_data.loc[df_data.affiliations.notnull() | df_data.author_affil.notnull(),
    ["affiliations", "author_affil"]] = df_data.loc[
        df_data.affiliations.notnull() | df_data.author_affil.notnull(),
        ["affiliations", "author_affil"]].apply(normalize_countries, axis=1)

# Extracting the countries from the feature "author_affil".
df_data.loc[df_data.author_affil.notnull(), "countries"] = df_data.author_affil[
    df_data.author_affil.notnull()].apply(lambda x: tuple(
        [affil["country"] for affil in x if affil["country"]]))

# Defining "None" to the papers without their affiliations.
df_data.loc[df_data.countries.notnull(), "countries"] = df_data.countries.apply(
    lambda x: x if x else None)

# Extracting the countries from the feature "affiliations".
df_data.loc[df_data.affiliations.notnull() & df_data.countries.isnull(), "countries"] = \
df_data.affiliations[df_data.affiliations.notnull() & df_data.countries.isnull()].apply(
    lambda x: tuple([affil["country"] for affil in x if affil["country"]]))

# Defining "None" to the papers without their affiliations.
df_data.loc[df_data.countries.notnull(), "countries"] = df_data.countries.apply(
    lambda x: x if x else None)

# Defining the "num_brazilian" column for the number of brazilian people into the papers.
df_data["num_brazilian"] = [
    len([country for country in countries if country == "Brazil"]) if type(countries) == tuple else None
    for countries in df_data.countries.values
]

# Defining the "year" column from the "publication_date" column.
df_data["year"] = pd.DatetimeIndex(df_data.publication_date).year

# Defining the "month" column from the "publication_date" column.
df_data["month"] = pd.DatetimeIndex(df_data.publication_date).month

# Defining the "preprint" type.
df_data.loc[~df_data.data_source.isin(["Scopus", "PubMed"]), "production_type"] = "Preprint"
df_data.loc[~df_data.data_source.isin(["Scopus", "PubMed"]), "source_type"] = "pp"

# Defining the "Others" type to PubMed database.
df_data.loc[df_data.data_source == "PubMed", "production_type"] = "Others"
df_data.loc[df_data.data_source == "PubMed", "source_type"] = "o"

# Checking if there are duplicates by "id" column.
print("Number of duplicated records:", df_data[df_data.id.duplicated()].id.size)

# Removing the duplicated records.
df_data.drop_duplicates("id", inplace=True)

# Checking if there are duplicates by "id" column.
print("Number of duplicated records:", df_data[df_data.id.duplicated(keep=False)].id.size)

########################################################################
# 2.3. InCites Journal Citation Reports (Web of Science)
########################################################################

# Importing the impact factor data.
df_jcr = pd.read_csv("suplementary_data/incites_jcr_wos_2020.csv", delimiter=",", header=0)

# Changing the invalid values to "None".
df_jcr.replace({np.nan: None, "-": None, "****-****": None,
                "Not Available": None, "N/A": None, "n/a": None}, inplace=True)

# Normalizing the "impact_factor_2020" feature.
df_jcr.impact_factor_2020.fillna(0, inplace=True)
df_jcr.loc[df_jcr.impact_factor_2020.notnull(), "impact_factor_2020"] = \
df_jcr.loc[df_jcr.impact_factor_2020.notnull(), "impact_factor_2020"].apply(lambda x: float(x))

# Creating the "label" feature from the discretization of the impact factor.
df_jcr.loc[df_jcr.impact_factor_2020.notnull(), "label"] = \
df_jcr.loc[df_jcr.impact_factor_2020.notnull(), "impact_factor_2020"].apply(
    lambda x: "E" if x == 0 or not x else "D" if x < 1 else "C" \
        if 1 <= x <= 3 else "B" if 3 < x < 5 else "A")

# Removing the hyphen from ISSN and eISSN.
df_jcr.issn = df_jcr.issn.apply(lambda x: x.replace("-", "") if x else None)
df_jcr.e_issn = df_jcr.e_issn.apply(lambda x: x.replace("-", "") if x else None)

# Checking if there are duplicates.
print("Number of duplicated records:", df_jcr[df_jcr.duplicated()].journal_name.size)

# Removing the records duplicated.
df_jcr.drop_duplicates(inplace=True)

# Checking if there are duplicates.
print("Number of duplicated records:", df_jcr[df_jcr.duplicated()].journal_name.size)

########################################################################
# 3. Saving the data
########################################################################

# Saving the Production data.
df_data.to_csv("preprocessing_pipeline/data/raw/manuscript_covid_processed.csv", index=False, quoting=csv.QUOTE_ALL)

# Saving the 2020 InCites JCR data.
df_jcr.to_csv("preprocessing_pipeline/data/raw/jcr_2020_processed.csv", index=False, quoting=csv.QUOTE_ALL)