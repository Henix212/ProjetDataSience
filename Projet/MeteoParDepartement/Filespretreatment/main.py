import pandas as pd
import numpy as np

import csvCleaner.treatment as depTreatment

df = pd.read_csv('Projet/MeteoParDepartement/72/H_72_2000-2009.csv.gz', sep=';')

cleaned_column_df = depTreatment.deleteEmptyColumns(df)
cl_rown_df = depTreatment.dateReformatage(cleaned_column_df)

print(cl_rown_df.head(10))