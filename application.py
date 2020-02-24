import pandas as pd
import numpy as np
import pdb
import seaborn as sns
import collections
import os
import re


#export some pandas options
with open('pandasOptions.env', 'r') as f:
    for line in f:
        pd.set_option(line.split("=")[0], int(line.split("=")[1]))



#get full path of the data directory
data_dir = os.path.abspath('Data')

#get the file names within the data directory
files = os.listdir(data_dir)

#get the full path of the files
files = [os.path.join(data_dir, x) for x in files]


def prepData(df, year):
    if int(year) < 2016:
        df_prepped = df.assign(
            FirstName=[x[1][0:-2] for x in df.Employee.str.split(", ")],
            LastName=[x[0] for x in df.Employee.str.split(", ")],
            City=df.City.str.title()
        ).drop(["Employee"], axis=1).rename(columns={"Compensation in " + year: "Compensation",
                                                     "Job Title/Duties": "Job Title"})

    elif int(year) == 2016:
        df_prepped = df.rename(columns={"total_compensation": "Compensation",
                                        "job_title": "Job Title",
                                        "first_name": "Employee",
                                        "Textbox14": "City"
                                        })

        df_prepped = df_prepped.assign(
            FirstName=[x[1][0:-2] for x in df_prepped.Employee.str.split(", ")],
            LastName=[x[0] for x in df_prepped.Employee.str.split(", ")],
            Compensation=[x[1:].replace(",", "") for x in df_prepped.Compensation],
            City=df_prepped.City.str.title()
        ).drop(["Textbox6", "Employee"], axis=1)

    elif int(year) == 2017:
        df_prepped = df.rename(columns={"Salary": "Compensation",
                                        "Name": "Employee"
                                        })

        df_prepped = df_prepped.assign(
            City="No information",
            FirstName=[x[1][0:-2] for x in df_prepped.Employee.str.split(", ")],
            LastName=[x[0] for x in df_prepped.Employee.str.split(", ")],
            Compensation=[x.replace(",", "") for x in df_prepped.Compensation]
        ).drop(["Employee"], axis=1).rename(columns={"Compensation in " + year: "Compensation",
                                                     "Job Title/Duties": "Job Title"})

    return df_prepped

for x in files:
    print(x, ' file loaded!')
    tmp_df = pd.read_csv(x)
    tmp_year = re.search(r'\d{4}', x).group()
    tmp_df['Year'] = tmp_year
    tmp_name = "df_" + tmp_year

    globals()[tmp_name] = prepData(tmp_df, tmp_year)

    del tmp_df
    del tmp_name

#Data preparation
len(df_2012.columns)
len(df_2013.columns)
len(df_2014.columns)
len(df_2015.columns)
len(df_2016.columns)
len(df_2017.columns)
