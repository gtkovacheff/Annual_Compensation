import pandas as pd
import numpy as np
import pdb
import seaborn as sns
import collections
import os
import re
import warnings
from statsmodels.tsa.ar_model import AutoReg
from collections import Counter

warnings.filterwarnings("ignore")

# export some pandas options
with open('pandasOptions.env', 'r') as f:
    for line in f:
        pd.set_option(line.split("=")[0], int(line.split("=")[1]))

# get full path of the data directory
data_dir = os.path.abspath('Data')

# get the file names within the data directory
files = os.listdir(data_dir)

# get the full path of the files
files = [os.path.join(data_dir, x) for x in files]


# Data prep function

def prepData(df, year):
    """
    Arguments:
        df: the raw dataframe
        year: the corresponding year, taken from the file name

    The function preps the data based on the year, because for some years the files are in different format.
    From 2012 to 2015 the files are almost the same, just a little data wrangling there.
    For 2016, the column names are totally different + the Compensation column has $ signs and dots.
    For 2017, the column names are different again + City is missing the Compensation column has dots
    """

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
            City="Bloomington",
            FirstName=[x[1][0:-2] for x in df_prepped.Employee.str.split(", ")],
            LastName=[x[0] for x in df_prepped.Employee.str.split(", ")],
            Compensation=[x.replace(",", "") for x in df_prepped.Compensation]
        ).drop(["Employee"], axis=1).rename(columns={"Compensation in " + year: "Compensation",
                                                     "Job Title/Duties": "Job Title"})

    return df_prepped


# for loop through files, read, prep, assign a variable
for x in files:
    print(x, ' file loaded!')
    tmp_df = pd.read_csv(x)  # read the file
    tmp_year = re.search(r'\d{4}', x).group()  # year as variable
    tmp_df['Year'] = tmp_year  # assign year to a new column
    tmp_name = "df_" + tmp_year  # assign name to the dataframe

    globals()[tmp_name] = prepData(tmp_df, tmp_year)  # prep the data

    del tmp_df  # remove tmp_df variable
    del tmp_name  # remove tmp_df name

# concatenate all the data
data = pd.concat([df_2012,
                  df_2013,
                  df_2014,
                  df_2015,
                  df_2016,
                  df_2017])

# reorder columns
data = data[['FirstName', 'LastName', 'Department', 'Job Title', 'City', 'Compensation', 'Year']]

# Convert Compensation and Year to numeric
data['Compensation'] = data.Compensation.astype('float')
data['Year'] = data.Year.astype('int')

years = [2012, 2013, 2014, 2015, 2016, 2017]
new_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]


def forecast(df, column, plot_the_forecast=False):
    """
    Arguments:
        df: the preped dataframe
        column: which column to forecast, options are mean, min and max
        plot the forecast: if True the function will plot the forecast
    Returns:
        df_tmp, which is the forecast from 2018 to 2028
    For the forecast I used AutoReg function from the statsmodels module.
    It basically creates Autoregressive model using Conditional Maximum Likelihood with
    lags = 1, and trend = 'ct' (constant and timetrend)
    Why lags = 1 - because we can observe at most 4 years of data, for particular department,
    not all the departments have data points during all the years. Since this is a big
    problem I had to fill the missing values with backward and forward fill method, which
    basically takes the previous or the next value within the data. This will preserve the trend
    and the estimate would be more accurate. The model is evaluated with two performance metrics:
    AIC - Akaike information criterion, which goal is to be as minimum as possible
    BIC - Bayesian information criterion, the same. We can see that the models are performing well
    given that we estimate 10 new points with 6 points. That's why departments with less than 3 points are being
    filtered for the sake of prediction accuracy
    """

    # create the model
    model = AutoReg(df[column], lags=1, trend='ct').fit()

    # predict next 10 years
    y_hat = model.predict(6, 16)

    # forecast as a dataframe
    y_hat = pd.DataFrame(y_hat).rename(columns={0: column}).assign(
        Year=new_years
    )
    # bind the data with the forecast
    df_tmp = pd.concat([df.loc[:, [column, "Year"]], y_hat])

    # plot the forecast
    if plot_the_forecast:
        g = sns.relplot(x="Year", y=column, kind="line", data=df_tmp)
        g.fig.autofmt_xdate()

    # set the index to be Year, this will be usefull when joining the data
    df_tmp = df_tmp.set_index("Year")

    # define evaluation metrics
    out = 'AIC: {0:0.3f}, BIC: {1:0.3f}'

    # print the evaluation metrics
    # print('Department: {}'.format(df.Department[0]))
    # print(f"The performance of the forecast for {column} is: ", out.format(model.aic, model.bic))

    return df_tmp


def generate_forecast_df(df, print_df=False, filter_above_2017=True):
    """
    Arguments:
        df: DataFrame with the data, which will be forecasted
        print_df: Parameter if the data should be printed
        filter_above_2017: default True, if the user wants to filter the data
    Returns:
        forcasted df for the report number 4

    This function takes the output from the forecast function and
    joins all the data, resulting into a new dataframe
    """

    # Create the forecasts for mean, min and max columns
    forecast_mean = forecast(df, "Compensation_mean")
    forecast_min = forecast(df, "Compensation_min")
    forecast_max = forecast(df, "Compensation_max")

    # Join the data frame
    forecasted_df = forecast_mean.join([forecast_min, forecast_max]).round(2).assign(
        Department=df.iloc[0, 1]
    ).reset_index()

    # Rearrange the data frame
    forecasted_df = forecasted_df[["Year", "Department", "Compensation_mean", "Compensation_min", "Compensation_max"]]

    # Filter the dataFrame from 2018 to 2028
    if filter_above_2017:
        forecasted_df = forecasted_df.loc[forecasted_df.Year > 2017]

    # print the df
    if print_df:
        print("The forecast looks like this: '\n'", forecasted_df)

    return forecasted_df


def generate_report(num, year='all'):
    """
    Arguments:
        num: the number from 1 to 4, from the assignment, which tells us which report to generate
        year: default value 'all', filter the report based on this variable
    The function generates all the 4 reports, then prints them on the console
    """
    if num == 1:
        # assert if the year has the right value
        assert (
                year == 'all' or year in years), 'year variable should be \'all\' or one of the following years: 2012, 2013, 2014, 2015, 2016, 2017'

        # group by the dataframe by Year and aggregate Compensation (mean, min, max) and count the number of people
        data_grouped = data.groupby('Year').agg({'Compensation': ['mean', 'min', 'max'], 'Year': 'count'})
        # rename the column names, as MultiIndex is created
        data_grouped.columns = ['Compensation_mean', 'Compensation_min', 'Compensation_max', 'NumberOfPeople']
        # Round the Compensation_mean column
        data_grouped['Compensation_mean'] = round(data_grouped['Compensation_mean'], 2)
        # adjust the index to start from 0
        data_grouped = data_grouped.reset_index()

        # if variable year != all, filter the data
        if year != 'all':
            data_grouped = data_grouped.loc[data_grouped.Year == year]

        # print the report
        print(data_grouped)

    elif num == 2:
        assert (
                year == 'all' or year in years), 'year variable should be \'all\' or one of the following years: 2012, 2013, 2014, 2015, 2016, 2017'

        # group by the dataframe by Year, Department and aggregate Compensation (mean) and count the number of people
        data_grouped = data.groupby(['Year', 'Department']).agg({'Compensation': ['mean'],
                                                                 'Year': 'count'})
        # rename the column names, as MultiIndex is created
        data_grouped.columns = ['Compensation_mean', 'NumberOfPeople']

        # Round the Compensation_mean column
        data_grouped['Compensation_mean'] = round(data_grouped['Compensation_mean'], 2)

        # adjust the index to start from 0
        data_grouped = data_grouped.reset_index()

        # if variable year != all, filter the data
        if year != 'all':
            data_grouped = data_grouped.loc[data_grouped.Year == year]

        # print the report
        print(data_grouped)

    elif num == 3:
        assert (
                year == 'all' or year in years), 'year variable should be \'all\' or one of the following years: 2012, 2013, 2014, 2015, 2016, 2017'

        # group by the dataframe by Year, Department and Job Title and aggregate Compensation (mean) and count the number of people
        data_grouped = data.groupby(['Year', 'Department', 'Job Title']).agg(
            {'Compensation': ['mean'], 'Year': 'count'})

        # rename the column names, as MultiIndex is created
        data_grouped.columns = ['Compensation_mean', 'NumberOfPeople']

        # Round the Compensation_mean column
        data_grouped['Compensation_mean'] = round(data_grouped['Compensation_mean'], 2)

        # adjust the index to start from 0
        data_grouped = data_grouped.reset_index()

        # if variable year != all, filter the data
        if year != 'all':
            data_grouped = data_grouped.loc[data_grouped.Year == year]

        # print the report
        print(data_grouped)

    elif num == 4:
        # This is different report
        assert (
                year == 'all' or year in new_years), 'year variable should be \'all\' or one of the following years: 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028'

        # group by the dataframe by Year, Department and Job Title and aggregate Compensation (mean) and count the number of people
        data_grouped = data.groupby(['Year', 'Department']).agg({'Compensation': ['mean', 'min', 'max']})

        # rename the column names, as MultiIndex is created
        data_grouped.columns = ['Compensation_mean', 'Compensation_min', 'Compensation_max']

        # Round the Compensation_mean column
        data_grouped['Compensation_mean'] = round(data_grouped['Compensation_mean'], 2)

        # adjust the index to start from 0
        data_grouped = data_grouped.reset_index()

        # create a temp dictionary which will help us to filter the departments with less than 2 points
        tmp_dict = dict(Counter(data_grouped.Department))

        # reformat tmp_dict as dataframe for filter the departments easily
        Department_count = pd.DataFrame(zip([x for x in tmp_dict.keys()], [x for x in tmp_dict.values()])). \
            rename(columns={0: "Department", 1: "count"}). \
            sort_values("count", ascending=False)

        # filter the departments
        Department_over_2 = Department_count.loc[Department_count['count'] > 2, 'Department']

        # create empty dataframe, to which we will append all the predictions
        df = pd.DataFrame()

        # This for loop imputes the missing variabels with the responding values
        # First of all for every department that has atleast 3 points (3 years of data)
        # I observe which are missing, and the add the missing ones, sort them by Year
        # and then fill in the NaN values with forward fill, then with backward fill

        for i in Department_over_2:
            missing_years = list(set(years) - set(data_grouped.loc[data_grouped.Department == i, "Year"]))
            tmp_df_to_be_added = pd.DataFrame(
                [data_grouped.loc[data_grouped.Department == i].iloc[0]] * len(missing_years))
            tmp_df_to_be_added = tmp_df_to_be_added.assign(
                Year=missing_years,
                Compensation_mean=np.NaN,
                Compensation_min=np.NaN,
                Compensation_max=np.NaN
            )

            # first forward fill,then backward fill imputation
            tmp_to_append = pd.concat([data_grouped.loc[data_grouped.Department == i], tmp_df_to_be_added]). \
                sort_values(['Year']). \
                reset_index(drop=True). \
                fillna(method='ffill').fillna(method='bfill')

            # generate the forecast and append
            df = pd.concat([
                df, generate_forecast_df(tmp_to_append)]
            )
        # reset the index
        df.reset_index(drop=True, inplace=True)

        # filter the forecasted dataframe
        if year != 'all':
            df = df.loc[df.Year == year]

        print(df)


def start():
    while True:

        user_input = input("Please select report from 1 to 4: ")

        try:
            user_input_int = int(user_input)

            if user_input_int in [1, 2, 3]:
                user_input_year = input("Please select year for the report: (from 2012 to 2017 or \'all\') ")

                if user_input_year != 'all':
                    user_input_year = int(user_input_year)

                generate_report(user_input_int, user_input_year)

            elif user_input_int == 4:

                user_input_year = input(
                    "Please select year for the forecast: (from 2018 to 2028 or \'all\') ")

                if user_input_year != 'all':
                    user_input_year = int(user_input_year)

                generate_report(user_input_int, user_input_year)

            else:
                print("Not a valid integer for report! Try again. ")
                start()

        except ValueError:
            print("Error! The selected report is not a valid number. Try again. ")

        else:
            yes_no_text = input("Would you like to generate another report? Yes or No ")

            if yes_no_text == "Yes":
                start()
            elif yes_no_text == "No":
                print("Have a successful day!")
                break


start()
