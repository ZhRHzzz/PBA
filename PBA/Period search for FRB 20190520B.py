"""
The script performs a period search on the FRB 20190520B data, with a search range from 1 to 2024 days.
Before running the script, please first calculate the FAP distribution for FRB 20190520B
(using the script "Calculate the FAP distribution (FRB 20190520B)").
The file reading and storage paths can be adjusted by the user.
"""


import numpy as np
import math
from scipy.stats import norm
from scipy.stats import binom
import pandas as pd
import os
import re


def read_all_files(folder_path):
    """
    Read all data files in the folder.
    :return: Return the data.
    """
    # Create a variable to store the data.
    pcumu_data = np.array([])
    # Get all the .npy filenames in the folder.
    file_names = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
    # Define a function to extract the numerical part from the filename.
    def extract_numbers(file_name):
        match = re.search(r'\d+(\.\d+)?', file_name)
        if match:
            return float(match.group())
        else:
            return float('inf')

    # Sort based on the numerical part in the filenames.
    file_names.sort(key=extract_numbers)
    # Traverse and read .npy files.
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path)
        # Process the data.
        pcumu_data = np.append(pcumu_data, data)

    return pcumu_data


def phase_folding(T, JD, d):
    """
    A function used to calculate the corresponding phases of the burst under different central phases for a specific test period.
    :param T: Test period, with the input value being a number.
    :param JD: Burst time (MJD), with the input value being a 2D array of size (n*1), or a 1D array containing n numbers.
    :param d: Burst data volume, with the input value being a number, i.e., len(JD).
    :return: Output an array with ncp rows and d columns, where rows represent different phase centers, and columns represent the phase values of each burst relative to the phase center.
    """
    dp = np.zeros((ncp, d))
    for i in range(ncp):  # Cycle the phase for different center phases.
        for j in range(d):  # Calculate the phase for each burst.
            deltap = (JD[j] % T) / T - (0.05 * i)
            # 0.05 is the step size set for searching the active center phase.
            # This calculation corresponds to the formula in the article:
            # MJD = CT + P'T_1, where the resulting value represents the distance between the burst phase and the center phase.
            # The next step is to convert the obtained distance into a phase range of 0-1 (e.g., -0.1 corresponds to 0.9), and store the values in an array.
            if deltap < 0.:
                dp[i, j] = 1 - abs(deltap)

            else:
                dp[i, j] = deltap
    return dp


def obs_distribution(T, Start, Dur):
    """
    A function used to calculate the distribution of observation times with respect to phase for a specific period T and different center phases.
    :param T: The test period, with the input value being a single number.
    :param Start: The observation start time, with the input value being an (n, 1) ndarray array.
    :param Dur: The duration of each observation, in hours, with the input value being an (n, 1) ndarray array.
    :return: Outputs an array with ncp rows and n_o columns, where rows represent different phase centers and columns represent the phase size of each observation activity relative to the phase center.
    """
    obs_dis = np.zeros((ncp, nop))
    bp_obs = phase_folding(T, Start, n_o)  # Obtain the phase for different centers within this period, with each observation start time (start) corresponding to the phase difference.
    obs_p = Dur / T  # Calculate the phase occupied by the observation duration (dur) within this period.
    for i in range(ncp):  # The observation start phase differs for each phase center calculated.
        for j in range(n_o):  # Perform calculations for each observation activity.
            while obs_p[j] >= 1:
                obs_dis[i, :] = obs_dis[i, :] + int(obs_p[j]) * T / nop
                obs_p[j] = obs_p[j] - int(obs_p[j])
            for k in range(nop):  # For each phase bin that stores the observation time.
                crit_p = (k + 1) / nop  # Define the critical phase as 0.05 * k.
                if bp_obs[i, j] < crit_p:  # If the observation start time is within the range of this bin.
                    delta = crit_p - bp_obs[i, j]  # Calculate the difference between the observation start phase and the critical phase.

                    if obs_p[j] <= delta:  # If the phase occupied by the observation time is smaller than the calculated difference, the entire observation time of the activity is accumulated into the (k+1)-th bin (where k=0 for the first bin).
                        obs_dis[i, k] += Dur[j]  # Note that when judging the condition, phase is used, while time is added during the accumulation.

                    elif obs_p[j] > delta:  # If the phase occupied by the observation time is greater than the phase difference, the time corresponding to the phase difference is accumulated into the current bin, and the remaining time needs to be carried over to subsequent bins.
                        obs_dis[i, k] += delta * T
                        obph = obs_p[j] - delta  # "obph" refers to the phase occupied by the remaining observation time after subtracting the time already added to the bins.

                        while obph > 1 / nop:  # If this value is still greater than 0.05 (i.e., the phase width of one bin), then 0.05 phase of the observation time is added to the next bin, and the remaining time is recalculated.

                            if k < nop - 1:  # When k is less than 19, the index of the next bin is k+1
                                k = k + 1
                                obs_dis[i, k] += (1 / nop) * T
                            else:  # When k=19 , the next bin should be k=0
                                k = 0
                                obs_dis[i, k] += (1 / nop) * T
                            obph = obph - (1 / nop)  # The phase occupied by the remaining observation time.
                        if obph <= 1 / nop:  # If obph is smaller than the phase width of one bin, the remaining time should be fully accumulated into the next bin.
                            if k < nop - 1:
                                obs_dis[i, k + 1] = obs_dis[i, k + 1] + obph * T
                            else:
                                obs_dis[i, 0] = obs_dis[i, 0] + obph * T

                    break

    return obs_dis


def False_alarm_Probability(F, N):
    """

    :param F: The binomial probability calculated from the binomial distribution, with the input value being a single number.
    :param N: The number of independent experiments, Ni (independent frequency) = fNy * T_1 * number of search center frequencies * number of searched phase widths, fNy = 1/2 * 1/FRB accuracy (10ms)
    :return: FAP
    """

    if N * F / 2 < 0.01:
        FAP = N * F
    elif ((N * F) ** 2) / 3 < 0.01:
        FAP = N * F - ((N * F) ** 2) / 2
    elif ((N * F) ** 3) / 4 < 0.01:
        FAP = N * F - ((N * F) ** 2) / 2 + ((N * F) ** 3) / 6
    elif ((N * F) ** 4) / 5 < 0.01:
        FAP = N * F - ((N * F) ** 2) / 2 + ((N * F) ** 3) / 6 - ((N * F) ** 4) / 24
    else:
        FAP = 1 - math.pow((1 - F), N)

    return FAP


def period_search(test_start_d, test_end_d, obs_dis):
    """

    Period search process.
    :param test_start_d: Search initial time.
    :param test_end_d: Search end time.
    :param obs_dis: Observation time phase distribution for different search periods.
    :return: sigma matrix, bp, p_cumu matrix, T：Corresponding search time series.
    """
    bp = np.zeros((nT, ncp, n_d))  # Used to store the phases corresponding to burst events under different test periods/frequencies and different central active phases.
    count = np.zeros((nT, ncp, npw))  # Used to store the number of bursts occurring within the active region under different test periods/frequencies, central active phases, and active region widths.
    r = np.zeros((nT, ncp, npw))  # Used to store the sum of the count rates corresponding to bursts occurring within the active region under different test periods/frequencies, central active phases, and active region widths.
    F = np.zeros((nT, ncp, npw))  # Used to store the calculated binomial distribution probabilities under different test periods/frequencies, central active phases, and active region widths.
    sigma = np.zeros((nT, ncp, npw))  # Used to store sigma.
    p_cumu = np.zeros((nT, ncp, npw))  # Used to store the cumulative probability of the binomial distribution.

    fre = np.linspace(1 / test_start_d, 1 / test_end_d, nT)  # Set the test frequency. nT=1000
    T = 1. / fre  # Test period = 1/Test frequency
    total_rate = int(Rate_tot + 0.5)
    for t in range(nT):  # Different test periods/frequencies
        bp[t, :, :] = phase_folding(T[t], MJD, n_d)  # Calculate the test period and the phase corresponding to the occurrence time of each burst at different center phases (with the active center phase as 0 phase).
        for i in range(ncp):  # Different center phases.
            for j in range(npw):  # Different active region widths.
                hw = aphw[j]  # Half width.
                filter1 = (bp[t, i, :] < hw) | (bp[t, i, :] > 1 - hw)  # Filter out the bursts located within the active region.
                count[t, i, j] = len(bp[t, i, filter1])  # The number of bursts within the active region.
                r[t, i, j] = np.sum(Rate[filter1])  # The sum of the burst counting rates within the active region.
                obs_hw = np.sum(obs_dis[t, i, 0:j + 1]) + np.sum(obs_dis[t, i, -(j + 1):])  # Total observation duration of the active region.
                p = obs_hw / obs_tot  # Under the random hypothesis, the probability that a single observed burst occurs within the active region is the percentage of the observation duration of the active region relative to the total observation duration.
                non_act_burst_num = total_rate - r[t, i, j]
                if p == 0:
                    non_act_burst_num = total_rate

                P_binom = binom.cdf(non_act_burst_num, total_rate, 1 - p)
                p_cumu[t, i, j] = P_binom
                FAP = False_alarm_Probability(P_binom, Ni)

                # Calculate sigma using the standard Gaussian curve.
                # if FAP <= 0:
                #     FAP = 1e-322
                #
                # F[t, i, j] = FAP
                # if FAP < 1e-16:
                #     sigma[t, i, j] = norm.isf(FAP)
                # else:
                #     FAP = 1. - FAP
                #     x, sigma[t, i, j] = norm.interval(FAP)
                #     if sigma[t, i, j] == float('inf'):
                #         sigma[t, i, j] = 8.21

                # Calculate sigma using the Gaussian fit of the FAP distribution obtained.
                if FAP <= 0:
                    FAP = 1e-322

                F[t, i, j] = FAP
                if FAP < 1e-16:
                    up_lim_confiden = norm.isf(FAP, loc=mu_FAP, scale=sigma_FAP)
                    # Calculate the sigma value corresponding to this number.
                    sigma[t, i, j] = (up_lim_confiden - mu_FAP) / sigma_FAP
                else:
                    FAP = 1. - FAP
                    x, up_lim_confiden = norm.interval(FAP, loc=mu_FAP, scale=sigma_FAP)
                    sigma[t, i, j] = (up_lim_confiden - mu_FAP) / sigma_FAP

    return sigma, bp, p_cumu, T


def get_obs_data(f1, f2):
    """
    Obtain the observation time information between different fluences.
    :param f1: Lower limit of fluence.
    :param f2: Upper limit of fluence.
    :return: Observation window information, DataFrame data.
    """
    ### Check whether the old burst data of FAST of FRB 20190520B is incorrect.
    # Read Excel file.
    df = pd.read_excel('C:/Users/银河/Desktop/FRB20190520B新数据.xlsx', engine='openpyxl', sheet_name=0)
    # Delete the first four rows (burst rate too high).
    df = df.drop(index=df.index[:4])
    # Select bursts with energy between f1 and f2.
    df.loc[~df['Fluence'].between(f1, f2), 'Burst Count'] = 0
    # Sort.
    df = df.sort_values(by='MJD_topo')
    # Extract the integer part.
    df['integer_part'] = df['MJD_topo'].astype(int)
    # Group by the integer part.
    grouped = df.groupby('integer_part')
    # Extract the first burst time of each group.
    first_burst = grouped['MJD_topo'].first()
    # Calculate how many bursts are observed in each instance.
    burst_number = np.array(grouped['Burst Count'].sum())

    # Define the aggregation operation.
    def range_diff(series):
        return (series.max() - series.min()) * 24 * 60

    # Use the agg method to calculate the range (maximum minus minimum) and the mean
    result_fast_old = grouped.agg({
        'MJD_topo': range_diff,  # Calculate the difference between the maximum and minimum values.
        'Obs_time': 'mean'  # Calculate the average value.
    }).reset_index()
    # Rename the column.
    result_fast_old.columns = ['Start Time(MJD)(LSPx)', 'burst time difference', 'obs time']
    # Generate a new column based on a condition.
    result_fast_old['Err?'] = (result_fast_old['burst time difference'] > result_fast_old['obs time']).astype(int)
    result_fast_old['Telescope'] = 'FAST'
    # Update the observation time.
    result_fast_old.loc[result_fast_old['Err?'] == 1, 'Duration(hr)'] = 10 * (
        np.ceil((result_fast_old.loc[result_fast_old['Err?'] == 1, 'burst time difference']) / 10))
    result_fast_old.loc[result_fast_old['Err?'] == 0, 'Duration(hr)'] = result_fast_old.loc[
        result_fast_old['Err?'] == 0, 'obs time']

    # Calculate the difference between the third column and the second column in df2.
    result_fast_old['difference'] = (result_fast_old['Duration(hr)'] - result_fast_old['burst time difference']) / (
                2 * 60 * 24)
    difference_list = result_fast_old['difference'].tolist()
    # The start time of the observation.
    obs_start_time = np.array(first_burst - difference_list)
    # Assign the value correctly.
    result_fast_old['Start Time(MJD)(LSPx)'] = obs_start_time
    result_fast_old['Burst number'] = burst_number
    result_fast_old['LSP(y)'] = 1
    result_fast_old['Duration(hr)'] = result_fast_old['Duration(hr)'] / 60
    result_fast_old = result_fast_old.drop(columns=['difference'])
    ### Organize.
    # Get the columns that need to be moved.
    columns_to_move = ['burst time difference', 'obs time', 'Err?']
    # Rearrange the columns.
    new_order = [col for col in result_fast_old.columns if col not in columns_to_move] + columns_to_move
    # Rearrange the DataFrame according to the new column order.
    result_fast_old = result_fast_old[new_order]

    ### Check if the burst data for PKS in the FRB 20190520B dataset is incorrect.
    # Read the Excel file.
    df = pd.read_excel('C:/Users/银河/Desktop/FRB20190520B新数据.xlsx', engine='openpyxl', sheet_name=2)
    # Select the bursts with energy between f1 and f2.
    df.loc[~df['Fluence'].between(f1, f2), 'Burst Count'] = 0
    # Sort.
    df = df.sort_values(by='MJD')
    # Extract the integer part.
    df['integer_part'] = df['MJD'].astype(int)
    # Delete 66 bursts (due to excessively high burst rate).
    df = df[~df['integer_part'].isin([59373])]
    # Group by integer part.
    grouped = df.groupby('integer_part')

    # Filter data based on data completeness.
    if f1 < erglimit_PKS and erglimit_PKS < f2 and f2 < 1000:
        # Filter out all groups containing "Flu" values greater than f1 and less than the energy threshold (lacking completeness).
        invalid_groups = grouped.filter(lambda x: ((x['Fluence'] > erglimit_PKS) & (x['Fluence'] < f2)).any())
        # Extract data without completeness.
        mjd_invalues_pks = np.array(invalid_groups['MJD'])
        # Select all rows to be deleted (lacking completeness).
        invalid_indices = invalid_groups.index
        # Delete these rows from the original DataFrame.
        df.drop(invalid_indices, inplace=True)
        # sort.
        df = df.sort_values(by='MJD')
        # Group again by the integer part.
        grouped = df.groupby('integer_part')
    else:
        mjd_invalues_pks = np.array([])
    # Extract the first burst time from each group.
    first_burst = grouped['MJD'].first()
    # Calculate how many bursts are observed each time.
    burst_number = np.array(grouped['Burst Count'].sum())
    # Use the agg method to calculate the difference between the maximum and minimum values, and the average of another column.
    result_pks = grouped.agg({
        'MJD': range_diff,  # Calculate the difference between the maximum and minimum values.
        'Obs_time': 'mean'  # Calculate the average value.
    }).reset_index()
    # Rename the column.
    result_pks.columns = ['Start Time(MJD)(LSPx)', 'burst time difference', 'obs time']
    # Generate a new column based on a condition.
    result_pks['Err?'] = (result_pks['burst time difference'] > result_pks['obs time']).astype(int)
    result_pks['Telescope'] = 'Parkes'
    # Update the observation time.
    result_pks.loc[result_pks['Err?'] == 1, 'Duration(hr)'] = 10 * (
        np.ceil((result_pks.loc[result_pks['Err?'] == 1, 'burst time difference']) / 10))
    result_pks.loc[result_pks['Err?'] == 0, 'Duration(hr)'] = result_pks.loc[result_pks['Err?'] == 0, 'obs time']

    # Calculate the difference between the third and second columns in df2.
    result_pks['difference'] = (result_pks['Duration(hr)'] - result_pks['burst time difference']) / (2 * 60 * 24)
    difference_list = result_pks['difference'].tolist()
    # The start time of the observation.
    obs_start_time = np.array(first_burst - difference_list)
    # Correct assignment.
    result_pks['Start Time(MJD)(LSPx)'] = obs_start_time
    result_pks['Burst number'] = burst_number
    result_pks['LSP(y)'] = 1
    result_pks['Duration(hr)'] = result_pks['Duration(hr)'] / 60
    result_pks = result_pks.drop(columns=['difference'])
    ### Organize.
    # Get the columns that need to be moved.
    columns_to_move = ['burst time difference', 'obs time', 'Err?']
    # Rearrange the columns.
    new_order = [col for col in result_pks.columns if col not in columns_to_move] + columns_to_move
    # Rearrange the DataFrame according to the new column order.
    result_pks = result_pks[new_order]
    # If the energy threshold is greater than the maximum value of the energy bin, it indicates that the telescope cannot observe anything in this energy range, so all observations should be discarded.
    if erglimit_PKS >= f2:
        result_pks = pd.DataFrame(columns=result_pks.columns)

    ### Check if the burst data from GBT in the new data of FRB 20190520B is correct.
    # Read the Excel file.
    df = pd.read_excel('C:/Users/银河/Desktop/FRB20190520B新数据.xlsx', engine='openpyxl', sheet_name=3)
    # Select the bursts with energy between f1 and f2.
    df.loc[~df['Fluence(Jy ms)'].between(f1, f2), 'Burst Count'] = 0
    # sort.
    df = df.sort_values(by='MJD')
    # Group by observation date.
    grouped = df.groupby('Obs_Data')
    # Filter the data based on completeness.
    if f1 < erglimit_GBT and erglimit_GBT < f2 and f2 < 1000:
        # Filter out all groups containing "Flu" values greater than f1 and less than the energy threshold (lacking completeness).
        invalid_groups = grouped.filter(lambda x: ((x['Fluence(Jy ms)'] > erglimit_GBT) & (x['Fluence(Jy ms)'] < f2)).any())
        # Extract data without completeness.
        mjd_invalues_gbt = np.array(invalid_groups['MJD'])
        # Select all rows to be deleted (those lacking completeness).
        invalid_indices = invalid_groups.index
        # Delete these rows from the original DataFrame.
        df.drop(invalid_indices, inplace=True)
        # sort.
        df = df.sort_values(by='MJD')
        # Group again by the integer part.
        grouped = df.groupby('Obs_Data')
    else:
        mjd_invalues_gbt = np.array([])
    # Extract the first burst time for each group.
    first_burst = grouped['MJD'].first()
    # Calculate how many bursts were observed each time.
    burst_number = np.array(grouped['Burst Count'].sum())
    # Use the agg method to calculate the difference between the maximum and minimum values, as well as the average of another column.
    result_GBT = grouped.agg({
        'MJD': range_diff,  # Calculate the difference between the maximum and minimum values.
        'Obs_time': 'mean'  # Calculate the average value.
    }).reset_index()
    # Rename the column.
    result_GBT.columns = ['Start Time(MJD)(LSPx)', 'burst time difference', 'obs time']
    result_GBT['obs time'] = result_GBT['obs time'] * 60
    # Generate a new column based on a condition.
    result_GBT['Err?'] = (result_GBT['burst time difference'] > result_GBT['obs time']).astype(int)
    result_GBT['Telescope'] = 'GBT'
    # Update the observation time.
    result_GBT.loc[result_GBT['Err?'] == 1, 'Duration(hr)'] = 10 * (
        np.ceil((result_GBT.loc[result_GBT['Err?'] == 1, 'burst time difference']) / 10))
    result_GBT.loc[result_GBT['Err?'] == 0, 'Duration(hr)'] = result_GBT.loc[result_GBT['Err?'] == 0, 'obs time']

    # Calculate the difference between the third column and the second column in df2.
    result_GBT['difference'] = (result_GBT['Duration(hr)'] - result_GBT['burst time difference']) / (2 * 60 * 24)
    difference_list = result_GBT['difference'].tolist()
    # The start time of the observation.
    obs_start_time = np.array(first_burst - difference_list)
    # Assign the value correctly.
    result_GBT['Start Time(MJD)(LSPx)'] = obs_start_time
    result_GBT['Burst number'] = burst_number
    result_GBT['LSP(y)'] = 1
    result_GBT['Duration(hr)'] = result_GBT['Duration(hr)'] / 60
    result_GBT = result_GBT.drop(columns=['difference'])
    ### Organize.
    # Get the columns that need to be moved.
    columns_to_move = ['burst time difference', 'obs time', 'Err?']
    # Rearrange the columns.
    new_order = [col for col in result_GBT.columns if col not in columns_to_move] + columns_to_move
    # Rearrange the DataFrame based on the new column order.
    result_GBT = result_GBT[new_order]
    # If the energy threshold is greater than the maximum value of the energy bin, it indicates that the telescope cannot detect anything in this energy range, so all observations in this range should be removed.
    if erglimit_GBT >= f2:
        result_GBT = pd.DataFrame(columns=result_GBT.columns)

    result = pd.concat([result_fast_old, result_pks, result_GBT], ignore_index=True)
    result = result.sort_values(by='Start Time(MJD)(LSPx)')

    result_append = pd.read_excel('C:/Users/银河/Desktop/FRB20190520B整合数据.xlsx', engine='openpyxl', sheet_name=1)
    result_append = result_append[result_append['Burst number'] == 0]
    result = pd.concat([result, result_append], ignore_index=True)
    # Remove the incomplete observations where the value is 0.
    if erglimit_PKS > f1:
        result = result[~((result['Burst number'] == 0) & (result['Telescope'] == 'Parkes'))]
    if erglimit_GBT > f1:
        result = result[~((result['Burst number'] == 0) & (result['Telescope'] == 'GBT'))]
    invalid_mjd = np.concatenate((mjd_invalues_pks, mjd_invalues_gbt))
    result = result.sort_values(by='Start Time(MJD)(LSPx)')
    return result, invalid_mjd


# Set the fluence selection range.
f1 = 0
f2 = 0.5
if f2 == 2:
    f2 = 3000000000

# Read the FAP distribution. The user reads the calculation result file based on the FAP distribution (FRB 20190520B).
p_cumu_dis = read_all_files('D:/PyCharm/Project/Project_FRB/模拟数据/PPBA/高精度周期检测/Pcumu分布/实际flu'+str(f1)+'~'+str(f2)+'观测窗口和暴发率(去除58个后)/最小Pcumu数据')
# The data has a large range, so convert it to log10(P_cumu).
log_data = np.log10(p_cumu_dis)
# Fit a Gaussian distribution.
mu_FAP, sigma_FAP = norm.fit(log_data)

# Telescope energy threshold. Obtain the corresponding observation time information.
erglimit_GBT = 0.1
erglimit_PKS = 0.5
obs_data, invalid_MJD = get_obs_data(f1, f2)
# Read the burst time and count rate data from the dataset.
Set = pd.read_excel(r'C:\Users\银河\Desktop\FRB20190520B整合数据(No FAST New).xlsx', engine='openpyxl', sheet_name=2)   # Data set.
Set = Set[(Set['Fluence(Jy ms)'] >= f1) & (Set['Fluence(Jy ms)'] < f2)]
MJD1 = Set['MJD']
MJD1 = np.array(MJD1)
# Find and delete the values in MJD1 that are present in invalid_MJD.
mask = np.isin(MJD1, invalid_MJD)
MJD1 = MJD1[~mask]
MJD1 = np.sort(MJD1)

waiting_time = np.diff(MJD1) * 24 * 3600 * 1000  # The waiting time between bursts, in milliseconds (ms).
indices = np.where(waiting_time < 100)   # Search for bursts where the inter-burst interval is less than 100 ms.
MJD1 = np.delete(MJD1, indices)   # Merge bursts with an inter-burst interval of less than 100 ms into a single burst.

MJD1 = MJD1 - 56000.  # The corresponding time of bursts in the dataset, simplified for calculation.
MJD1 = np.sort(np.squeeze(MJD1))   # Store burst data.
MJD1 = MJD1.reshape(-1, 1)   # Array shape transformation.
d1 = len(MJD1[:, 0])   # The total number of data points in the dataset.
set1_Rate = np.ones((len(MJD1),))    # Counting rate.
r1 = np.array(set1_Rate)
r1 = r1.reshape(-1, 1)
r1_tot = np.sum(r1)   # Total counting rate.

# The observational data corresponding to the dataset.
set1_win = obs_data['Start Time(MJD)(LSPx)']    # Observation start time.
start1 = np.array(set1_win) - 56000.
start1 = start1.reshape(-1, 1)
d2 = len(start1[:, 0])
set1_DUR = obs_data['Duration(hr)']   # Observation duration (unit: h).
dur1 = np.array(set1_DUR)
dur1 = dur1.reshape(-1, 1)
obs_tot1 = np.sum(dur1)   # Total observation time.

# Array shape transformation.
obs_day_time_r = start1  # The start time of the observation.
obs_dur_time_r = dur1  # The duration of the observation.
obs_dur_time_r = obs_dur_time_r / 24  # Unit conversion to days.
# Calculate the observation distribution.
nT = 1000   # The total number of cycles for frequency search is 1000.
nop = 40
ncp = 20
n_o = len(obs_day_time_r)
obs_dis_1 = np.zeros((nT, ncp, nop))  # Used to store the distribution of observation time with respect to phase for different test periods/frequencies and different central active phases.  # 1 to 2-day test period.
obs_dis_2 = np.zeros((nT, ncp, nop))  # 2 to 4-day test period.
obs_dis_3 = np.zeros((nT, ncp, nop))  # 4 to 8-day test period.
obs_dis_4 = np.zeros((nT, ncp, nop))  # 8 to 16-day test period.
obs_dis_5 = np.zeros((nT, ncp, nop))  # 16 to 32-day test period.
obs_dis_6 = np.zeros((nT, ncp, nop))  # 32 to 64-day test period.
obs_dis_7 = np.zeros((nT, ncp, nop))  # 64 to 128-day test period.
obs_dis_8 = np.zeros((nT, ncp, nop))  # 128 to 256-day test period.
obs_dis_9 = np.zeros((nT, ncp, nop))  # 256 to 512-day test period.
obs_dis_10 = np.zeros((nT, ncp, nop))  # 512 to 1024-day test period.
# # 1 to 2-day test period.
fre = np.linspace(1 / 1, 1 / 2, nT)  # Set test frequency. nT=1000
T_1 = 1. / fre
for t in range(nT):  # Different test periods/frequencies.
    obs_dis_1[t, :, :] = obs_distribution(T_1[t], obs_day_time_r, obs_dur_time_r)  # Calculate the test period and the phase distribution of the observation time under different center phases.
# # 2 to 4-day test period.
fre = np.linspace(1 / 2, 1 / 4, nT)
T_1 = 1. / fre
for t in range(nT):
    obs_dis_2[t, :, :] = obs_distribution(T_1[t], obs_day_time_r, obs_dur_time_r)
# # 4 to 8-day test period.
fre = np.linspace(1 / 4, 1 / 8, nT)
T_1 = 1. / fre
for t in range(nT):
    obs_dis_3[t, :, :] = obs_distribution(T_1[t], obs_day_time_r, obs_dur_time_r)
# # 8 to 16-day test period.
fre = np.linspace(1 / 8, 1 / 16, nT)
T_1 = 1. / fre
for t in range(nT):
    obs_dis_4[t, :, :] = obs_distribution(T_1[t], obs_day_time_r, obs_dur_time_r)
# # 16 to 32-day test period.
fre = np.linspace(1 / 16, 1 / 32, nT)
T_1 = 1. / fre
for t in range(nT):
    obs_dis_5[t, :, :] = obs_distribution(T_1[t], obs_day_time_r, obs_dur_time_r)
# # 32 to 64-day test period.
fre = np.linspace(1 / 32, 1 / 64, nT)
T_1 = 1. / fre
for t in range(nT):
    obs_dis_6[t, :, :] = obs_distribution(T_1[t], obs_day_time_r, obs_dur_time_r)
# 64 to 128-day test period.
fre = np.linspace(1 / 64, 1 / 128, nT)
T = 1. / fre
for t in range(nT):
    obs_dis_7[t, :, :] = obs_distribution(T[t], obs_day_time_r, obs_dur_time_r)
# 128 to 256-day test period.
fre = np.linspace(1 / 128, 1 / 256, nT)
T = 1. / fre
for t in range(nT):
    obs_dis_8[t, :, :] = obs_distribution(T[t], obs_day_time_r, obs_dur_time_r)
# 256 to 512-day test period.
fre = np.linspace(1 / 256, 1 / 512, nT)
T = 1. / fre
for t in range(nT):
    obs_dis_9[t, :, :] = obs_distribution(T[t], obs_day_time_r, obs_dur_time_r)
# 512 to 1024-day test period.
fre = np.linspace(1 / 512, 1 / 1024, nT)
T = 1. / fre
for t in range(nT):
    obs_dis_10[t, :, :] = obs_distribution(T[t], obs_day_time_r, obs_dur_time_r)

# Set other variables during period analysis.
aphase = np.linspace(0, 0.95, 20)  # Set the assumed strong magnetic field center phase from 0 to 0.95, with a phase interval of 0.05.
ncp = 20  # number of center phase, The total number of cycles for center phase search is 20.

aphw = np.linspace(0.025, 0.2, 8)  # Set the half-width of the active area to be searched.
npw = 8  # The number of cycles for active area width search is 8.

obs_phase = np.linspace(0, 0.95, 40)
nop = 40  # The observation time distribution is divided into 40 equal parts from 0 to 0.95 phase.

MJD = MJD1   # All bursts.
n_d = d1   # Number of bursts.
Rate_tot = r1_tot   # Total count rate.
Rate = r1   # Single burst count rate.
obs_tot = obs_tot1   # Total observation time (hours).
obs_tot = obs_tot / 24  # Total observation time (days).
start = start1   # Start time of each observation.
dur = dur1   # Duration of each observation (hours).
dur = dur / 24  # Duration of each observation (days).
n_o = d2   # Number of observations.


# Conduct period searches.
Ni = np.round((np.max(MJD) - np.min(MJD))) * 0.5 / 0.0000001  # fNy = 1/2 * 1/FRB accuracy. (10 ms converted to days).
Ni = Ni * 20 * 8  # Ni = fNy * T_1 * Search center frequency count * Number of search phase widths
sigma_1_initial, bp_1, p_cumu_1, T_1 = period_search(1, 2, obs_dis_1)
sigma_2_initial, bp_2, p_cumu_2, T_2 = period_search(2, 4, obs_dis_2)
sigma_3_initial, bp_3, p_cumu_3, T_3 = period_search(4, 8, obs_dis_3)
sigma_4_initial, bp_4, p_cumu_4, T_4 = period_search(8, 16, obs_dis_4)
sigma_5_initial, bp_5, p_cumu_5, T_5 = period_search(16, 32, obs_dis_5)
sigma_6_initial, bp_6, p_cumu_6, T_6 = period_search(32, 64, obs_dis_6)
sigma_7_initial, bp_7, p_cumu_7, T_7 = period_search(64, 128, obs_dis_7)
sigma_8_initial, bp_8, p_cumu_8, T_8 = period_search(128, 256, obs_dis_8)
sigma_9_initial, bp_9, p_cumu_9, T_9 = period_search(256, 512, obs_dis_9)
sigma_10_initial, bp_10, p_cumu_10, T_10 = period_search(512, 1024, obs_dis_10)


if f2 > 2:
    f2 = 2


np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻sigma结果(代入拟合高斯曲线算sigma)/"+str(f1)+"~"+str(f2)+"搜寻周期1~2天", sigma_1_initial)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻P_cumu结果/"+str(f1)+"~"+str(f2)+"搜寻周期1~2天", p_cumu_1)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻暴发相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期1~2天", bp_1)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻观测相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期1~2天", obs_dis_1)
#
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻sigma结果(代入拟合高斯曲线算sigma)/"+str(f1)+"~"+str(f2)+"搜寻周期2~4天", sigma_2_initial)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻P_cumu结果/"+str(f1)+"~"+str(f2)+"搜寻周期2~4天", p_cumu_2)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻暴发相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期2~4天", bp_2)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻观测相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期2~4天", obs_dis_2)
#
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻sigma结果(代入拟合高斯曲线算sigma)/"+str(f1)+"~"+str(f2)+"搜寻周期4~8天", sigma_3_initial)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻P_cumu结果/"+str(f1)+"~"+str(f2)+"搜寻周期4~8天", p_cumu_3)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻暴发相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期4~8天", bp_3)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻观测相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期4~8天", obs_dis_3)
#
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻sigma结果(代入拟合高斯曲线算sigma)/"+str(f1)+"~"+str(f2)+"搜寻周期8~16天", sigma_4_initial)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻P_cumu结果/"+str(f1)+"~"+str(f2)+"搜寻周期8~16天", p_cumu_4)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻暴发相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期8~16天", bp_4)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻观测相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期8~16天", obs_dis_4)
#
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻sigma结果(代入拟合高斯曲线算sigma)/"+str(f1)+"~"+str(f2)+"搜寻周期16~32天", sigma_5_initial)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻P_cumu结果/"+str(f1)+"~"+str(f2)+"搜寻周期16~32天", p_cumu_5)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻暴发相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期16~32天", bp_5)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻观测相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期16~32天", obs_dis_5)
#
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻sigma结果(代入拟合高斯曲线算sigma)/"+str(f1)+"~"+str(f2)+"搜寻周期32~64天", sigma_6_initial)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻P_cumu结果/"+str(f1)+"~"+str(f2)+"搜寻周期32~64天", p_cumu_6)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻暴发相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期32~64天", bp_6)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻观测相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期32~64天", obs_dis_6)

np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻sigma结果(代入拟合高斯曲线算sigma)/"+str(f1)+"~"+str(f2)+"搜寻周期64~128天", sigma_7_initial)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻P_cumu结果/"+str(f1)+"~"+str(f2)+"搜寻周期64~128天", p_cumu_7)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻暴发相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期64~128天", bp_7)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻观测相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期64~128天", obs_dis_7)
#
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻sigma结果(代入拟合高斯曲线算sigma)/"+str(f1)+"~"+str(f2)+"搜寻周期128~256天", sigma_8_initial)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻P_cumu结果/"+str(f1)+"~"+str(f2)+"搜寻周期128~256天", p_cumu_8)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻暴发相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期128~256天", bp_8)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻观测相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期128~256天", obs_dis_8)
#
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻sigma结果(代入拟合高斯曲线算sigma)/"+str(f1)+"~"+str(f2)+"搜寻周期256~512天", sigma_9_initial)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻P_cumu结果/"+str(f1)+"~"+str(f2)+"搜寻周期256~512天", p_cumu_9)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻暴发相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期256~512天", bp_9)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻观测相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期256~512天", obs_dis_9)
#
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻sigma结果(代入拟合高斯曲线算sigma)/"+str(f1)+"~"+str(f2)+"搜寻周期512~1024天", sigma_10_initial)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻P_cumu结果/"+str(f1)+"~"+str(f2)+"搜寻周期512~1024天", p_cumu_10)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻暴发相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期512~1024天", bp_10)
np.save("D:/PyCharm/Project/Project_FRB/190520B周期搜寻_新数据/190520B搜寻结果(No FAST New)/190520B搜寻观测相位结果/"+str(f1)+"~"+str(f2)+"搜寻周期512~1024天", obs_dis_10)