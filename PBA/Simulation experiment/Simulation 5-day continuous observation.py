"""
The script is designed for a simulation experiment: continuous observation for 5 days, totaling 20 hours.
It generates bursts using random numbers and performs periodic search using PBA.
The file reading and storage paths can be adjusted by the user.
"""

import numpy as np
import math
from scipy.stats import norm
from scipy.stats import binom
import random
import multiprocessing
import os
import re


def task(actual_period, actual_p, actual_pbp, actual_burst_rate):

    def simulated_data_arrays(t=20, p=0.2, pbp=0.67, each=1, x=1, obs_day_num=10, obs_hours_all=20, burst_rate=50):
        """

        This function is used to generate simulated signals. More specific simulation settings for FRB bursts and observational settings can be adjusted within the function.
        :param t: The input value is a number, representing the period of the simulated FRB's actual existence.
        :param p: The input value is a decimal between 0 and 1, representing the width of the active region of the simulated FRB.
        :param pbp: The input value is a number, representing the proportion of periodic signals in the simulated FRB's actual existence. The proportion of periodic signals is the ratio of all periodic signals within one period to the total number of signals.
        :param each: The input value is a number, representing the basis for the simulated observation. "Each" indicates that the observation is conducted once per day.
        :param x: The input value is a number, representing the basis for the simulated observation. "x" indicates that each observation lasts for x days.
        :param obs_day_num: The input value is a number, representing the total number of days for the simulated observation.
        :param obs_hours_all: The input value is a number, representing the total number of hours for the simulated observation.
        :param burst_rate: The input value is a number, serving as the basis for the burst generation. A burst_rate of 50 means that, on average, 50 bursts are generated per day.
        :return: Return the simulated observation data.
        """

        arrays_list = []  # Used for return value.
        ### Simulate burst.
        # Simulated data with adjustable initial settings.
        # Preset period.
        T = t
        # Preset active region width.
        P = p
        # Set the start and end MJD of the simulated data.
        start = 56000 + random.uniform(0, T)
        end = 62000
        # Preset daily burst numbers for the inactive and active regions.
        PBP = pbp  # Periodic bursts percentage
        # Daily burst count generated in the inactive region (noise signal).
        num_non = (burst_rate - burst_rate * PBP) * (1 - P)
        num_act = burst_rate * PBP + (
                burst_rate - burst_rate * PBP) * P  # Daily burst count generated in the active region (noise signal + periodic signal).

        # Set the observation time.
        obs_hours = obs_hours_all / obs_day_num  # Set the simulation observation duration for each observation.
        obs_day = np.arange(56100, 56200 + each * obs_day_num / x)
        obs_day = obs_day[(obs_day) % each < x]
        obs_day = obs_day[:math.ceil(obs_day_num)]
        # Set the observation time for each observation.
        obs_time_start = 14 / 24  # The observation starts at 2:00 PM every day, converted to MJD.
        obs_time_end = obs_time_start + obs_hours / 24  # converted to MJD

        # Store the entire dataset.
        data_set = set()

        # Simulate burst data generation.
        for t in np.arange(start, end, T):
            start_t = t
            if start_t + T <= end:
                non_act_end = start_t + T * (1 - P)  # Inactive zone time endpoint.
                act_end = start_t + T  # Active zone time endpoint.
                num_generate_non = round(
                    T * num_non)  # Total number of bursts to be generated for one cycle in the inactive zone.
                num_generate = round(
                    T * num_act)  # Total number of bursts to be generated for one cycle in the active zone.
                # Inactive zone data generation.
                for m in range(num_generate_non):
                    data_set.add(random.uniform(start_t, non_act_end))

                # Active zone data generation.
                for m in range(num_generate):
                    data_set.add(random.uniform(non_act_end, act_end))

            else:  # Generation of simulation data for the last cycle.

                if start_t + T * (1 - P) < end:  # The last cycle can include the entire inactive zone.

                    non_act_end = start_t + T * (1 - P)  # Inactive zone time endpoint.
                    act_end = end  # Active zone time endpoint.
                    T_rest = end - start_t  # Remaining burst generation cycle length.
                    num_generate_non = round(
                        T_rest * num_non)  # Total number of bursts to be generated in the inactive zone.
                    num_generate = round(
                        T_rest * num_act)  # Total number of bursts to be generated in the active zone.
                    # Inactive zone data generation.
                    for m in range(num_generate_non):
                        data_set.add(random.uniform(start_t, non_act_end))

                    # Active zone data generation.
                    for m in range(num_generate):
                        data_set.add(random.uniform(non_act_end, act_end))

                else:  # The last cycle cannot include the inactive zone.

                    non_act_end = end  # Inactive zone time endpoint.
                    T_rest = non_act_end - start_t  # The duration of the inactive zone.
                    num_generate_non = round(
                        T_rest * num_non)  # The total number of bursts to be generated in the inactive zone.
                    # Inactive zone data generation.
                    for m in range(num_generate_non):
                        data_set.add(random.uniform(start_t, non_act_end))
        data_set = list(data_set)
        data_set = np.array(data_set)
        data_set = np.sort(data_set)

        ### Simulated observation.
        # Storage of observation dataset.
        data_obs_burst = np.array([])  # Storage of burst data.
        data_obs_time = np.array([])  # Storage of observation start time.
        data_obs_dur = np.array([])  # Storage of observation duration for each instance (in hours).
        data_obs_ifb = np.array(
            [])  # Storage of whether an burst occurred during this observation: 1 for burst, 0 for no burst.

        # Obtain the observation start time dataset and store it in the data_obs_time array.
        # Obtain the simulated observation burst dataset and store it in the data_obs_burst array.
        for day in obs_day:
            data_obs_time = np.append(data_obs_time, day + obs_time_start)
            data_obs_dur = np.append(data_obs_dur, obs_time_end * 24 - obs_time_start * 24)
            data_obs_burst = np.append(data_obs_burst,
                                       data_set[(data_set >= day + obs_time_start) & (data_set <= day + obs_time_end)])
            if len(data_set[(data_set >= day + obs_time_start) & (data_set <= day + obs_time_end)]) == 0:
                data_obs_ifb = np.append(data_obs_ifb, 0)
            else:
                data_obs_ifb = np.append(data_obs_ifb, 1)

        data_obs_rate = np.ones(len(data_obs_burst))  # Store the observed burst count rate.
        # Array reshaping.
        data_obs_burst = data_obs_burst.reshape(len(data_obs_burst), 1)  # Store the observed burst data.
        data_obs_rate = data_obs_rate.reshape(len(data_obs_rate), 1)  # Store the observed burst count rates.
        data_obs_time = data_obs_time.reshape(len(data_obs_time), 1)  # Store the observation start time.
        data_obs_dur = data_obs_dur.reshape(len(data_obs_dur), 1)  # Store the duration of each observation (in hours).
        data_obs_ifb = data_obs_ifb.reshape(len(data_obs_ifb),
                                            1)  # Store whether an burst occurred during this observation: 1 for burst, 0 for no burst.
        # Return data.
        arrays_list.append(data_obs_burst)
        arrays_list.append(data_obs_rate)
        arrays_list.append(data_obs_time)
        arrays_list.append(data_obs_dur)
        arrays_list.append(data_obs_ifb)
        return arrays_list

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
        bp_obs = phase_folding(T, Start,
                               n_o)  # Obtain the phase for different centers within this period, with each observation start time (start) corresponding to the phase difference.
        obs_p = Dur / T  # Calculate the phase occupied by the observation duration (dur) within this period.
        for i in range(ncp):  # The observation start phase differs for each phase center calculated.
            for j in range(n_o):  # Perform calculations for each observation activity.
                while obs_p[j] >= 1:
                    obs_dis[i, :] = obs_dis[i, :] + int(obs_p[j]) * T / nop
                    obs_p[j] = obs_p[j] - int(obs_p[j])
                for k in range(nop):  # For each phase bin that stores the observation time.
                    crit_p = (k + 1) / nop  # Define the critical phase as 0.05 * k.
                    if bp_obs[i, j] < crit_p:  # If the observation start time is within the range of this bin.
                        delta = crit_p - bp_obs[
                            i, j]  # Calculate the difference between the observation start phase and the critical phase.

                        if obs_p[
                            j] <= delta:  # If the phase occupied by the observation time is smaller than the calculated difference, the entire observation time of the activity is accumulated into the (k+1)-th bin (where k=0 for the first bin).
                            obs_dis[i, k] += Dur[
                                j]  # Note that when judging the condition, phase is used, while time is added during the accumulation.

                        elif obs_p[
                            j] > delta:  # If the phase occupied by the observation time is greater than the phase difference, the time corresponding to the phase difference is accumulated into the current bin, and the remaining time needs to be carried over to subsequent bins.
                            obs_dis[i, k] += delta * T
                            obph = obs_p[
                                       j] - delta  # "obph" refers to the phase occupied by the remaining observation time after subtracting the time already added to the bins.

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
        bp = np.zeros((nT, ncp,
                       n_d))  # Used to store the phases corresponding to burst events under different test periods/frequencies and different central active phases.
        count = np.zeros((nT, ncp,
                          npw))  # Used to store the number of bursts occurring within the active region under different test periods/frequencies, central active phases, and active region widths.
        r = np.zeros((nT, ncp,
                      npw))  # Used to store the sum of the count rates corresponding to bursts occurring within the active region under different test periods/frequencies, central active phases, and active region widths.
        F = np.zeros((nT, ncp,
                      npw))  # Used to store the calculated binomial distribution probabilities under different test periods/frequencies, central active phases, and active region widths.
        sigma = np.zeros((nT, ncp, npw))  # Used to store sigma.
        p_cumu = np.zeros((nT, ncp, npw))  # Used to store the cumulative probability of the binomial distribution.

        fre = np.linspace(1 / test_start_d, 1 / test_end_d, nT)  # Set the test frequency. nT=1000
        T = 1. / fre  # Test period = 1/Test frequency
        total_rate = int(Rate_tot + 0.5)
        for t in range(nT):  # Different test periods/frequencies
            bp[t, :, :] = phase_folding(T[t], MJD,
                                        n_d)  # Calculate the test period and the phase corresponding to the occurrence time of each burst at different center phases (with the active center phase as 0 phase).
            for i in range(ncp):  # Different center phases.
                for j in range(npw):  # Different active region widths.
                    hw = aphw[j]  # Half width.
                    filter1 = (bp[t, i, :] < hw) | (
                                bp[t, i, :] > 1 - hw)  # Filter out the bursts located within the active region.
                    count[t, i, j] = len(bp[t, i, filter1])  # The number of bursts within the active region.
                    r[t, i, j] = np.sum(Rate[filter1])  # The sum of the burst counting rates within the active region.
                    obs_hw = np.sum(obs_dis[t, i, 0:j + 1]) + np.sum(
                        obs_dis[t, i, -(j + 1):])  # Total observation duration of the active region.
                    p = obs_hw / obs_tot  # Under the random hypothesis, the probability that a single observed burst occurs within the active region is the percentage of the observation duration of the active region relative to the total observation duration.
                    non_act_burst_num = total_rate - r[t, i, j]
                    if p == 0:
                        non_act_burst_num = total_rate

                    P_binom = binom.cdf(non_act_burst_num, total_rate, 1 - p)
                    p_cumu[t, i, j] = P_binom
                    FAP = False_alarm_Probability(P_binom, Ni)

                    # Calculate sigma using the standard Gaussian curve.
                    if FAP <= 0:
                        FAP = 1e-322

                    F[t, i, j] = FAP
                    if FAP < 1e-16:
                        sigma[t, i, j] = norm.isf(FAP)
                    else:
                        FAP = 1. - FAP
                        x, sigma[t, i, j] = norm.interval(FAP)
                        if sigma[t, i, j] == float('inf'):
                            sigma[t, i, j] = 8.21

        return sigma, bp, p_cumu, T

    def choose_false_period(sigma_max, sigma_n_max, T_n):
        """
        Select the false period where sigma is greater than the actual period.
        :param sigma_max: Sigma of the actual period.
        :param sigma_n_max: The maximum sigma value for each searched period.
        :param T_n: The searched time series.
        :return: The false periods where sigma is greater than the actual period.
        """

        indices = np.array(np.where(sigma_n_max > sigma_max))
        indices = indices.reshape(-1)
        False_period = T_n[indices]

        return False_period

    def calculate_max_sigma_num_false(T, sigma):
        """
        Calculate the maximum sigma value of the actual period found, as well as the number of false detections.
        :param T: The searched time series closest to the actual period.
        :param sigma: Sigma matrix with a specific phase half-width.
        :return:Return the maximum sigma value of the actual period found, as well as the number of false detections.
        """
        xushu = int(np.argmin(np.abs(T - simulated_t)))  # Output the index value of the day that is closest to the simulated FRB actual period in the period search.
        if xushu >= 15 and xushu <= len(T) - 16:
            sigma_max_range = sigma[xushu - 15:xushu + 16, :]
        elif xushu < 15:
            sigma_max_range = sigma[0:xushu + 16, :]
        else:
            sigma_max_range = sigma[xushu - 15:, :]
        sigma_max = np.max(sigma_max_range)  # Output the highest sigma value among the days near the simulated FRB actual period.

        sigma_1_max = np.max(sigma_1, axis=1)
        sigma_2_max = np.max(sigma_2, axis=1)
        sigma_3_max = np.max(sigma_3, axis=1)

        sigma_max_combin = np.concatenate([sigma_1_max, sigma_2_max, sigma_3_max])
        count_mis = np.sum(sigma_max_combin > sigma_max)
        false_period_1 = choose_false_period(sigma_max, sigma_1_max, T_1)
        false_period_2 = choose_false_period(sigma_max, sigma_2_max, T_2)
        false_period_3 = choose_false_period(sigma_max, sigma_3_max, T_3)

        false_period = np.concatenate([false_period_1, false_period_2, false_period_3])

        return sigma_max, count_mis, false_period

    ######################################## Generate simulated data.
    simulated_p = actual_p[0]  # Simulate the phase width of the active region where the actual FRB exists.
    simulated_pbp = actual_pbp[0]  # Simulate the proportion of periodic signals in the actual FRB active region.
    simulated_each = 1  # Simulate the observational setup of the observed data.
    simulated_x = 1  # Simulate the observational setup of the observed data.
    simulated_obs_day_num = 5  # Simulate the observational setup of the observed data.
    simulated_obs_hours_all = 20   # Simulate the observational setup of the observed data.
    simulated_burst_rate = actual_burst_rate[0]   # Simulate the burst rate of the bursts.
    ############################################################################################################
    ############################################################################################################

    sigma_each = np.zeros((8, 100))  # Store the sigma results for each cycle of every period.
    miscal_each = np.zeros((8, 100))  # Store the number of false detections for each cycle of every period.
    burst_num_each = np.zeros((100,))  # Store the number of bursts for each cycle of every period.

    # Observation time (used to calculate obs_dis and reduce computation time).
    each_r = simulated_each
    x_r = simulated_x
    obs_hours_all_r = simulated_obs_hours_all
    obs_hours_r = obs_hours_all_r / simulated_obs_day_num
    obs_day_num_r = obs_hours_all_r / obs_hours_r
    obs_day_r = np.arange(56100, 56200 + each_r * obs_day_num_r / x_r)
    obs_day_r = obs_day_r[obs_day_r % each_r < x_r]
    obs_day_r = obs_day_r[:math.ceil(obs_day_num_r)]

    # Set the observation time for each observation.
    obs_time_start_r = 14 / 24  # Start observing from 2:00 PM every day, convert to MJD.
    obs_time_end_r = obs_time_start_r + obs_hours_r / 24  # convert to MJD
    # The start time of each observation.
    obs_day_time_r = obs_day_r + obs_time_start_r - 55000
    # The duration of each observation.
    obs_dur_time_r = np.zeros((len(obs_day_time_r),)) + obs_hours_r
    # Array shape conversion.
    obs_day_time_r = obs_day_time_r.reshape(len(obs_day_time_r), 1)  # The observation start time.
    obs_dur_time_r = obs_dur_time_r.reshape(len(obs_dur_time_r), 1)  # The observation duration.
    obs_dur_time_r = obs_dur_time_r / 24  # Convert to days.
    # Calculate obs_dis.
    nT = 1000
    nop = 40
    ncp = 20
    n_o = len(obs_day_time_r)
    obs_dis_1 = np.zeros((nT, ncp, nop))  # Used to store the distribution of observation time with respect to phase for different test periods/frequencies and different central active phases.  # 1 to 2-day test period.
    obs_dis_2 = np.zeros((nT, ncp, nop))  # 2 to 4-day test period.
    obs_dis_3 = np.zeros((nT, ncp, nop))  # 4 to 8-day test period.
    # # 1 to 2-day test period.
    fre = np.linspace(1 / 1, 1 / 2, nT)  # Set test frequency. nT=1000
    T_1 = 1. / fre
    for t in range(nT):  # Different test periods/frequencies.
        obs_dis_1[t, :, :] = obs_distribution(T_1[t], obs_day_time_r,
                                              obs_dur_time_r)  # Calculate the test period and the phase distribution of the observation time under different center phases.
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

    false_period = [np.empty((0,)) for _ in range(8)]
    for number in range(100):  # Run 100 times for each actual cycle.
        simulated_t = actual_period[0]  # Simulate the actual period of FRB existence.
        # Generate simulated data.
        simulated_data = simulated_data_arrays(t=simulated_t,
                                               p=simulated_p,
                                               pbp=simulated_pbp,
                                               each=simulated_each,
                                               x=simulated_x,
                                               obs_day_num=simulated_obs_day_num,
                                               obs_hours_all=simulated_obs_hours_all,
                                               burst_rate=simulated_burst_rate)
        data_obs_burst_s = simulated_data[0]  # Store the observed burst time data.
        while len(data_obs_burst_s) == 0:   # If no burst is observed in the observation, regenerate the burst data (in reality, if nothing is observed, it holds no significance).
            simulated_data = simulated_data_arrays(t=simulated_t,
                                                   p=simulated_p,
                                                   pbp=simulated_pbp,
                                                   each=simulated_each,
                                                   x=simulated_x,
                                                   obs_day_num=simulated_obs_day_num,
                                                   obs_hours_all=simulated_obs_hours_all,
                                                   burst_rate=simulated_burst_rate)
            data_obs_burst_s = simulated_data[0]   # Store the observed burst time data.
        data_obs_rate_s = simulated_data[1]  # Store the observed burst count rate.
        data_obs_time_s = simulated_data[2]  # Store the observation start time.
        data_obs_dur_s = simulated_data[3]  # Store the duration of each observation (in hours).
        data_obs_ifb_s = simulated_data[4]  # Store whether there was a burst in this observation (1 for burst, 0 for no burst).
        # Read the burst times and count rate data from the dataset.
        set1_MJD = data_obs_burst_s  # Burst time data.
        MJD1 = np.array(set1_MJD) - 55000.  # The time corresponding to the dataset outbreak, simplifying the calculation.
        d1 = len(MJD1[:, 0])  # urst corresponding times in the dataset, simplified for calculation.
        burst_num_each[number] = d1  # Extract the number of bursts for a single cycle.
        set1_Rate = data_obs_rate_s
        r1 = np.array(set1_Rate)
        r1_tot = np.sum(r1)

        # The observational data corresponding to the dataset.
        set1_win = data_obs_time_s  # Observation start time.
        start1 = np.array(set1_win) - 55000.
        d2 = len(start1[:, 0])
        set1_DUR = data_obs_dur_s  # Observation duration.
        dur1 = np.array(set1_DUR)
        obs_tot1 = np.sum(dur1)  # Total observation time.

        # Other variables when setting up the period analysis.
        aphase = np.linspace(0, 0.95, 20)  # Set the assumed strong magnetic field center phase, from 0 to 0.95, with an interval of 0.05 phases.
        ncp = 20  # number of center phase

        aphw = np.linspace(0.025, 0.2, 8)  # Set the semi-width of the active region to search, assuming the active region width range is from 0.05 to 0.4 phases.
        npw = 8

        obs_phase = np.linspace(0, 0.95, 40)
        nop = 40  # Distribute the observation time into 40 equal intervals from 0 to 0.95 phases.

        MJD = MJD1
        n_d = d1
        Rate_tot = r1_tot
        Rate = r1
        obs_tot = obs_tot1
        obs_tot = obs_tot / 24
        start = start1
        dur = dur1
        dur = dur / 24
        n_o = d2

        # Perform period search.
        qk = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        search_index = np.argmin(abs(qk - simulated_p))
        Ni = np.round((np.max(MJD) - np.min(MJD))) * 0.5 / 0.0000001
        Ni = Ni * 20 * 8
        sigma_1_initial, T_1 = period_search(1, 2, obs_dis_1)
        sigma_2_initial, T_2 = period_search(2, 4, obs_dis_2)
        sigma_3_initial, T_3 = period_search(4, 8, obs_dis_3)

        #####################################################################################################
        #####################################################################################################
        for bk in range(search_index, search_index+1):
            sigma_1 = sigma_1_initial[:, :, bk]
            sigma_2 = sigma_2_initial[:, :, bk]
            sigma_3 = sigma_3_initial[:, :, bk]
            if simulated_t > 1 and simulated_t <= 2:
                sigma_max, count_mis, false_period_x = calculate_max_sigma_num_false(T_1, sigma_1)

            elif simulated_t > 2 and simulated_t <= 4:
                sigma_max, count_mis, false_period_x = calculate_max_sigma_num_false(T_2, sigma_2)

            else:
                sigma_max, count_mis, false_period_x = calculate_max_sigma_num_false(T_3, sigma_3)

            false_period[bk] = np.concatenate((false_period[bk], false_period_x))
            sigma_each[bk, number] = sigma_max  ### Extract the maximum sigma value from each cycle.
            miscal_each[bk, number] = count_mis  ### Extract the number of misclassifications in each cycle.

    # Find the length of the longest array.
    max_length = max(len(arr) for arr in false_period)
    # Append zeros to the end of the shorter arrays to make their lengths equal to the longest array.
    padded_list = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in false_period]
    false_period_array = np.array(padded_list)
    # Define the file path.
    path1 = "D:/PyCharm/Project/Project_FRB/模拟数据/PPBA/高精度周期检测/观测" + str(simulated_obs_hours_all) +"小时/连续观测" + str(simulated_obs_day_num) +"天/暴发率" + str(simulated_burst_rate) + "/实际相位宽度" + str(simulated_p) + "/周期信号占比" + str(simulated_pbp) + "/每个周期的误判周期"
    path2 = "D:/PyCharm/Project/Project_FRB/模拟数据/PPBA/高精度周期检测/观测" + str(
        simulated_obs_hours_all) + "小时/连续观测" + str(simulated_obs_day_num) + "天/暴发率" + str(
        simulated_burst_rate) + "/实际相位宽度" + str(simulated_p) + "/周期信号占比" + str(simulated_pbp) + "/sigma"
    path3 = "D:/PyCharm/Project/Project_FRB/模拟数据/PPBA/高精度周期检测/观测" + str(
        simulated_obs_hours_all) + "小时/连续观测" + str(simulated_obs_day_num) + "天/暴发率" + str(
        simulated_burst_rate) + "/实际相位宽度" + str(simulated_p) + "/周期信号占比" + str(simulated_pbp) + "/暴发数量"

    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    if not os.path.exists(path3):
        os.makedirs(path3)
    np.save("D:/PyCharm/Project/Project_FRB/模拟数据/PPBA/高精度周期检测/观测" + str(simulated_obs_hours_all) +"小时/连续观测" + str(simulated_obs_day_num) +"天/暴发率" + str(simulated_burst_rate) + "/实际相位宽度" + str(simulated_p) + "/周期信号占比" + str(simulated_pbp) + "/每个周期的误判周期/" + str(actual_period) + "天的误判周期.npy", false_period_array)
    np.save("D:/PyCharm/Project/Project_FRB/模拟数据/PPBA/高精度周期检测/观测" + str(simulated_obs_hours_all) +"小时/连续观测" + str(simulated_obs_day_num) +"天/暴发率" + str(simulated_burst_rate) + "/实际相位宽度" + str(simulated_p) + "/周期信号占比" + str(simulated_pbp) + "/sigma/" + str(actual_period) + "天的sigma.npy", sigma_each)
    np.save("D:/PyCharm/Project/Project_FRB/模拟数据/PPBA/高精度周期检测/观测" + str(simulated_obs_hours_all) +"小时/连续观测" + str(simulated_obs_day_num) +"天/暴发率" + str(simulated_burst_rate) + "/实际相位宽度" + str(simulated_p) + "/周期信号占比" + str(simulated_pbp) + "/暴发数量/" + str(actual_period) + "天的暴发数量.npy", burst_num_each)
    print(f"已完成{actual_period}天的周期检测循环")


if __name__ == "__main__":

    Actual_period = np.array([1.17, 1.24, 1.3,  1.36, 1.43, 1.59, 1.67, 1.75,
                              2.33, 2.41, 2.51, 2.62, 3.41, 4.3, 4.71, 4.85, 4.96])
    Actual_p = np.array([0.1, 0.2, 0.4])
    Actual_pbp = np.array([0.5, 0.67, 0.8])
    Actual_burst_rate = np.array([25, 50, 100])

    # Create multiple processes.
    processes = []
    for i in range(len(Actual_period)):   # Create processes for different valid observation periods.
        for j in range(len(Actual_p)):   # Create processes for different active phase widths.
            for k in range(len(Actual_pbp)):   # Create processes for different PBP.
                for z in range(len(Actual_burst_rate)):   # Create processes for different burst rates.
                    p = multiprocessing.Process(target=task, args=(Actual_period[i:i+1], Actual_p[j:j+1],
                                                                   Actual_pbp[k:k+1], Actual_burst_rate[z:z+1]))
                    processes.append(p)

    for n in range(math.ceil(len(processes)/20)):
        start_index = n * 20
        end_index = min((n + 1) * 20, len(processes))
        for process in processes[start_index:end_index]:
            process.start()
        for process in processes[start_index:end_index]:
            process.join()

    print("观测策略.5天, 完成")


