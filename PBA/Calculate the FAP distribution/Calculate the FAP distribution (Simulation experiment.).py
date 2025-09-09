"""
   该脚本的观测为实际观测
   可靠周期:Actual_period = np.array([360.09083306 362.1530895  363.49092657 366.32051579 367.70607099
                                    369.00295559 370.06799059 371.81365743 373.06792343 374.31224096
                                    375.38106866 376.66664946 377.7697653  379.51357019 380.57896503
                                    382.00872974 383.03266063 384.36576888 387.414032   388.9552563
                                    390.01230558 391.05484257 392.11366359 394.00291544 395.05218124
                                    396.37181849 398.02226318 399.64148309 400.94377976])
"""
import numpy as np                # 用于数组的创建和赋值
import math                       # 用于计算
from scipy.stats import norm      # 用于计算正态分布（sigma level）
from scipy.stats import binom     # 计算二项分布概率
import random                     # 用于生成随机数据
import multiprocessing            # 多进程运算
import os
import pandas as pd


def task(actual_period, actual_p, actual_pbp, actual_burst_rate, jincheng):
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
        # 数组形状转换
        data_obs_burst = data_obs_burst.reshape(len(data_obs_burst), 1)  # Store the observed burst data.
        data_obs_rate = data_obs_rate.reshape(len(data_obs_rate), 1)  # Store the observed burst count rates.
        data_obs_time = data_obs_time.reshape(len(data_obs_time), 1)  # Store the observation start time.
        data_obs_dur = data_obs_dur.reshape(len(data_obs_dur), 1)  # Store the duration of each observation (in hours).
        data_obs_ifb = data_obs_ifb.reshape(len(data_obs_ifb),
                                            1)  # Store whether an burst occurred during this observation: 1 for burst, 0 for no burst.
        # 返回数据
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

    def period_search(test_start_d, test_end_d, obs_dis):
        """
        Period search process.
        :param test_start_d: Search initial time.
        :param test_end_d: Search end time.
        :param obs_dis: Observation time phase distribution for different search periods.
        :return: p_cumu matrix, T：Corresponding search time series.
        """
        p_cumu = np.zeros((nT, ncp, npw))  # Store the cumulative probability of the binomial distribution.
        fre = np.linspace(1 / test_start_d, 1 / test_end_d, nT)
        T = 1. / fre
        total_rate = Rate_tot
        for t in range(nT):  # Different testing periods/frequencies.
            bp[t, :, :] = phase_folding(T[t], MJD,
                                        n_d)  # Calculate the phase corresponding to the occurrence times of bursts for different central phases in this test period (with the active central phase as 0 phase).
            for i in range(ncp):  # Different central phases.
                for j in range(search_index, search_index + 1):  # Different active region widths.
                    hw = aphw[j]  # Half-width.
                    filter1 = (bp[t, i, :] < hw) | (
                                bp[t, i, :] > 1 - hw)  # Select the bursts located within the active region.
                    count[t, i, j] = len(bp[t, i, filter1])  # The number of bursts within the active region.
                    r[t, i, j] = np.sum(
                        Rate[filter1])  # The sum of the counting rates of bursts within the active region.
                    obs_hw = np.sum(obs_dis[t, i, 0:j + 1]) + np.sum(
                        obs_dis[t, i, -(j + 1):])  # Total observation duration of the active region.
                    p = obs_hw / obs_tot  # Under random assumption, the probability that a single observed burst occurs within the active region is the percentage of the active region observation duration relative to the total observation duration.
                    non_act_burst_num = total_rate - r[t, i, j]
                    if p == 0:
                        non_act_burst_num = total_rate
                    P_binom = binom.cdf(non_act_burst_num, total_rate, 1 - p)

                    p_cumu[t, i, j] = P_binom

        return p_cumu, T

    # Generate simulated data settings.
    simulated_p = actual_p[0]  # Simulate the actual active region phase width of an FRB.
    simulated_pbp = actual_pbp[0]  # Simulate the actual PBP of an FRB.
    simulated_each = 1  # Simulate the observational setup.
    simulated_x = 1  # Simulate the observational setup.
    simulated_obs_day_num = 10  # Simulate the observational setup.
    simulated_obs_hours_all = actual_period[0]   # Simulate the observational setup.
    simulated_burst_rate = actual_burst_rate[0]   # Simulate the burst rate of the bursts.

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
    nT = 100
    nop = 40
    ncp = 20
    n_o = len(obs_day_time_r)
    obs_dis_1 = np.zeros((nT, ncp, nop))  # Used to store the distribution of observation time with respect to phase for different test periods/frequencies and different central active phases.  # 1 to 2-day test period.
    obs_dis_2 = np.zeros((nT, ncp, nop))  # 2 to 4-day test period.
    obs_dis_3 = np.zeros((nT, ncp, nop))  # 4 to 8-day test period.
    obs_dis_4 = np.zeros((nT, ncp, nop))  # 8 to 16-day test period.
    obs_dis_5 = np.zeros((nT, ncp, nop))  # 16 to 32-day test period.
    obs_dis_6 = np.zeros((nT, ncp, nop))  # 32 to 64-day test period.
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
    T = 1. / fre
    for t in range(nT):
        obs_dis_4[t, :, :] = obs_distribution(T[t], obs_day_time_r, obs_dur_time_r)
    # # 16 to 32-day test period.
    fre = np.linspace(1 / 16, 1 / 32, nT)
    T = 1. / fre
    for t in range(nT):
        obs_dis_5[t, :, :] = obs_distribution(T[t], obs_day_time_r, obs_dur_time_r)
    # # 32 to 64-day test period.
    fre = np.linspace(1 / 32, 1 / 64, nT)
    T = 1. / fre
    for t in range(nT):
        obs_dis_6[t, :, :] = obs_distribution(T[t], obs_day_time_r, obs_dur_time_r)


    p_cumu_min_dis = np.zeros((500,))
    p_cumu_min_period_dis = np.zeros((500,))
    for number in range(500):  # Each process runs 500 simulations.
        simulated_t = 381.14
        # Generate simulated data.
        simulated_data = simulated_data_arrays(t=simulated_t,
                                               p=simulated_p,
                                               pbp=simulated_pbp,
                                               each=simulated_each,
                                               x=simulated_x,
                                               obs_day_num=simulated_obs_day_num,
                                               obs_hours_all=simulated_obs_hours_all,
                                               burst_rate=simulated_burst_rate)
        data_obs_burst_s = simulated_data[0]  # Store the observation burst time data.
        while len(data_obs_burst_s) == 0:  # If no bursts are observed, regenerate the burst data (in reality, if nothing is observed, it has no meaning).
            simulated_data = simulated_data_arrays(t=simulated_t,
                                                   p=simulated_p,
                                                   pbp=simulated_pbp,
                                                   each=simulated_each,
                                                   x=simulated_x,
                                                   obs_day_num=simulated_obs_day_num,
                                                   obs_hours_all=simulated_obs_hours_all,
                                                   burst_rate=simulated_burst_rate)
            data_obs_burst_s = simulated_data[0]  # Store the observed burst time data.

        data_obs_rate_s = simulated_data[1]  # Store the observed burst count rate.
        data_obs_time_s = simulated_data[2]  # Store the observation start time.
        data_obs_dur_s = simulated_data[3]  # Store the duration of each observation (in hours).
        data_obs_ifb_s = simulated_data[4]  # Store whether there was a burst in this observation, with 1 for a burst and 0 for no burst.
        # Read the burst times and count rate data from the dataset.
        set1_MJD = data_obs_burst_s  # Burst time data.
        MJD1 = np.array(set1_MJD) - 55000.  # Burst corresponding times in the dataset for simplified calculation.
        d1 = len(MJD1[:, 0])
        set1_Rate = data_obs_rate_s
        r1 = np.array(set1_Rate)
        r1_tot = np.sum(r1)

        # Observation data corresponding to the dataset.
        set1_win = data_obs_time_s  # Observation start time.
        start1 = np.array(set1_win) - 55000.
        d2 = len(start1[:, 0])
        set1_DUR = data_obs_dur_s  # Observation duration.
        dur1 = np.array(set1_DUR)
        obs_tot1 = np.sum(dur1)  # Total observation time.

        # Other variables when setting up cycle analysis.
        aphase = np.linspace(0, 0.95, 20)  # Set the assumed central phase of the strong magnetic field, ranging from 0 to 0.95, with a phase interval of 0.05.
        ncp = 20  # number of center phase

        aphw = np.linspace(0.025, 0.2, 8)  # Set the half-width of the active region to search for, assuming an active region width range of 0.05 to 0.4 phases.
        npw = 8

        obs_phase = np.linspace(0, 0.95, 40)
        nop = 40

        MJD = MJD1
        n_d = d1
        Rate_tot = r1_tot
        Rate = r1
        obs_tot = obs_tot1
        obs_tot = obs_tot
        start = start1
        dur = dur1
        dur = dur
        n_o = d2

        bp = np.zeros((nT, ncp, n_d))  # Used to store the phases corresponding to burst events for different test periods/frequencies and different central active phases.
        count = np.zeros((nT, ncp, npw))  # Used to store the number of bursts within the active region for different test periods/frequencies, central active phases, and active region widths.
        r = np.zeros((nT, ncp, npw))  # Used to store the sum of the count rates of bursts within the active region for different test periods/frequencies, central active phases, and active region widths.

        # Perform period search.
        qk = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        search_index = np.argmin(abs(qk - simulated_p))
        sigma_1_initial, T_1 = period_search(1, 2, obs_dis_1)
        sigma_2_initial, T_2 = period_search(2, 4, obs_dis_2)
        sigma_3_initial, T_3 = period_search(4, 8, obs_dis_3)
        sigma_4_initial, T_4 = period_search(8, 16, obs_dis_4)
        sigma_5_initial, T_5 = period_search(16, 32, obs_dis_5)
        sigma_6_initial, T_6 = period_search(32, 64, obs_dis_6)

        for bk in range(search_index, search_index + 1):
            p_cumu_1 = sigma_1_initial[:, :, bk]
            p_cumu_2 = sigma_2_initial[:, :, bk]
            p_cumu_3 = sigma_3_initial[:, :, bk]
            p_cumu_4 = sigma_4_initial[:, :, bk]
            p_cumu_5 = sigma_5_initial[:, :, bk]
            p_cumu_6 = sigma_6_initial[:, :, bk]

        p_cumu_1_min_value = np.min(p_cumu_1)
        # Obtain the period with the minimum p_cumu.
        min_index = np.argmin(p_cumu_1)
        min_index_2d = np.unravel_index(min_index, p_cumu_1.shape)
        row_index = min_index_2d[0]
        T1 = T_1[row_index]

        p_cumu_2_min_value = np.min(p_cumu_2)
        # Obtain the period with the minimum p_cumu.
        min_index = np.argmin(p_cumu_2)
        min_index_2d = np.unravel_index(min_index, p_cumu_2.shape)
        row_index = min_index_2d[0]
        T2 = T_2[row_index]

        p_cumu_3_min_value = np.min(p_cumu_3)
        # Obtain the period with the minimum p_cumu.
        min_index = np.argmin(p_cumu_3)
        min_index_2d = np.unravel_index(min_index, p_cumu_3.shape)
        row_index = min_index_2d[0]
        T3 = T_3[row_index]

        p_cumu_4_min_value = np.min(p_cumu_4)
        # Obtain the period with the minimum p_cumu.
        min_index = np.argmin(p_cumu_4)
        min_index_2d = np.unravel_index(min_index, p_cumu_4.shape)
        row_index = min_index_2d[0]
        T4 = T_4[row_index]

        p_cumu_5_min_value = np.min(p_cumu_5)
        # Obtain the period with the minimum p_cumu.
        min_index = np.argmin(p_cumu_5)
        min_index_2d = np.unravel_index(min_index, p_cumu_5.shape)
        row_index = min_index_2d[0]
        T5 = T_5[row_index]

        p_cumu_6_min_value = np.min(p_cumu_6)
        # Obtain the period with the minimum p_cumu.
        min_index = np.argmin(p_cumu_6)
        min_index_2d = np.unravel_index(min_index, p_cumu_6.shape)
        row_index = min_index_2d[0]
        T6 = T_6[row_index]

        all_pcumu_min_values = np.array([p_cumu_1_min_value, p_cumu_2_min_value, p_cumu_3_min_value, p_cumu_4_min_value,
                                         p_cumu_5_min_value, p_cumu_6_min_value])  # The minimum value of p_cumu within the search range for each period.

        all_pcumu_min = np.min(all_pcumu_min_values)  # The minimum value of p_cumu across all search ranges.

        all_pcumu_min_index = np.argmin(all_pcumu_min_values)  # The index of the minimum value of p_cumu across all search ranges.
        all_pcumu_min_period = np.array([T1, T2, T3, T4, T5, T6])  # The period corresponding to the minimum value of p_cumu across all search ranges.

        all_period_min = all_pcumu_min_period[all_pcumu_min_index]  # The period with the minimum value of p_cumu across all search ranges.

        p_cumu_min_dis[number] = all_pcumu_min
        p_cumu_min_period_dis[number] = all_period_min

    # Target file path.
    file_path1 = 'D:/PyCharm/Project/Project_FRB/模拟数据/PPBA/高精度周期检测/Pcumu分布/' \
                 '连续观测'+str(simulated_obs_day_num)+'天窗口和暴发率'+str(simulated_burst_rate)+'相宽'+str(simulated_p)+'/最小Pcumu数据'
    file_path2 = 'D:/PyCharm/Project/Project_FRB/模拟数据/PPBA/高精度周期检测/Pcumu分布/' \
                 '连续观测'+str(simulated_obs_day_num)+'天窗口和暴发率'+str(simulated_burst_rate)+'相宽'+str(simulated_p)+'/最小Pcumu的周期'

    if not os.path.exists(file_path1):
        os.makedirs(file_path1)
    if not os.path.exists(file_path2):
        os.makedirs(file_path2)

    np.save('D:/PyCharm/Project/Project_FRB/模拟数据/PPBA/高精度周期检测/Pcumu分布/连续观测'+str(simulated_obs_day_num)+'天窗口和暴发率'
            +str(simulated_burst_rate)+'相宽'+str(simulated_p)+'/最小Pcumu数据/第'+str(
        jincheng)+'个进程的最小Pcumu数据.npy', p_cumu_min_dis)
    np.save('D:/PyCharm/Project/Project_FRB/模拟数据/PPBA/高精度周期检测/Pcumu分布/连续观测'+str(simulated_obs_day_num)+'天窗口和暴发率'
            +str(simulated_burst_rate)+'相宽'+str(simulated_p)+'/最小Pcumu的周期/第' + str(
        jincheng) + '个进程的最小Pcumu周期.npy', p_cumu_min_period_dis)
    print(f"已完成第{jincheng}个进程")


if __name__ == "__main__":

    Obs_dur = np.array([20])   # Used as the total observation duration.
    Actual_p = np.array([0.1])
    Actual_pbp = np.zeros((20,))   # Twenty zeros, representing 20 processes.
    Actual_burst_rate = np.array([50])

    # Create multiple processes.
    processes = []
    for i in range(len(Obs_dur)):
        for j in range(len(Actual_p)):
            for k in range(len(Actual_pbp)):
                for z in range(len(Actual_burst_rate)):
                    p = multiprocessing.Process(target=task, args=(Obs_dur[i:i + 1], Actual_p[j:j + 1],
                                                                   Actual_pbp[k:k + 1], Actual_burst_rate[z:z + 1], k))
                    processes.append(p)

    for n in range(math.ceil(len(processes)/20)):
        start_index = n * 20
        end_index = min((n + 1) * 20, len(processes))  # Ensure that the list index is not out of range.
        for process in processes[start_index:end_index]:
            process.start()
        for process in processes[start_index:end_index]:
            process.join()

    print("任务完成")



