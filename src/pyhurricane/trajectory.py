##### import modules

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import pyhurricane.util as util


#####
def boxplot_distance_interp_trajectory_eachpoint_timeseries(traj_data: list, model_data: list, legends: list,
                                                            trajectry_inittime: int, skip_hour_time: int, end_hour_time: int, bottom_height: float, top_height: float)  -> plt:
    """
    This function generates a boxplot of the distance between trajectory points and the minimum sea level pressure (MSLP) point over time.
    It takes trajectory data, model data, legends, and other parameters as input and returns a matplotlib plot.
    The boxplot shows the distribution of distances at different time intervals, allowing for visual comparison of the trajectory points' proximity to the MSLP point.

    Args:
        traj_data (list): Trajectory data containing x, y coordinates and height.
        model_data (list): Model data, netcdf file containing MSLP values.
        legends (list): Legend names corresponding to the trajectory data.
        trajectry_inittime (int): Trajectory initial time in hours.
        skip_hour_time (int):  Skip time in hours.
        end_hour_time (int): End time in hours.
        bottom_height (float): Bottom height in meters.
        top_height (float): Top height in meters.

    Returns:
        plt: Matplotlib plot object containing the boxplot.
    """

    base_ds     = xr.open_dataset(model_data[0])

    x_cordinate = base_ds.x
    y_cordinate = base_ds.y
    x_array     = np.array(range(len(x_cordinate)))
    y_array     = np.array(range(len(y_cordinate)))

    ds                = np.load(traj_data[0])
    shape_size        = ds.shape
    time_size         = shape_size[0]
    track_number_size = shape_size[1]

    time_start            = 0                               # Trajectory data start time [hours]
    interval_time         = 60                              # Trajectory data interval time [seconds]
    time_end              = interval_time * end_hour_time   # Trajectory data end time [hours]
    track_base_time_start = trajectry_inittime              # Start time  of analayze trajectory corresponding to model data [hours]

    radius = util.extract_number_before_m(str(traj_data))

    draw_skip_time        = interval_time * skip_hour_time  # minitues

    X_array = ds[:, :, 0]
    Y_array = ds[:, :, 1]
    height  = ds[:, :, 2]

    var_tp = np.zeros((time_size, track_number_size),dtype='float32')

    min_mslp_x, min_mslp_y = util.find_min_mslp_xy(base_ds.MSLP)
    min_mslp_interp_x, min_mslp_interp_y = util.interpolate_time_xy(min_mslp_x, min_mslp_y)
    min_mslp_interp_xcoord = util.interpolate_coord_x(x_cordinate, x_array, min_mslp_interp_x)
    min_mslp_interp_ycoord = util.interpolate_coord_x(y_cordinate, y_array, min_mslp_interp_y)

    min_mslp_Y = min_mslp_interp_ycoord[track_base_time_start*interval_time]
    min_mslp_X = min_mslp_interp_xcoord[track_base_time_start*interval_time]


    for tt in range(time_start, time_end, 1):

        for TRACK_NUMBER in range(0, track_number_size, 1):

            if height[tt, TRACK_NUMBER] >= bottom_height and height[tt, TRACK_NUMBER] <= top_height:
                X_array_distance = np.abs(X_array[tt, TRACK_NUMBER] - min_mslp_X)
                Y_array_distance = np.abs(Y_array[tt, TRACK_NUMBER] - min_mslp_Y)
                var_tp[tt, TRACK_NUMBER] = np.sqrt((X_array_distance)**2 + (Y_array_distance)**2) / 1000

            else:
                var_tp[tt, TRACK_NUMBER] = np.nan

    plt.figure(figsize=(10, 6)) # prepare figure

    hour_axis = np.arange(1, end_hour_time * interval_time // draw_skip_time + 1, 1)
    hour_name = [str(int(HOUR * skip_hour_time  + track_base_time_start)) for HOUR in hour_axis-1]

    valid_distances = []

    for row in var_tp[::interval_time*skip_hour_time, :]:
        valid_distances.append([d for d in row if not np.isnan(d)])

    bp = plt.boxplot(valid_distances)

    plt.xticks(hour_axis,hour_name,fontsize=14)
    plt.xlabel('Calculation time [Hour]',fontsize=18, labelpad=10)

    plt.yticks(np.arange(0,501,100),np.arange(0,501,100),fontsize=14)
    plt.ylabel('Distance [km]',fontsize=18)

    plt.title(f"{legends[0]}, Trajectry : {int(radius) * 0.001}km, from T{trajectry_inittime} to T{trajectry_inittime + end_hour_time - skip_hour_time}, over {bottom_height* 0.001} km under {(top_height * 0.001)}km", fontsize=16)
    plt.grid(True)

    labels = []

    for i, row in enumerate(var_tp[::interval_time*skip_hour_time, :]):
        valid_d = [d for d in row if not np.isnan(d)]
        valid_distances.append(valid_d)
        labels.append(f'{len(valid_d)}') # Make label about numbers of useable data.

    for i, label in enumerate(labels):
        plt.text(bp['medians'][i].get_xdata()[0], 450, label, ha='center', va='top' , fontsize=11) # y=0 の位置を調整する必要があるかもしれません

    plt.tight_layout()

    return plt

#####
def boxplot_distance_interp_trajectory_eachpoint_timeseries_ens(traj_data: list, model_data: list, ex_name: list, legends: list,
                                                                trajectry_inittime: int, skip_hour_time: int, end_hour_time: int, bottom_height: float, top_height: float) -> plt:
    """
    This function generates a boxplot of the distance between trajectory points and the minimum sea level pressure (MSLP) point over time.
    It takes trajectory data, model data, legends, and other parameters as input and returns a matplotlib plot.
    The boxplot shows the distribution of distances at different time intervals, allowing for visual comparison of the trajectory points' proximity to the MSLP point.

    Args:
        traj_data (list): Trajectory data containing x, y coordinates and height.
        model_data (list): Model data, netcdf file containing MSLP values.
        ex_name (list): Experiment names corresponding to the trajectory data
        legends (list): Legend names corresponding to the trajectory data.
        trajectry_inittime (int): Trajectory initial time in hours.
        skip_hour_time (int):  Skip time in hours.
        end_hour_time (int): End time in hours.
        bottom_height (float): Bottom height in meters.
        top_height (float): Top height in meters.

    Returns:
        plt: Matplotlib plot object containing the boxplot for each experiment.
    """

    ds                  = np.load(traj_data[0])
    shape_size          = ds.shape
    time_size           = shape_size[0]
    track_number_size   = shape_size[1]

    time_start            = 0                               # Trajectory data start time [hours]
    interval_time         = 60                              # Trajectory data interval time [seconds]
    time_end              = interval_time * end_hour_time   # Trajectory data end time [hours]
    track_base_time_start = trajectry_inittime              # Start time  of analayze trajectory corresponding to model data [hours]

    radius = util.extract_number_before_m(str(traj_data[0]))

    draw_skip_time        = interval_time * skip_hour_time  # minitues

    var_etp = np.zeros((len(ex_name), time_size, track_number_size), dtype='float32')

    fig, ax = plt.subplots(figsize=(12, 7))     # prepare figure

    for ii in range(len(ex_name)):

        ds      = np.load(traj_data[ii])
        base_ds = xr.open_dataset(model_data[ii])

        x_cordinate = base_ds.x
        y_cordinate = base_ds.y
        x_array = np.array(range(len(x_cordinate)))
        y_array = np.array(range(len(y_cordinate)))

        X_array = ds[:, :, 0]
        Y_array = ds[:, :, 1]
        height  = ds[:, :, 2]

        min_mslp_x, min_mslp_y = util.find_min_mslp_xy(base_ds.MSLP)
        min_mslp_interp_x, min_mslp_interp_y = util.interpolate_time_xy(min_mslp_x, min_mslp_y)
        min_mslp_interp_xcoord = util.interpolate_coord_x(x_cordinate, x_array, min_mslp_interp_x)
        min_mslp_interp_ycoord = util.interpolate_coord_x(y_cordinate, y_array, min_mslp_interp_y)

        min_mslp_Y = min_mslp_interp_ycoord[trajectry_inittime*interval_time]
        min_mslp_X = min_mslp_interp_xcoord[trajectry_inittime*interval_time]

        for tt in range(time_start, time_end, 1):
            for TRACK_NUMBER in range(0, track_number_size, 1):
                if height[tt, TRACK_NUMBER] >= bottom_height and height[tt, TRACK_NUMBER] <= top_height:
                    X_array_distance = np.abs(X_array[tt, TRACK_NUMBER] - min_mslp_X)
                    Y_array_distance = np.abs(Y_array[tt, TRACK_NUMBER] - min_mslp_Y)
                    var_etp[ii, tt, TRACK_NUMBER] = np.sqrt((X_array_distance)**2 + (Y_array_distance)**2) / 1000

                else:
                    var_etp[ii, tt, TRACK_NUMBER] = np.nan

    hour_axis = np.arange(1, end_hour_time * interval_time // draw_skip_time + 1, 1)
    hour_name = [str(int(HOUR * skip_hour_time  + track_base_time_start)) for HOUR in hour_axis-1]

    num_cases = var_etp.shape[0]
    #num_times = var_etp.shape[1]
    #num_tracks = var_etp.shape[2]
    num_times_to_plot = int(end_hour_time * interval_time / draw_skip_time)

    print(f"num_times_to_plot = {num_times_to_plot}")

    positions = []
    box_data = []
    x_pos = np.arange(1, num_cases + 1)

    for t_idx in range(num_times_to_plot):
        for case_idx in range(num_cases):
            row = var_etp[case_idx, t_idx * draw_skip_time, :]
            valid_d = [d for d in row if not np.isnan(d)]
            if valid_d:
                box_data.append(valid_d)
                positions.append(t_idx * (num_cases + 1) + x_pos[case_idx])

    print(f"Length of box_data : {len(box_data)}")
    print(f"Length of positions : {len(positions)}")

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)

    cmap = plt.cm.tab10  # colormap selection

    for i, patch in enumerate(bp['boxes']):
        if i % num_cases == 0:
            color = 'black'
        else:
            color_index = i % num_cases
            color = cmap(color_index)
        patch.set_facecolor(color)

    # Tick label setup
    ax.set_xticks(np.arange(num_times_to_plot) * (num_cases + 1) + (num_cases + 1) / 2)
    ax.set_xticklabels(hour_name, fontsize=14)
    ax.set_xlabel('Calculation time [Hour]', fontsize=18, labelpad=10)

    ax.set_yticks(np.arange(0, 501, 100))
    ax.set_yticklabels(np.arange(0, 501, 100), fontsize=14)
    ax.set_ylabel('Distance [km]', fontsize=18)

    ax.set_title(f"Trajectry : {int(radius) * 0.001}km, from T{trajectry_inittime} to T{trajectry_inittime + end_hour_time - skip_hour_time}, over {bottom_height* 0.001} km under {(top_height * 0.001)}km", fontsize=20)
    ax.grid(True)

    # Legend setup
    handles = bp['boxes'][0:num_cases] #  Include the first time step boxes in the legend
    ax.legend(handles, legends, loc='upper left', fontsize=12)

    plt.tight_layout()

    return fig

#####
def boxplot_param_interp_trajectory_eachpoint_timeseries(traj_data: list, legends: list,
                                                            trajectry_inittime: int, skip_hour_time: int, end_hour_time: int, bottom_height: float, top_height: float, param: str)  -> plt:
    """

    Args:
        traj_data (list): Trajectory data containing x, y coordinates and height.
        model_data (list): Model data, netcdf file containing MSLP values.
        legends (list): Legend names corresponding to the trajectory data.
        trajectry_inittime (int): Trajectory initial time in hours.
        skip_hour_time (int):  Skip time in hours.
        end_hour_time (int): End time in hours.
        bottom_height (float): Bottom height in meters.
        top_height (float): Top height in meters.
        param (str):

    Returns:
        plt: Matplotlib plot object containing the boxplot.
    """

    ds                = np.load(traj_data[0])
    shape_size        = ds.shape
    time_size         = shape_size[0]
    track_number_size = shape_size[1]

    time_start            = 0                               # Trajectory data start time [hours]
    interval_time         = 60                              # Trajectory data interval time [seconds]
    time_end              = interval_time * end_hour_time   # Trajectory data end time [hours]
    track_base_time_start = trajectry_inittime              # Start time  of analayze trajectory corresponding to model data [hours]

    radius = util.extract_number_before_m(str(traj_data))

    draw_skip_time        = interval_time * skip_hour_time  # minitues

    var_tp = np.zeros((time_size, track_number_size),dtype='float32')
    fig, ax = plt.subplots(figsize=(10, 6))   # prepare figure

    height  = ds[:, :, 2]
    if param == 'qv':
        var_array = ds[:, :, 6]
    elif param == 'rh':
        var_array = ds[:, :, 7]
    elif param == 'hgt':
        var_array = ds[:, :, 2] * 0.001
        top_height = 10000000.0

    for tt in range(time_start, time_end, 1):
        for TRACK_NUMBER in range(0, track_number_size, 1):
            if height[tt, TRACK_NUMBER] >= bottom_height and height[tt, TRACK_NUMBER] <= top_height:
                var_tp[tt, TRACK_NUMBER] = var_array[tt, TRACK_NUMBER]
            else:
                var_tp[tt, TRACK_NUMBER] = np.nan

    hour_axis = np.arange(1, end_hour_time * interval_time // draw_skip_time + 1, 1)
    hour_name = [str(int(HOUR * skip_hour_time  + track_base_time_start)) for HOUR in hour_axis-1]

    valid_distances = []

    for row in var_tp[::interval_time*skip_hour_time, :]:
        valid_distances.append([d for d in row if not np.isnan(d)])

    bp = ax.boxplot(valid_distances)

    # Tick label setup
    ax.set_xticks(hour_axis,hour_name,fontsize=14)
    ax.set_xticklabels(hour_name, fontsize=14)
    ax.set_xlabel('Calculation time [Hour]', fontsize=18, labelpad=10)

    if param == 'qv':
        ax.set_ylabel('Spesific Humidity [$kg kg^{-1}$]', fontsize=18)
        ax.set_yticks(np.arange(0.0, 0.0301, 0.005))
        ax.set_yticklabels(np.arange(0.0, 0.0301, 0.005), fontsize=14)
        label_pos_y = 0.027
    elif param == 'rh':
        ax.set_ylabel('Relative Humidity [%]', fontsize=18)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_yticklabels(np.arange(0, 101, 10), fontsize=14)
        label_pos_y = 90
    elif param == 'hgt':
        ax.set_ylabel('Height [km]', fontsize=18)
        ax.set_yticks(np.arange(0, 23.1, 2))
        ax.set_yticklabels(np.arange(0, 23.1, 2), fontsize=14)
        label_pos_y = 22.0

    ax.set_title(f"Trajectry : {int(radius) * 0.001}km, from T{trajectry_inittime} to T{trajectry_inittime + end_hour_time - skip_hour_time}, over {bottom_height* 0.001} km under {(top_height * 0.001)}km", fontsize=20)
    ax.grid(True)

    labels = []

    for i, row in enumerate(var_tp[::interval_time*skip_hour_time, :]):
        valid_d = [d for d in row if not np.isnan(d)]
        valid_distances.append(valid_d)
        labels.append(f'{len(valid_d)}') # Make label about numbers of useable data.

    for i, label in enumerate(labels):
        ax.text(bp['medians'][i].get_xdata()[0], label_pos_y, label, ha='center', va='top' , fontsize=11)

    plt.tight_layout()

    return fig

#####
def boxplot_param_interp_trajectory_eachpoint_timeseries_ens(traj_data: list, ex_name: list, legends: list,
                                                            trajectry_inittime: int, skip_hour_time: int, end_hour_time: int, bottom_height: float, top_height: float, param: str) -> plt:



    ds                  = np.load(traj_data[0])
    shape_size          = ds.shape
    time_size           = shape_size[0]
    track_number_size   = shape_size[1]

    time_start            = 0                               # Trajectory data start time [hours]
    interval_time         = 60                              # Trajectory data interval time [seconds]
    time_end              = interval_time * end_hour_time   # Trajectory data end time [hours]
    track_base_time_start = trajectry_inittime              # Start time  of analayze trajectory corresponding to model data [hours]

    radius = util.extract_number_before_m(str(traj_data[0]))

    draw_skip_time        = interval_time * skip_hour_time  # minitues

    var_etp = np.zeros((len(ex_name), time_size, track_number_size), dtype='float32')

    fig, ax = plt.subplots(figsize=(12, 7))     # prepare figure

    for ii in range(len(ex_name)):

        ds      = np.load(traj_data[ii])
        height  = ds[:, :, 2]
        if param == 'qv':
            var_array = ds[:, :, 6]
        elif param == 'rh':
            var_array = ds[:, :, 7]
        elif param == 'hgt':
            var_array = ds[:, :, 2] * 0.001
            top_height = 10000000.0

        for tt in range(time_start, time_end, 1):
            for TRACK_NUMBER in range(0, track_number_size, 1):
                if height[tt, TRACK_NUMBER] >= bottom_height and height[tt, TRACK_NUMBER] <= top_height:
                    var_etp[ii, tt, TRACK_NUMBER] = var_array[tt, TRACK_NUMBER]
                else:
                    var_etp[ii, tt, TRACK_NUMBER] = np.nan

    hour_axis = np.arange(1, end_hour_time * interval_time // draw_skip_time + 1, 1)
    hour_name = [str(int(HOUR * skip_hour_time  + track_base_time_start)) for HOUR in hour_axis-1]

    num_cases = var_etp.shape[0]
    #num_times = var_etp.shape[1]
    #num_tracks = var_etp.shape[2]
    num_times_to_plot = int(end_hour_time * interval_time / draw_skip_time)

    print(f"num_times_to_plot = {num_times_to_plot}")

    positions = []
    box_data = []
    x_pos = np.arange(1, num_cases + 1)

    for t_idx in range(num_times_to_plot):
        for case_idx in range(num_cases):
            row = var_etp[case_idx, t_idx * draw_skip_time, :]
            valid_d = [d for d in row if not np.isnan(d)]
            if valid_d:
                box_data.append(valid_d)
                positions.append(t_idx * (num_cases + 1) + x_pos[case_idx])

    print(f"Length of box_data : {len(box_data)}")
    print(f"Length of positions : {len(positions)}")

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)

    cmap = plt.cm.tab10  # colormap selection

    for i, patch in enumerate(bp['boxes']):
        if i % num_cases == 0:
            color = 'black'
        else:
            color_index = i % num_cases
            color = cmap(color_index)
        patch.set_facecolor(color)

    # Tick label setup
    ax.set_xticks(np.arange(num_times_to_plot) * (num_cases + 1) + (num_cases + 1) / 2)
    ax.set_xticklabels(hour_name, fontsize=14)
    ax.set_xlabel('Calculation time [Hour]', fontsize=18, labelpad=10)

    if param == 'qv':
        ax.set_ylabel('Spesific Humidity [$kg kg^{-1}$]', fontsize=18)
        ax.set_yticks(np.arange(0.0, 0.0301, 0.005))
        ax.set_yticklabels(np.arange(0.0, 0.0301, 0.005), fontsize=14)
    elif param == 'rh':
        ax.set_ylabel('Relative Humidity [%]', fontsize=18)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_yticklabels(np.arange(0, 101, 10), fontsize=14)
    elif param == 'hgt':
        ax.set_ylabel('Height [km]', fontsize=18)
        ax.set_yticks(np.arange(0, 23.1, 2))
        ax.set_yticklabels(np.arange(0, 23.1, 2), fontsize=14)

    ax.set_title(f"Trajectry : {int(radius) * 0.001}km, from T{trajectry_inittime} to T{trajectry_inittime + end_hour_time - skip_hour_time}, over {bottom_height* 0.001} km under {(top_height * 0.001)}km", fontsize=20)
    ax.grid(True)

    # Legend setup
    handles = bp['boxes'][0:num_cases] #  Include the first time step boxes in the legend
    ax.legend(handles, legends, loc='best', fontsize=12)

    plt.tight_layout()

    return fig



def example_function(arg1: int, arg2: str) -> bool:
    # この行にカーソルを置いて """ (3つのダブルクォート) を入力
    pass