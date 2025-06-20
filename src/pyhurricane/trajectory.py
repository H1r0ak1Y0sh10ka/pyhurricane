##### import modules

import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pyhurricane.util as util

#### Logger setting part        ;
logger = logging.getLogger(__name__)
#logging.basicConfig(filename=log_dir+log_name,level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

########## Trajectory analyis

#####
def trajectory_analysis_round(model_data,
                        track_base_it,
                        track_time_Tsize,
                        track_time_delta,
                        track_type,
                        radius,
                        theta_interval,
                        track_base_Z,
                        track_variable_size=8
                        ):
    ds =  None

    try:
        logger.info(f"Attempting to open dataset with chunks: {model_data}")
        #ds = xr.open_dataset(model_data, chunks={'time': 1, 'z': 'auto', 'y': 'auto', 'x': 'auto'})
        ds = xr.open_dataset(model_data, chunks={'time': 1, 'z': 5, 'y': 50, 'x': 50}) 
        logger.info(f"Successfully opened {model_data} with chunks.")
        #logger.info(f"Dataset info: {ds}")
    except Exception as e_chunk:
        logger.warning(f"Failed to open dataset {model_data} with chunks: {e_chunk}. Trying without chunks.")
        try:
            ds = xr.open_dataset(model_data)
            logger.info(f"Successfully opened {model_data} without chunks.")
            logger.info(f"Dataset info (no chunks): {ds}")
        except Exception as e_no_chunk:
            logger.error(f"Failed to open dataset {model_data} even without chunks: {e_no_chunk}")
            return # データセットが開けない場合は処理を中断

    if ds is None:
        logger.error("Dataset could not be opened. Aborting trajectory analysis.")
        return

    try:
        track_base_time = ds.time[track_base_it].values
        x_coord = ds.x
        y_coord = ds.y
        z_coord = ds.z
        logger.info(f"Coordinate sizes: x={x_coord.size}, y={y_coord.size}, z={z_coord.size}, time={ds.time.size}")
        x_array = np.array(range(len(x_coord)))
        y_array = np.array(range(len(y_coord)))
        z_array = np.array(range(len(z_coord)))

        # 最小値を取る x と y 座標を求める
        # MSLPデータもチャンクされている可能性があるため、.load() で明示的にメモリに読み込むか、
        # find_min_mslp_xy がチャンクされたDataArrayを扱えるようにする
        mslp_data = ds.MSLP
        logger.info(f"MSLP data shape: {mslp_data.shape}, dtype: {mslp_data.dtype}, chunks: {mslp_data.chunks}")
        logger.info(f"MSLP data estimated size before any operation: {mslp_data.nbytes / 1e6:.2f} MB")
        # The following .load() call is a major suspect for OOM errors and should generally be avoided
        # if hasattr(mslp_data, 'chunks') and mslp_data.chunks is not None:
        #     logger.info("Loading MSLP data into memory for min/max operations.")
        #     mslp_data = mslp_data.load() # This line is problematic for large datasets

        min_mslp_x, min_mslp_y = util.find_min_mslp_xy(mslp_data)
        logger.info(f"Calculated min_mslp_x (shape: {min_mslp_x.shape}, size: {min_mslp_x.nbytes / 1e6:.2f} MB)")
        logger.info(f"Calculated min_mslp_y (shape: {min_mslp_y.shape}, size: {min_mslp_y.nbytes / 1e6:.2f} MB)")

        # --- ステップ1の終了点 ---
        #logger.info("Step 1 finished: MSLP min coordinates found.")
        # ds.close() # finallyブロックで実行されるのでここでは不要
        #return # ここで一旦処理を終了してメモリ状況を確認
        # -------------------------
        # 1 分ごとに内挿
        min_mslp_interp_x, min_mslp_interp_y = util.interpolate_time_xy(min_mslp_x, min_mslp_y)
        logger.info(f"Interpolated min_mslp_interp_x (shape: {min_mslp_interp_x.shape}, size: {min_mslp_interp_x.nbytes / 1e6:.2f} MB)")
        logger.info(f"Interpolated min_mslp_interp_y (shape: {min_mslp_interp_y.shape}, size: {min_mslp_interp_y.nbytes / 1e6:.2f} MB)")
        # 新しい x 座標に対応する Xcoord を計算
        min_mslp_interp_xcoord = util.interpolate_coord_x(x_coord.values, x_array, min_mslp_interp_x) # .valuesでNumPy配列として渡す
        min_mslp_interp_ycoord = util.interpolate_coord_x(y_coord.values, y_array, min_mslp_interp_y) # .valuesでNumPy配列として渡す
        logger.info(f"Interpolated min_mslp_interp_xcoord (shape: {min_mslp_interp_xcoord.shape}, size: {min_mslp_interp_xcoord.nbytes / 1e6:.2f} MB)")
        logger.info(f"Interpolated min_mslp_interp_ycoord (shape: {min_mslp_interp_ycoord.shape}, size: {min_mslp_interp_ycoord.nbytes / 1e6:.2f} MB)")

        # --- ステップ2の終了点 ---
        #logger.info("Step 2 finished: MSLP min coordinates interpolated and converted to physical coords.")
        #return # ここで一旦処理を終了してメモリ状況を確認
        # -------------------------

        theta = np.arange(0, 2*np.pi+1e-5, np.deg2rad(theta_interval))[:-1]
        track_time_size = int(track_time_Tsize * 3600 / track_time_delta)
        track_data      = np.zeros((track_time_size, theta.size, track_variable_size), dtype=np.float32)
        logger.info(f"Initialized track_data with shape {track_data.shape}, dtype: {track_data.dtype}, estimated size: {track_data.nbytes / 1e6:.2f} MB")

        min_mslp_org_xcoord = min_mslp_interp_xcoord[track_base_it*60]
        min_mslp_org_ycoord = min_mslp_interp_ycoord[track_base_it*60]

        #トラックナンバーごとに中心からの距離を求めて座標を作成する
        track_xs_np = min_mslp_org_xcoord + (radius * np.cos(theta))
        track_ys_np = min_mslp_org_ycoord + (radius * np.sin(theta))
        track_zs_np = np.full(theta.size, z_coord[track_base_Z].item(), dtype=np.float32)

        #トラックナンバーごとに中心からの距離を求めて座標を作成する
        track_xs_xr = xr.DataArray(track_xs_np, dims=["track_number"])
        track_ys_xr = xr.DataArray(track_ys_np, dims=["track_number"])
        track_zs_xr = xr.DataArray(track_zs_np, dims=["track_number"])

        track_data[0,:,0] = track_xs_np
        track_data[0,:,1] = track_ys_np
        track_data[0,:,2] = track_zs_np
        # isel().interp() はチャンクされたデータに対して効率的に動作するはず
        logger.info("Populating initial track data (U, V, W, QV, RH) using interp.")
        track_data[0,:,3] = ds.U.isel(time=track_base_it).interp(x=track_xs_xr, y=track_ys_xr, z=track_zs_xr).values
        track_data[0,:,4] = ds.V.isel(time=track_base_it).interp(x=track_xs_xr, y=track_ys_xr, z=track_zs_xr).values
        track_data[0,:,5] = ds.W.isel(time=track_base_it).interp(x=track_xs_xr, y=track_ys_xr, z=track_zs_xr).values
        track_data[0,:,6] = ds.QV.isel(time=track_base_it).interp(x=track_xs_xr, y=track_ys_xr, z=track_zs_xr).values
        track_data[0,:,7] = ds.RH.isel(time=track_base_it).interp(x=track_xs_xr, y=track_ys_xr, z=track_zs_xr).values

        # --- ステップ3の終了点 ---
        logger.info("Step 3 finished: Initial track data populated.")
        #return # ここで一旦処理を終了してメモリ状況を確認
        # -------------------------

        #フォワードトラジェクトリー解析
        if track_type == 'forward':
            #風速等を使って次の時間のトラックデータ座標を求める
            for PASS_SEC in range(track_time_delta, track_time_Tsize * 3600, track_time_delta):
                logger.info('Now timestep is ' + str(PASS_SEC) )
                track_time_dt64 = track_base_time + np.timedelta64(PASS_SEC, 's') # 変数名を変更して明確化
                pass_time_step = int(PASS_SEC/track_time_delta)

                x_coord_u_deltas = track_data[(pass_time_step)-1, :, 3] * track_time_delta
                y_coord_v_deltas = track_data[(pass_time_step)-1, :, 4] * track_time_delta
                z_coord_w_deltas = track_data[(pass_time_step)-1, :, 5] * track_time_delta

                #track_xs = xr.DataArray(track_data[(pass_time_step)-1, :, 0] + x_coord_u_deltas, dims=["track_number"])
                #track_ys = xr.DataArray(track_data[(pass_time_step)-1, :, 1] + y_coord_v_deltas, dims=["track_number"])
                #track_zs = xr.DataArray(track_data[(pass_time_step)-1, :, 2] + z_coord_w_deltas, dims=["track_number"])
                #track_data[pass_time_step, :, 0] = track_xs.values
                #track_data[pass_time_step, :, 1] = track_ys.values
                #track_data[pass_time_step, :, 2] = track_zs.values
                current_track_xs_np = track_data[(pass_time_step)-1, :, 0] + x_coord_u_deltas
                current_track_ys_np = track_data[(pass_time_step)-1, :, 1] + y_coord_v_deltas
                current_track_zs_np = track_data[(pass_time_step)-1, :, 2] + z_coord_w_deltas
                track_data[pass_time_step, :, 0] = current_track_xs_np
                track_data[pass_time_step, :, 1] = current_track_ys_np
                track_data[pass_time_step, :, 2] = current_track_zs_np

                #track_zs_intp = track_zs.clip(min=z_coord[0], max=z_coord[-1])

                track_xs_for_interp_xr = xr.DataArray(current_track_xs_np, dims=["track_number"])
                track_ys_for_interp_xr = xr.DataArray(current_track_ys_np, dims=["track_number"])
                track_zs_for_interp_xr = xr.DataArray(current_track_zs_np, dims=["track_number"])
                track_zs_clipped_xr = track_zs_for_interp_xr.clip(min=z_coord[0].item(), max=z_coord[-1].item())

                track_data[pass_time_step, :, 3] = ds.U.interp(time=track_time_dt64, x=track_xs_for_interp_xr, y=track_ys_for_interp_xr, z=track_zs_clipped_xr, method='linear').values
                track_data[pass_time_step, :, 4] = ds.V.interp(time=track_time_dt64, x=track_xs_for_interp_xr, y=track_ys_for_interp_xr, z=track_zs_clipped_xr, method='linear').values
                track_data[pass_time_step, :, 5] = ds.W.interp(time=track_time_dt64, x=track_xs_for_interp_xr, y=track_ys_for_interp_xr, z=track_zs_clipped_xr, method='linear').values
                track_data[pass_time_step, :, 6] = ds.QV.interp(time=track_time_dt64, x=track_xs_for_interp_xr, y=track_ys_for_interp_xr, z=track_zs_clipped_xr, method='linear').values
                track_data[pass_time_step, :, 7] = ds.RH.interp(time=track_time_dt64, x=track_xs_for_interp_xr, y=track_ys_for_interp_xr, z=track_zs_clipped_xr, method='linear').values
                # --- ステップ4の途中終了点 ---
                #if pass_time_step >= 3: # 例えば3ステップで終了
                #    logger.info(f"Step 4: Loop interrupted after {pass_time_step} steps for testing.")
                #    break
                # -------------------------
            logger.info('************* Forward-trajectory calculation has been finished !! *************')

        #elif track_type == 'back':
            #風速等を使って次の時間のトラックデータ座標を求める
        #    for PASS_SEC in range(track_time_delta, track_time_Tsize * 3600, track_time_delta):
                #track_time = track_base_time + PASS_SEC
        #        track_time = track_base_time + np.timedelta64(PASS_SEC, 's')
        #        pass_time_step = int(PASS_SEC/track_time_delta)
        elif track_type == 'back':
            #風速等を使って次の時間のトラックデータ座標を求める
            for PASS_SEC in range(track_time_delta, track_time_Tsize * 3600, track_time_delta):
                track_time_dt64 = track_base_time + np.timedelta64(PASS_SEC, 's') # 変数名を変更して明確化
                pass_time_step = int(PASS_SEC/track_time_delta)

                #x_coord_u_deltas = track_data[(pass_time_step)-1, :, 3] * -track_time_delta
                #y_coord_v_deltas = track_data[(pass_time_step)-1, :, 4] * -track_time_delta
                #z_coord_w_deltas = track_data[(pass_time_step)-1, :, 5] * -track_time_delta
                x_coord_u_deltas = track_data[(pass_time_step)-1, :, 3] * -track_time_delta
                y_coord_v_deltas = track_data[(pass_time_step)-1, :, 4] * -track_time_delta
                z_coord_w_deltas = track_data[(pass_time_step)-1, :, 5] * -track_time_delta


                #track_xs = xr.DataArray(track_data[(pass_time_step)-1, :, 0] + x_coord_u_deltas, dims=["track_number"])
                #track_ys = xr.DataArray(track_data[(pass_time_step)-1, :, 1] + y_coord_v_deltas, dims=["track_number"])
                #track_zs = xr.DataArray(track_data[(pass_time_step)-1, :, 2] + z_coord_w_deltas, dims=["track_number"])
                #track_data[pass_time_step, :, 0] = track_xs.values
                #track_data[pass_time_step, :, 1] = track_ys.values
                #track_data[pass_time_step, :, 2] = track_zs.values
                current_track_xs_np = track_data[(pass_time_step)-1, :, 0] + x_coord_u_deltas
                current_track_ys_np = track_data[(pass_time_step)-1, :, 1] + y_coord_v_deltas
                current_track_zs_np = track_data[(pass_time_step)-1, :, 2] + z_coord_w_deltas
                track_data[pass_time_step, :, 0] = current_track_xs_np
                track_data[pass_time_step, :, 1] = current_track_ys_np
                track_data[pass_time_step, :, 2] = current_track_zs_np

                #track_zs_intp = track_zs.clip(min=z_coord[0], max=z_coord[-1])
                track_xs_for_interp_xr = xr.DataArray(current_track_xs_np, dims=["track_number"])
                track_ys_for_interp_xr = xr.DataArray(current_track_ys_np, dims=["track_number"])
                track_zs_for_interp_xr = xr.DataArray(current_track_zs_np, dims=["track_number"])
                track_zs_clipped_xr = track_zs_for_interp_xr.clip(min=z_coord[0].item(), max=z_coord[-1].item())

                #track_data[pass_time_step, :, 3] = ds.U.interp(time=track_time, x=track_xs, y=track_ys, z=track_zs_intp).values
                #track_data[pass_time_step, :, 4] = ds.V.interp(time=track_time, x=track_xs, y=track_ys, z=track_zs_intp).values
                #track_data[pass_time_step, :, 5] = ds.W.interp(time=track_time, x=track_xs, y=track_ys, z=track_zs_intp).values
                #track_data[pass_time_step, :, 6] = ds.QV.interp(time=track_time, x=track_xs, y=track_ys, z=track_zs_intp).values
                #track_data[pass_time_step, :, 7] = ds.RH.interp(time=track_time, x=track_xs, y=track_ys, z=track_zs_intp).values
            #logger.info('************* Back-trajectory calculation has been finished !! *************')
                # バックトラジェクトリでは method='cubic' が指定されていなかったため、デフォルトの線形補間になります。
                # 必要であれば 'cubic' を指定してください。
                track_data[pass_time_step, :, 3] = ds.U.interp(time=track_time_dt64, x=track_xs_for_interp_xr, y=track_ys_for_interp_xr, z=track_zs_clipped_xr).values
                track_data[pass_time_step, :, 4] = ds.V.interp(time=track_time_dt64, x=track_xs_for_interp_xr, y=track_ys_for_interp_xr, z=track_zs_clipped_xr).values
                track_data[pass_time_step, :, 5] = ds.W.interp(time=track_time_dt64, x=track_xs_for_interp_xr, y=track_ys_for_interp_xr, z=track_zs_clipped_xr).values
                track_data[pass_time_step, :, 6] = ds.QV.interp(time=track_time_dt64, x=track_xs_for_interp_xr, y=track_ys_for_interp_xr, z=track_zs_clipped_xr).values
                track_data[pass_time_step, :, 7] = ds.RH.interp(time=track_time_dt64, x=track_xs_for_interp_xr, y=track_ys_for_interp_xr, z=track_zs_clipped_xr).values
            logger.info('************* Back-trajectory calculation has been finished !! *************')

        logger.info(f"First trajectory's X coordinates: {track_data[:,0,0]}") # より詳細なログ

    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()
            logger.info(f"Closed dataset: {model_data}")

    #logger.info(track_data[:,0,0])
    #np.save(save_path + save_name, track_data)

########## Make figure

#####
def timeseries_trajectory_ens(traj_data: list, legends,
                            track_base_it: int,
                            time_start=0,
                            time_end=60*24,
                            alpha=None):
    """timeseries_trajectory_ens _summary_

    This function

    Args:
        traj_data (list): Trajectory data containing x, y coordinates and height.
        label (_type_): _description_
        track_base_it (int): _description_
        time_start (int, optional): _description_. Defaults to 0.
        time_end (_type_, optional): _description_. Defaults to 60*24.
        alpha (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    ds = np.load(traj_data[0])
    QV_size = ds.shape
    time_size = QV_size[0]
    cmap = cm.tab10

    time_axis = np.arange(time_start, time_end, 1)
    var_et = np.zeros((len(legends), time_size),dtype='float32')
    print('var_et.shape=', var_et.shape)

    for ii in range(len(legends)):
        ds = np.load(traj_data[ii])
        QV = ds[:, :, 6]
        QV_mean = np.zeros((time_size))

        for tt in range(time_start, time_end, 1):
            var_et[ii,tt - time_start] = np.mean(QV[tt, :]) * 1000

        if ii == 0:
            plt.plot(time_axis, var_et[0,:], linestyle='-', linewidth=3, color='black', label=legends[ii], alpha=alpha)
        else:
            plt.plot(time_axis, var_et[ii,:], linestyle='-', linewidth=3, color=cmap(ii/len(legends)), label=legends[ii], alpha=alpha)

    # Tick label setup
    plt.ylabel('Specific Humidity [g/kg]',fontsize=18,labelpad=10)
    plt.yticks(np.arange(0, 30.1, 5), np.arange(0, 30.1, 5))

    hour_axis = np.arange(0, time_end+1, 120)
    hour_name = [str(int((HOUR / 60) + track_base_it)) for HOUR in hour_axis]
    plt.xlabel('Calculation time [Hour]',fontsize=18,labelpad=10)
    plt.xticks(hour_axis, hour_name)

    # Legend setup
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=15)

    plt.grid(True)
    plt.tight_layout()

    return plt

#####
def boxplot_distance_interp_trajectory_eachpoint_timeseries(traj_data: list, model_data: list, legends: list,
                                                            trajectry_inittime: int,
                                                            skip_hour_time: int, end_hour_time: int,
                                                            bottom_height: float, top_height: float)  -> plt:
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