##### ----- import modules

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyhurricane.util as util
import tropycal.tracks as tracks

##### ----- Logger setting part
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

#####
def plot_timeseries(
                    trackdata_list: list[str],
                    #x_axis: float = None,
                    y_axis: str   = None,
                    legends: list[str]   = None,
                    dpi: int = 350
                    ):
    parameter = "mslp"
    n_cases = len(trackdata_list)
    figx = 13
    figy = 6
    cmap = cm.tab10
    plt.figure(figsize=(figx,figy))

    # Initialize record_count before iterating through files to find the minimum length
    record_count = float('inf')

    for ne in range(1,n_cases,1):
        trackdata =  pd.read_csv(trackdata_list[ne],index_col=0)
            # レコード長がNよりも小さい場合、Nを更新
        #if util.count_csv_records(trackdata) < record_count:
        #    record_count = util.ount_csv_records(trackdata)
        if util.count_csv_records(trackdata_list[ne]) < record_count: # trackdata を trackdata_list[ne] に変更
            record_count = util.count_csv_records(trackdata_list[ne])  # 同様に変更し、タイポも修正

            print("レコード数:", record_count)
            x_axis = np.linspace(0, record_count-1, record_count, dtype = 'int')

        var_t = trackdata[parameter].tolist()
        time_t = pd.to_datetime(trackdata['time'])

        plt.plot(x_axis, var_t, marker='o', linestyle='-', color=cmap(ne/(n_cases)), label=legends[ne])

    fontsize = 16

    if parameter == 'mslp' :
        plt.ylabel('Mean Sea Level Pressure [hPa]' ,fontsize=16)
        plt.ylim([910,1010])
        plt.yticks([920,940,960,980,1000],fontsize=16)

    plt.xlabel('Calicuration Time [hour]', fontsize=fontsize)
    plt.xlim([0,record_count-1])
    plt.xticks(np.arange(0, record_count, 12), fontsize=fontsize)

    plt.grid(True)
    plt.show()