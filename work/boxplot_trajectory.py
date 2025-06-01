#%%
##### import modules

import os
import sys
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

import logging
import matplotlib.pyplot as plt
import pyhurricane.trajectory as traj
import pyhurricane.util as util

#### Logger setting part
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

##### Read yaml file
with open('run_trajectory_conf.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_name       = config['model_name']    # Model name, you can set any name.
tc_name          = config['tc_name']       # Typhoon name, you can set any name.
ex_name          = config['ex_name']       # Experiment name, you can set any name.
plot_kind        = config['plot_kind']     # 'boxplot' or 'scatter'
param            = config['param']         # 'distance' or 'qv', 'rh' ###or hgt

ex_kind          = config["ex_kind"]       # Plot experiments, "individual" or "ensemble"
draw_height      = config["draw_height"]   # Number of height layers to be drawn

prefix           = config["prefix"]
radix            = config["radix"]
suffix           = config["suffix"]

init_data_type        = config["init_data_type"]        # Kinds of setting parcels, 'square' or 'round' but 'square' is not implemented yet.
radius                = config["radius"]                #  Setted parcels in distance from the center of typhoon [m].
track_type            = config["track_type"]            # Kinds of trajectory, 'forward' or 'back' but 'back' is not implemented yet.
track_init_time       = config["track_init_time"]       # Trajectory start time in analysis [hour]
track_base_time_start = config["track_base_time_start"] # Trajectry start time in calculation [hour]
track_time            = config["track_time"]            # Trajectry time in calculation [hour]

input_model_data_dir        = config["input_model_data_dir"]        #Input data and dirctory
input_model_name_suffix     = config["input_model_name_suffix"]
input_trajectry_data_dir    = config["input_trajectry_data_dir"]

case_names            = config["case_names"]
legends               = config["legends"]


########## Main part

#### Output data and dirctory

output_dir_path  = "./"+str(plot_kind[0].upper())+radix+param+"_"+ex_kind
if ex_kind == "individual":
    ex_kind = case_names[0]

if "timeseries" in prefix or "timeseries" in radix or "timeseries" in suffix or "timeseries" in plot_kind:
    save_prefix      = model_name+'-'+tc_name+"-"+ex_name+"-"+str(plot_kind[0].upper())+radix+param+"_"+ex_kind

output_fig_name  = save_prefix
output_fig_type  = config["output_fig_type"]
dpi              = config["dpi"]

#### Add configuration
util.create_dir(output_dir_path)

#### Reading trajectory data and model data.
input_model_data_list = []
input_trajectry_data_list = []
input_model_data_list = [f"{input_model_data_dir}/{case}/"
                        f"{case}.{input_model_name_suffix}" for case in case_names]
input_trajectry_data_list = [
                        f"{input_trajectry_data_dir}/"
                        f"trajectry_data_{case}_T{track_base_time_start}_round_{radius}m_CT{track_time}.npy"
                        for case in case_names
                        ]
"""
input_model_data_list = [f"/media/yoshioka/data/mochida_data/percent_experiment/{case}/"
                        f"evapo_0.0_{case}.pe000000.nc" for case in case_names]
input_trajectry_data_list = [
                        f"/media/yoshioka/data/mochida_data/trajectory/percent_analysis/"
                        f"{case}/track_data_0.0_{case}_T{track_base_time_start}=200km_circle_360points_24hours.npy"
                        for case in case_names
                        ]
"""

###### Select plotting kind function

#####　can use this function for boxplot and scatterplot
#traj.boxplot_distance_interp_trajectory_eachpoint_timeseries(input_trajectry_data_list,input_model_data_list, legends, track_base_time_start, 1, 24, 0.0, 2000.0)
#traj.boxplot_distance_interp_trajectory_eachpoint_timeseries_ens(input_trajectry_data_list,input_model_data_list, case_names, legends, track_base_time_start, 2, 24, 0.0, 2000.0)

traj.boxplot_param_interp_trajectory_eachpoint_timeseries(input_trajectry_data_list, legends, track_base_time_start, 1, 24, 0.0, 2000.0, param)
#traj.boxplot_param_interp_trajectory_eachpoint_timeseries_ens(input_trajectry_data_list,case_names, legends, track_base_time_start, 2, 24, 0.0, 2000.0, param)

#####

##### Save figure
plt.savefig(output_dir_path+'/'+output_fig_name+'.'+output_fig_type, format=output_fig_type, dpi=dpi)
plt.show()

logging.info(f"The figure is saved as {output_dir_path}/{output_fig_name}.{output_fig_type}")
