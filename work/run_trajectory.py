#%%
##### import modules

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

import logging
import matplotlib.pyplot as plt
import pyhurricane.trajectory as traj
import pyhurricane.util as util

#### Logger setting part
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

########## Main part

##### Configuration
model_name       = "scale"       # Model name, you can set any name.
tc_name          = "faxai"       # Typhoon name, you can set any name.
ex_name          = "cd_radius"   # Experiment name, you can set any name.
plot_kind        = 'boxplot'     # 'boxplot' or 'scatter'
param            = 'qv'          # 'distance' or 'qv', 'rh' ###or hgt

ex_kind          = "individual"   # Plot experiments, "individual" or "ensemble"
draw_height      = 1             # Number of height layers to be drawn

prefix           = ""
radix            = "_trajectory_timeseries_"
suffix           = ""

#### Input data and dirctory
init_data_type        = 'round'     # Kinds of setting parcels, 'square' or 'round' but 'square' is not implemented yet.
radius                = 250000      #  Setted parcels in distance from the center of typhoon [m].
track_type            = 'forward'   # Kinds of trajectory, 'forward' or 'back' but 'back' is not implemented yet.
track_init_time       = 0           # Trajectory start time in analysis [hour]
track_base_time_start = 18          # Trajectry start time in calculation [hour]
track_time            = 24          # Trajectry time in calculation [hour]
tack_time_delta       = 60

# Please set the directory and file name of the model data and trajectory data in your environment.
input_model_data_dir        = "/mnt/data1/model/scale/faxai/fnl/flow/"
input_model_name_suffix     = "pe000000.nc"
#input_trajectry_data_dir    = "/home/yoshioka/research/analysis/python/scale/trajectory/trajectory_data_360/"

#case_names            = ['CTL','Cd1730','Cd1530','Cd1330','Cd1130','Cd1030'] #case_names = ('CTL','25km','50km','75km','100km','150km','200km')
#legends               = ['CTL','R025km','R050km','R100km','R150km','R200km'] #legends = case_names
case_names = ['CTL']
legends    = ['CTL']

#### Output data and dirctory

#output_dir_path  = "./"+str(plot_kind[0].upper())+radix+param+"_"+ex_kind
#if ex_kind == "individual":
#    ex_kind = case_names[0]

#if "timeseries" in prefix or "timeseries" in radix or "timeseries" in suffix or "timeseries" in plot_kind:
#    save_prefix      = model_name+'-'+tc_name+"-"+ex_name+"-"+str(plot_kind[0].upper())+radix+param+"_"+ex_kind

#output_fig_name  = save_prefix
#output_fig_type  = 'png'
#dpi              = 350

#### Add configuration
#util.create_dir(output_dir_path)

#### Reading trajectory data and model data.
input_model_data_list = []
input_model_data_list = [f"{input_model_data_dir}/{case}/"
                        f"{case}.{input_model_name_suffix}" for case in case_names]

###### Select plotting kind function

#####　can use this function for boxplot and scatterplot
traj.trajectory_analysis_round(input_model_data_list[0],
                            track_base_time_start,
                            track_time,
                            tack_time_delta,
                            track_type,
                            radius,
                            theta_interval=1,
                            track_base_Z=0,
                            track_variable_size=8
                            )

#logging.info(f"The figure is saved as {output_dir_path}/{output_fig_name}.{output_fig_type}")

# %%
