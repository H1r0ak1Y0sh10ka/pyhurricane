#%%
##### import modules

import os
import sys
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

import logging
import matplotlib.pyplot as plt
import pyhurricane.trajectory as traj

##### Logger setting part
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

##### Read yaml file
with open('run_trajectory_conf.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(config['model_name'])

model_name       = config['model_name']    # Model name, you can set any name.
tc_name          = config['tc_name']       # Typhoon name, you can set any name.
ex_name          = config['ex_name']       # Experiment name, you can set any name.
plot_kind        = config['plot_kind']     # 'boxplot' or 'scatter'
param            = config['param']          # 'distance' or 'qv', 'rh' ###or hgt


########## Main part

##### Configuration

init_data_type        = config["init_data_type"]        # Kinds of setting parcels, 'square' or 'round' but 'square' is not implemented yet.
radius                = config["radius"]                #  Setted parcels in distance from the center of typhoon [m].
track_type            = config["track_type"]            # Kinds of trajectory, 'forward' or 'back' but 'back' is not implemented yet.
track_init_time       = config["track_init_time"]       # Trajectory start time in analysis [hour]
track_base_time_start = config["track_base_time_start"] # Trajectry start time in calculation [hour]
track_time            = config["track_time"]            # Trajectry time in calculation [hour]

input_model_data_dir        = config["input_model_data_dir"]        #Input data and dirctory
input_model_name_suffix     = config["input_model_name_suffix"]

case_names            = config["case_names"]
legends               = config["legends"]

#### Input data and dirctory
tack_time_delta       = 60

#### Output data and dirctory

#output_dir_path  = "./"+str(plot_kind[0].upper())+radix+param+"_"+ex_kind
#if ex_kind == "individual":
#    ex_kind = case_names[0]

#if "timeseries" in prefix or "timeseries" in radix or "timeseries" in suffix or "timeseries" in plot_kind:
#    save_prefix      = model_name+'-'+tc_name+"-"+ex_name+"-"+str(plot_kind[0].upper())+radix+param+"_"+ex_kind

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
                            theta_interval=90,
                            track_base_Z=0,
                            track_variable_size=8
                            )

#logging.info(f"The figure is saved as {output_dir_path}/{output_fig_name}.{output_fig_type}")

# %%
