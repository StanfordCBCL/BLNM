#!/usr/bin/python3
from BLNM import BLNM
import torch 
import numpy as np 
import pickle, time, os
import matplotlib.pyplot as plt
import time, pickle 
import pandas as pd
import pyvista as pv 
from train_AT import *

np.random.seed(1)
torch.manual_seed(1)

def adimensionalize_data(z_scores_count, geometry, space_min, space_max, AT_min, AT_max):
    if geometry.startswith('ToF'):
        AT_file = f'{os.path.dirname(os.getcwd())}/data/ToF/dataset_AT_{geometry}.pkl'
        space_file = f'{os.path.dirname(os.getcwd())}/data/ToF/dataset_space_AT_{geometry}.pkl'
    elif geometry.startswith('ct'):
        AT_file = f'{os.path.dirname(os.getcwd())}/data/ct/dataset_AT_{geometry}.pkl'
        space_file = f'{os.path.dirname(os.getcwd())}/data/ct/dataset_space_AT_{geometry}.pkl'
        
    dataset_AT = load_data(AT_file)
    AT_adim = BLNM.adimensionalize(dataset_AT['AT'], AT_min, AT_max)  
    
    dataset_space = load_data(space_file)
    num_points = dataset_space['space'].shape[1]
    all_indices = np.arange(num_points)
    space_adim = BLNM.adimensionalize(dataset_space['space'], space_min, space_max) 
   
    z_scores_data = z_scores.drop(columns=['dir_name'])
    z_scores_min = z_scores_data.min(axis=0).to_numpy()
    z_scores_max = z_scores_data.max(axis=0).to_numpy()
    z_scores_normalized = (z_scores_data - z_scores_min) / (z_scores_max - z_scores_min)
    z_scores_normalized.insert(0, 'dir_name', z_scores['dir_name'])

    z_scores_row = z_scores_normalized[z_scores_normalized['dir_name'] == geometry]
    z_scores_row = z_scores_row.iloc[:, 1:z_scores_count +1].to_numpy(dtype=np.float32)[0]
    all_z_scores = np.repeat(z_scores_row, num_points).reshape(len(z_scores_row), -1)
    
    return all_indices, space_adim, AT_adim, all_z_scores 

def measure_inference_time(model, test_geometries):
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = torch.nn.MSELoss()
    space_min, space_max, AT_min, AT_max = preprocess(num_geometries, geometries)

    total_time = 0.0

    total_testing_loss = 0
    with torch.no_grad():  # No need for gradients in inference
        for i in range(len(test_geometries)):  # Run inference multiple times
            geometry = test_geometries[i] # Pick a random test geometry
            print(geometry)
            _, space_adim, AT_adim, all_z_scores = adimensionalize_data(z_score_count, geometry, space_min, space_max, AT_min, AT_max)
            
            # Convert to tensors
            space_tensor = torch.tensor(space_adim.T, dtype=torch.float32).to(device).unsqueeze(0)
            z_scores_tensor = torch.tensor(all_z_scores.T, dtype=torch.float32).to(device).unsqueeze(0)

            AT_tensor = torch.tensor(AT_adim.T, dtype=torch.float32).to(device).unsqueeze(0)   
        
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                predictions = model(z_scores_tensor, space_tensor)[:, :, 0:1]
                testing_loss = criterion(predictions, AT_tensor).item()
                outputs = BLNM.dimensionalize(predictions, AT_min, AT_max).detach().cpu().numpy().flatten()

                print(f'Loss of {geometry} is {testing_loss:.4f}' )

                total_testing_loss += testing_loss
            print(total_testing_loss)

            end_time = time.time()

            total_time += (end_time - start_time)

    avg_time_sec = (total_time / len(test_geometries))  
    print(f"Average inference time over {len(test_geometries)} runs: {avg_time_sec:.6f} sec")
    
    avg_testing_loss = (total_testing_loss / len(test_geometries))  
    print(f"Average loss: {avg_testing_loss}")

if __name__ == '__main__':    
    # num_points_test = 6442 
    num_points_test = 6442 
    num_points_train = 1485

    num_epochs = 1000
    out_freq = 10


    real_geom =['ct_1010', 'ct_1012', 'ct_1015', 'ct_1028', 'ct_1037', 'ct_1046', 'ct_1074', 'ct_1124', 'ct_1129', 'ct_1143', 'ct_1146', 'ct_1147', 'ct_1178']
    synthetic_geom = [f"ToF_{i}" for i in range(52)]    
 

    train_geometries = synthetic_geom
    geometries =  train_geometries + real_geom 

    z_score_dir = f'{os.path.dirname(os.getcwd())}/data/z_scores'
    z_scores_file = f'{z_score_dir}/all_z_scores.csv' 
    z_scores = pd.read_csv(z_scores_file)

    num_geometries = len(geometries) 
    indices = list(range(0, num_geometries))
     
    num_train = len(train_geometries)

    train_indices = indices[0:num_train]
    test_indices = indices[num_train:num_geometries]
    
    config = {
        "neurons": 42,     
        "layers": 16,
        "lr":  0.0005,
        "num_states": 5,
        "disentanglement_level": 2
    }

    neurons_per_layer = [config["neurons"]]* config["layers"]
    z_score_count = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "trained_best_model_overall.pth"
    if not os.path.exists(model_path):
        print("Best overall model not found")
    
    model = BLNM(neurons_per_layer, z_score_count, 3, config["num_states"], config["disentanglement_level"]).to(device)
    measure_inference_time(model, real_geom)
        