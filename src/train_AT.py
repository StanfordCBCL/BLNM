#!/usr/bin/python3
from BLNM import BLNM
import torch 
import numpy as np 
import random, pickle, time, os
import time, pickle 
import pandas as pd

np.random.seed(1)
torch.manual_seed(1)

def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

## find space min and max and activation min and max for adimensionalizing
def preprocess(num_samples, geometries):
    num_coordinates = 3 
    num_activation = 1 

    space_min = np.full((num_coordinates, 1), np.inf)   ## 3 space coordinates
    space_max = np.full((num_coordinates, 1), -np.inf)
    AT_min = np.full((num_activation, 1), np.inf)      ## 1 activation time
    AT_max = np.full((num_activation, 1), -np.inf)

    for idx_p in range(num_samples):
        geometry = geometries[idx_p]
        if geometry.startswith('ToF'):
            AT_file = f'{os.path.dirname(os.getcwd())}/data/ToF/dataset_AT_{geometry}.pkl'
            space_file = f'{os.path.dirname(os.getcwd())}/data/ToF/dataset_space_AT_{geometry}.pkl'
        elif geometry.startswith('ct'):
            AT_file = f'{os.path.dirname(os.getcwd())}/data/ct/dataset_AT_{geometry}.pkl'
            space_file = f'{os.path.dirname(os.getcwd())}/data/ct/dataset_space_AT_{geometry}.pkl'
        
        dataset_AT = load_data(AT_file)
        local_AT_min = np.min(dataset_AT['AT'], axis=1)
        local_AT_max = np.max(dataset_AT['AT'], axis=1)
        
        dataset_space = load_data(space_file)
        local_space_min = np.min(dataset_space['space'], axis=1)
        local_space_max = np.max(dataset_space['space'], axis=1)

        for coord in range(num_coordinates):
            if local_space_min[coord] < space_min[coord]:
                space_min[coord] = local_space_min[coord]
            if local_space_max[coord] > space_max[coord]:
                space_max[coord] = local_space_max[coord]

        if local_AT_min[0] < AT_min[0]:
            AT_min[0] = local_AT_min[0]
        if local_AT_max[0] > AT_max[0]:
            AT_max[0] = local_AT_max[0]
    return space_min, space_max, AT_min, AT_max

## adimensionalize data 
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

## train model
def train(config):
    start_time = time.time()
    train_loss_list = []
    test_loss_list = []
    epochs_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.MSELoss()
    
    neurons_per_layer = [config["neurons"]]* config["layers"]
    space_coordinates = 3

    model = BLNM(neurons_per_layer, config["z_score_count"], space_coordinates, config["num_states"], config["disentanglement_level"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    space_min, space_max, AT_min, AT_max = preprocess(num_geometries, geometries)

    best_test_loss = float('inf') 
    best_train_loss = float('inf') 
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        random.shuffle(train_indices)

        for ind in train_indices:
            geometry = geometries[ind]
            all_indices, space_adim, AT_adim, all_z_scores = adimensionalize_data(config["z_score_count"], geometry, space_min, space_max, AT_min, AT_max)
            dofs_train = np.random.choice(all_indices, num_points_train, replace=False)
            space_train = torch.tensor(space_adim[:, dofs_train].T, dtype=torch.float32).to(device).unsqueeze(0)
            AT_train = torch.tensor(AT_adim[:, dofs_train].T, dtype=torch.float32).to(device).unsqueeze(0)   
    
            z_scores_train = torch.tensor(all_z_scores[:, :num_points_train].T, dtype=torch.float32).to(device).unsqueeze(0)
           # Forward pass
            optimizer.zero_grad()
            predictions = model(z_scores_train, space_train)[:, :, 0:1]
            loss = criterion(predictions, AT_train)

            # Backward pass 
            loss.backward()
            optimizer.step()  

            train_loss += loss.item() 
        
        avg_train_loss = train_loss/len(train_indices)
        
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss

        if (epoch + 1)%out_freq == 0: 
            epochs_list.append(epoch)
            train_loss_list.append(avg_train_loss)

            test_loss = 0
            for ind in test_indices:
                geometry = geometries[ind]
                all_indices, space_adim, AT_adim, all_z_scores = adimensionalize_data(config["z_score_count"],geometry, space_min, space_max, AT_min, AT_max)
                dofs_test = np.random.choice(all_indices, num_points_test, replace=False)

                space_test = torch.tensor(space_adim[:, dofs_test].T, dtype=torch.float32).to(device).unsqueeze(0)
                AT_test = torch.tensor(AT_adim[:, dofs_test].T, dtype=torch.float32).to(device).unsqueeze(0)
                z_scores_test = torch.tensor(all_z_scores[:, :num_points_test].T, dtype=torch.float32).to(device).unsqueeze(0)

                model.eval()

                with torch.no_grad():
                    predictions = model(z_scores_test, space_test)[:, :, 0:1]
                    loss = criterion(predictions, AT_test)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss/len(test_indices)
            if avg_test_loss < best_test_loss: 
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), "best_model_overall.pth")


            test_loss_list.append(avg_test_loss)
            
            print(f'Epoch {epoch}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Testing Loss: {avg_test_loss:.4f}')

    print(f'Training completed in {time.time() - start_time:.2f} seconds')
    print(f'Best Training Loss: {best_train_loss:.4f}, Best Testing Loss: {best_test_loss:.4f}')

    training_dir = f'{os.path.dirname(os.getcwd())}/training'
    os.makedirs(training_dir, exist_ok=True)
    AT_predictions_dir = f'{training_dir}/AT_predictions/'
    os.makedirs(AT_predictions_dir, exist_ok=True)
    AT_results = f'{AT_predictions_dir}/AT_results.txt'

    save_txt(epochs_list, train_loss_list, test_loss_list, best_train_loss, best_test_loss, config, AT_results)

## save training results
def save_txt(epoch_list, train_loss_list, test_loss_list, best_train_loss, best_test_loss, config, filename):
    with open(filename, 'a') as f:
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write(f"Best Training Loss: {best_train_loss}\n")
        f.write(f"Best Testing Loss: {best_test_loss}\n")
        f.write("\n")

        f.write("Epoch Losses: \n")
        for epoch, train_loss, test_loss in zip(epoch_list, train_loss_list, test_loss_list):
            f.write(f"epoch: {epoch}, train: {train_loss}, test: {test_loss}\n")
            
        print(f"Losses and configuration saved to {filename}")

if __name__ == '__main__':   
    num_points_test = 6442 
    num_points_train = 1485

    num_epochs = 1000
    out_freq = 10

    # original cohort
    real_geom =['ct_1010', 'ct_1012', 'ct_1015', 'ct_1028', 'ct_1037', 'ct_1046', 'ct_1074', 'ct_1124', 'ct_1129', 'ct_1143', 'ct_1146', 'ct_1147', 'ct_1178']
    
    # synthetic cohort
    synthetic_geom = [f"ToF_{i}" for i in range(52)]    

    train_geometries = synthetic_geom
    geometries =  train_geometries + real_geom 


    z_score_dir = f'{os.path.dirname(os.getcwd())}/data/z_scores'
    z_scores_file = f'{z_score_dir}/all_z_scores.csv' 
    z_scores = pd.read_csv(z_scores_file)

    num_geometries = len(geometries) 
    num_train = len(train_geometries)

    indices = list(range(0, num_geometries))
    train_indices = indices[0:num_train]
    test_indices = indices[num_train:num_geometries]
    
    config = { 
        "z_score_count": 12,
        "neurons": 42,     
        "layers": 16,
        "lr":  0.0005,
        "num_states": 5,
        "disentanglement_level": 2
    }

    train(config)
   