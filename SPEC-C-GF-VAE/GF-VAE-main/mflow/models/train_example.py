import torch.optim as optim
import torch
import torch.nn as nn
import enc_spec as vae
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time 
from sklearn.model_selection import train_test_split


def train(model, data_loader, epochs, device):

    ### To save the output
    reconstructions = []
    latents = []
    losses = []

    model.train() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        for batch in data_loader:
            batch = batch.to(device)  
            optimizer.zero_grad() 

            # Forward pass
            reconstructed_batch, _, latent_dist = model.forward_(batch)

            # Compute loss
            loss = model.loss(reconstructed_batch, batch, latent_dist)
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if epoch == epochs - 1: 
                reconstructions.append(reconstructed_batch.detach().cpu())
                latents.append(latent_dist)
        scheduler.step()

        average_loss = total_loss / len(data_loader)
        losses.append(average_loss)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}, Total Loss: {total_loss}, Average Loss: {average_loss}")
        print(f"Time for Epoch {epoch+1}: {epoch_duration:.2f} seconds")

    reconstructions = torch.cat(reconstructions)
    latents = [torch.cat([ld[i] for ld in latents]) for i in range(2)] 

    pd.DataFrame(reconstructions.numpy()).to_csv('../results/spectra_reconstructions.csv', index=False)

    pd.DataFrame(latents[0].detach().numpy()).to_csv('../results/spectra_latent_means.csv', index=False) 
    pd.DataFrame(latents[1].detach().numpy()).to_csv('../results/spectra_latent_logvars.csv', index=False)   

    total_duration = time.time() - start_time  
    print(f"Total Training Time: {total_duration:.2f} seconds")

    loss_df = pd.DataFrame(losses, columns=['Epoch Loss'])
    loss_df.to_csv('../results/training_losses.csv', index=False)


file_path = '../data/transformed_spectra.csv'
transformed_data = pd.read_csv(file_path)

train_data, test_data = train_test_split(transformed_data, test_size=0.20, random_state=42)  # random_state for reproducibility


class SpectrumDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx].values, dtype=torch.float)

#spectrum_dataset = dt.Spectra(transformed_data)
#print(spectrum_dataset)

train_dataset = SpectrumDataset(train_data)
test_dataset = SpectrumDataset(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4)


#spectrum_dataset_second = SpectrumDataset(transformed_data)
#spectrum_dataloader = DataLoader(spectrum_dataset_second, batch_size=10, shuffle=True, num_workers=4)
#print(len(spectrum_dataloader))

def test_and_save_reconstructions(model, data_loader, device, file_name):
    model.eval()  # Set the model to evaluation mode
    reconstructions = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            reconstructed_batch, _, _ = model.forward_(batch)
            reconstructions.append(reconstructed_batch.detach().cpu())

    reconstructions = torch.cat(reconstructions)
    pd.DataFrame(reconstructions.numpy()).to_csv(file_name, index=False)
    print(f'Reconstructions saved to {file_name}')


# Parameters
epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    #LAYER CONFIGURATIONS
    'layer_config': [[200, 128, 64, 32], [32, 64, 128, 200]], 
    'transform': None,
    'beta': 1.0, 
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spec_vae = vae.SpecVEA(config, device=device)

# Move the model to the device
spec_vae.to(device)
# Call the train function
train(spec_vae, train_dataloader, epochs, device)

test_and_save_reconstructions(spec_vae, train_dataloader, device, '../results/train_reconstructions.csv')
test_and_save_reconstructions(spec_vae, test_dataloader, device, '../results/test_reconstructions.csv')

def dataset_to_dataframe(dataset):
    data_list = []
    for i in range(len(dataset)):
        data_point = dataset[i].numpy()  
        data_list.append(data_point)
    
    # Convert list to DataFrame
    return pd.DataFrame(data_list)

# Convert datasets to DataFrames
train_df = dataset_to_dataframe(train_dataset)
test_df = dataset_to_dataframe(test_dataset)

train_df.to_csv('../results/train_data.csv', index=False)
test_df.to_csv('../results/test_data.csv', index=False)


