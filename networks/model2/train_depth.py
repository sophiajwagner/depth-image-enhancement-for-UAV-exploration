from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import os
import seaborn as sns

from networks.DepthAutoencoder import *
from networks.DepthDataset import DepthDataset


hparams = {
    "batch_size": 25,
    "learning_rate": 1e-3,
    "num_epochs": 30,
    "validation_split": 0.2,
    "input_data_path": '../../final_python_images/low_light/low_depth_tif',
    "gt_data_path": '../../final_python_images/high_light/high_depth_tif',
    "out_path": "out/train_depth",
}

# Train a model that takes a low-light depth image as input and output the corresponding high-light depth image

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(hparams).to(device).float()
    print(model)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])

    # Dataloader
    dataset = DepthDataset(hparams)
    shuffle_dataset = True
    random_seed = 42
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(hparams["validation_split"] * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams["batch_size"],
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams["batch_size"],
                                                    sampler=val_sampler)
    if not os.path.exists(hparams['out_path']):
        os.makedirs(hparams['out_path'])
    diz_loss = {'train_loss': [], 'val_loss': []}
    for epoch in range(hparams["num_epochs"]):
        train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
        visualize = False
        if (epoch+1)%5 == 0: 
            visualize = True
        val_loss = test_epoch(model, device, val_loader, criterion, visualize)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, hparams["num_epochs"], train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)

        if visualize: 
            plot_loss(epoch+1, diz_loss['train_loss'], diz_loss['val_loss'])
    
    torch.save(model, os.path.join(hparams['out_path'],"weights_pt.pt"))
    torch.save(model.state_dict(), os.path.join(hparams['out_path'],"weights"))
    
        
        
def test(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(hparams).to(device).float()
    model.load_state_dict(torch.load(os.path.join(hparams['out_path'],"weights")))
    
    # Loss function
    criterion = nn.MSELoss() 
    
    # Dataloader
    dataset = DepthDataset(hparams)
    shuffle_dataset = True
    random_seed = 42
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(hparams["validation_split"] * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams["batch_size"],
                                                    sampler=val_sampler) 
    
    val_loss = test_epoch(model, device, val_loader, criterion, True)
    

def test_image(idx): 
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(hparams).to(device).float()
    model.load_state_dict(torch.load(os.path.join(hparams['out_path'],"weights")))
    model.eval()
    # Load dataset
    dataset = DepthDataset(hparams) 
    input_np = dataset[idx-1]['input']
    input = torch.from_numpy(input_np).to(device).unsqueeze(0).unsqueeze(0).float()
    gt_np = dataset[idx-1]['gt']
    # Get model output
    output = model(input).squeeze().cpu().detach().numpy()
    
    # plot input, ground truth and output
    plt.imshow(input_np, cmap='gist_gray')
    plt.title("Low-light depth input image", fontsize=15)
    plt.axis('off')
    plt.show()
    
    plt.imshow(gt_np, cmap='gist_gray')
    plt.title("Ground truth depth image", fontsize=15)
    plt.axis('off')
    plt.show()
    
    plt.imshow(output, cmap='gist_gray')
    plt.title("Output of the model", fontsize=15)
    plt.axis('off')
    plt.show()
    return output
    
    
def error_heatmaps(idx):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(hparams).to(device).float()
    model.load_state_dict(torch.load(os.path.join(hparams['out_path'],"weights")))
    model.eval()
    # Load data
    dataset = DepthDataset(hparams) 
    input_np = dataset[idx-1]['input']
    gt_np = dataset[idx-1]['gt']
    input = torch.from_numpy(input_np).to(device).unsqueeze(0).unsqueeze(0).float()
    gt = torch.from_numpy(gt_np).to(device).unsqueeze(0).unsqueeze(0).float()
    # Get output
    output_np = model(input).squeeze().cpu().detach().numpy()
    # scale back
    input_np = input_np * 19.5 + 0.5
    gt = gt * 19.5 + 0.5
    output_np = output_np * 19.5 + 0.5
    
    # Plot error heatmaps
    sns.heatmap(abs(input_np-output_np), square=True, cbar=True, cmap='rocket_r', xticklabels=False, yticklabels=False, vmin=0, vmax=5)
    plt.title('Absolute error between input\n and predicted depth image')
    plt.savefig(os.path.join(hparams['out_path'],'error_input_output_'+str(idx)+'.png'))
    plt.show()
    
    sns.heatmap(abs(gt_np-output_np), square=True, cbar=True, cmap='rocket_r', xticklabels=False, yticklabels=False, vmin=0, vmax=5)
    plt.title('Absolute error between ground truth\n and predicted depth image')
    plt.savefig(os.path.join(hparams['out_path'],'error_gt_output_'+str(idx)+'.png'))
    plt.show()
    
    sns.heatmap(abs(input_np-gt_np), square=True, cbar=True, cmap='rocket_r', xticklabels=False, yticklabels=False, vmin=0, vmax=5)
    plt.title('Absolute error between input\n and ground truth depth image')
    plt.savefig(os.path.join(hparams['out_path'],'error_input_gt_'+str(idx)+'.png'))
    plt.show()



### Training function
#https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
def train_epoch(model, device, dataloader, loss_fn, optimizer):
    # Set train mode for the model
    model.train()
    train_loss = []
    # Iterate the dataloader
    for i, data in enumerate(dataloader):
        input = data['input'].to(device).unsqueeze(1).float()
        gt = data['gt'].to(device).unsqueeze(1).float()
        output = model(input)
        # Evaluate loss
        loss = loss_fn(output, gt)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)


### Testing function
#https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
def test_epoch(model, device, dataloader, loss_fn, visualize=False):
    # Set evaluation mode for the model
    model.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_gt = []
        conc_inp = []
        for i, data in enumerate(dataloader):
            input = data['input'].to(device).unsqueeze(1).float()
            gt = data['gt'].to(device).unsqueeze(1).float()
            output = model(input)
            # Append the network output and the ground truth to the lists
            conc_out.append(output.cpu())
            conc_gt.append(gt.cpu())
            conc_inp.append(input.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_gt = torch.cat(conc_gt)
        conc_inp = torch.cat(conc_inp)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_gt)
        
        if visualize: 
            plot_outputs(conc_out, conc_gt, conc_inp)
    return val_loss.data


def plot_outputs(conc_out, conc_gt, conc_inp, n=3): 
    plt.figure(figsize=(8,5)) 
    output = conc_out.squeeze().numpy()
    gt_img = conc_gt.squeeze().numpy()
    input = conc_inp.squeeze().numpy()
    for i in range(n): 
        ax = plt.subplot(3, n, i+1)
        plt.imshow(input[i], cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Input images')

        ax = plt.subplot(3, n, i+1+n)
        plt.imshow(gt_img[i], cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Ground truth images')

        ax = plt.subplot(3, n, i+1+2*n)
        plt.imshow(output[i], cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Output images')
    plt.savefig(os.path.join(hparams['out_path'],'preds.png'))
    plt.show()
    
    
def plot_loss(num_epochs, train_loss, val_loss):
    plt.plot(range(1,num_epochs+1), train_loss, '-b', label='train loss')
    plt.plot(range(1,num_epochs+1), val_loss, '-r', label='validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Train and validation loss')
    plt.savefig(os.path.join(hparams['out_path'],'loss.png'))
    plt.show()







