from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from networks.Autoencoder import *
from networks.Dataset import DepthDataset


hparams = {
    "batch_size": 20,
    "learning_rate": 1e-3,
    "num_epochs": 40,
    "validation_split": 0.2,
    "data_path": '../python_images_new',
}

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(hparams).to(device).float()
    print(model)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])

    # Dataloader
    #https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
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

    diz_loss = {'train_loss': [], 'val_loss': []}
    for epoch in range(hparams["num_epochs"]):
        train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
        visualize = False
        #if epoch+1==hparams["num_epochs"]: 
        if (epoch+1)%5 == 0: 
            visualize = True
        val_loss = test_epoch(model, device, val_loader, criterion, visualize)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, hparams["num_epochs"], train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)

    #plot_ae_outputs(model, device, dataset, val_indices, n=3)




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


def plot_outputs(conc_out, conc_gt, conc_inp, n=5): 
    plt.figure(figsize=(14,5)) 
    print(conc_out.size())
    output = conc_out.squeeze().numpy()
    gt_img = conc_gt.squeeze().numpy()
    input = conc_inp.squeeze().numpy()
    print(conc_out.shape)
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
            ax.set_title('Reconstructed images')
    plt.show()







