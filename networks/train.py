from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

from networks.Autoencoder import *

hparams = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 30
}

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(hparams).to(device)
    print(model)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])

    # TODO
    #train_dataset = ...
    #val_dataset = ...
    #test_dataset = ...

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=hparams["batch_size"], shuffle=False, num_workers=4
    )

    diz_loss = {'train_loss': [], 'val_loss': []}
    for epoch in range(hparams["num_epochs"]):
        train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
        val_loss = test_epoch(model, device, val_loader, criterion)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, hparams["num_epochs"], train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        plot_ae_outputs(model, n=3)




### Training function
#https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
def train_epoch(model, device, dataloader, loss_fn, optimizer):
    # Set train mode for the model
    model.train()
    train_loss = []
    # Iterate the dataloader
    for i, data in enumerate(dataloader):
        input = data['input'].to(device)
        gt = data['gt'].to(device)
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
def test_epoch(model, device, dataloader, loss_fn):
    # Set evaluation mode for the model
    model.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_gt = []
        for i, data in enumerate(dataloader):
            input = data['input'].to(device)
            gt = data['gt'].to(device)
            output = model(input)
            # Append the network output and the ground truth to the lists
            conc_out.append(output.cpu())
            conc_gt.append(gt.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_gt = torch.cat(conc_gt)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_gt)
    return val_loss.data


def plot_ae_outputs(model, val_dataset, n=3):
    plt.figure(figsize=(16,4.5))
    targets = val_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):

        #TODO
        input = val_dataset['input'][t_idx[i]][0].unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            rec_img  = model(input)
        gt_img = val_dataset['ground truth'][t_idx[i]][0].unsqueeze(0).to(device)

        ax = plt.subplot(3, n, i+1)
        plt.imshow(input.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Input images')

        ax = plt.subplot(3, n, i+1+n)
        plt.imshow(gt_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Ground truth images')

        ax = plt.subplot(3, n, i+1+2*n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
    plt.show()


