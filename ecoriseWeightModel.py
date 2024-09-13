
# [0] IMPORTS
import snntorch as snn; from snntorch import spikeplot as splt; from snntorch import spikegen
import torch; import torch.nn as nn; from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt; import numpy as np
import snntorch.spikeplot as splt; import matplotlib.pyplot as plt 
import torchvision.transforms as transforms; import torch.optim as optim
import torch.nn.functional as F

# [1] DECISIONS
image_dim = 5 #choose dimensions of input images
time_steps = 20 # choose number of timesteps
batch_size = 5 # choose number of train/test data
num_hidden = 0 # choose the number of hidden network layers
epoch_size = 2 # choose number of training iterations

# [2] DATA ENCODING
data_path = '/tmp/data/nyceExampleData'
dtype = torch.float; device = torch.device("cpu") 
transform = transforms.Compose([
            transforms.Resize((image_dim, image_dim)), 
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=True)
images, labels = next(iter(train_loader))
num_inputs = image_dim*image_dim; num_outputs = 10

# [3] SIMULATION 
def leaky_mod(images):
    spikes_all = torch.zeros(batch_size, time_steps, image_dim*image_dim)
    for i in range(1, batch_size):
        image = images[i]; image = image.squeeze(0)
        # normalize the image
        image = (image - image.min())/(image.max()-image.min())
        raw_vector = torch.cat([image.repeat(time_steps, 1, 1)], dim=0)
        rate_coded_vector = torch.poisson(raw_vector)

        # raster plot
        sample_shape = list(rate_coded_vector.shape)
        first_value = sample_shape[0]
        raster = rate_coded_vector.reshape((first_value, -1)) #[time steps, total # of pixels (lxw)]
        spike_sample_flat = raster.sum(0)
        ex_neuron = torch.argmax(spike_sample_flat).item() # choose sample neuron

        # normalize data
        min_rate = raster.min(); max_rate = raster.max()
        cur_in = (raster - min_rate)/(max_rate - min_rate)
        min_current = 0; max_current = 1
        cur_in = min_current + cur_in * (max_current - min_current)
        cur_in[cur_in > 0] = 1 #LOOK AT THIS LATER
        cur_in[cur_in > 0] = 1 #LOOK AT THIS LATER

        tau = 0.6; V_threshold = 0.4; V_reset = 0; dt = 0.1
        t_sim = np.arange(0, len(cur_in)*dt, dt)
        num_steps = cur_in.size(0)
        V = torch.zeros((num_steps, cur_in.size(1))) #timestamps x pixels(100), 
        cutoffs = torch.zeros(num_steps, cur_in.size(1)) #timestamps x pixels(100), 
        spike_ex = torch.zeros((num_steps, cur_in.size(1)))

        for n in range(1,cur_in.shape[1]):
            cutoff = 1
            for t in range(1, num_steps):
                dV = (-(V[t - 1,n] - V_reset) + cur_in[t, n]) / tau*dt 
                cutoff = cutoff + 1
                if cutoff == 10:
                    V[t,n] = V_reset
                    cutoff = 0
                    cutoffs[t,n] = V_threshold
        
                elif V[t-1,n] < V_threshold:
                    V[t,n] = V[t-1,n] + dV

                    if V[t,n] >= V_threshold:
                        spike_ex[t,n] = 1
                        V[t,n] = V_threshold
        
                else:
                    V[t,n] = V_threshold
        spikes_all[i] = spike_ex
        return image, cur_in, ex_neuron, raster, spike_ex, V, t_sim, cutoffs, spikes_all

# [4] SIMULATION VERIFICATION FIGURES (will show the final image)
[image, cur_in, ex_neuron, raster, spike_ex, V, t_sim, cutoffs, spikes_all] = leaky_mod(images)
mosaic = [
    ["A", "B", "C", "D"],
    ["E", "E", "E", "E"],
    ["F", "F", "F", "F"],
    ["G", "G", "G", "G"],
    ["H", "H", "H", "H"]
]
fig, ax = plt.subplot_mosaic(mosaic, figsize=(10, 10))
ax['A'].imshow(image)
#ax['A'].imshow(image.permute([1, 2, 0]))
ax['A'].set_xlabel('Original image', fontsize=18)
ax['B'].imshow(spike_ex[0].reshape(image_dim,image_dim))
ax['B'].set_xlabel('t = 0', fontsize=18)
ax['C'].imshow(spike_ex[:time_steps//3].mean(0).reshape(image_dim,image_dim))
ax['C'].set_xlabel(f't = {time_steps//3}', fontsize=18)
ax['D'].imshow(spike_ex.mean(0).reshape(image_dim,image_dim))
ax['D'].set_xlabel(f't = {time_steps}', fontsize=18)
splt.raster(raster, ax['E'], s=1.5, c="black")
ax['E'].set_xlabel('Time step'); ax['E'].set_ylabel('Neuron Number')
ax['F'].plot(t_sim, cur_in[:,ex_neuron].unsqueeze(1), label='Input current')
ax['F'].set_xlabel('Time step'); ax['F'].set_ylabel('Input Current Ex Neuron')
ax['G'].plot(t_sim, V[:, ex_neuron], label='Membrane Potential', color = 'green')
ax['G'].set_xlabel('Time step'); ax['G'].set_ylabel('Membrane Potential Ex Neuron')
ax['G'].plot(t_sim, cutoffs[:,ex_neuron], color='gray', linestyle='--', alpha=0.5) 
ax['H'].plot(t_sim, spike_ex[:,ex_neuron], label='Spikes', color = 'red')
ax['H'].set_xlabel('Time step'); ax['H'].set_ylabel('Spikes Ex Neuron');
plt.tight_layout(); plt.show()

# [5] DEFINE NETWORK
class LeakyNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = torch.randn(output_size, input_size, device=device)  # Weight matrix on GPU/CPU
        self.b = torch.zeros(output_size, 1, device=device)  # Bias vector on GPU/CPU
        self.lr = 0.01  # Learning rate
        self.loss_fn = nn.CrossEntropyLoss()  # Loss function
    
    def forward(self, inputs):
        _, _, _, _, _, _, _, _, spikes_all = leaky_mod(inputs)
        avg_spikes = spikes_all.mean(dim=1)  # Average spikes across time steps
        outputs = torch.matmul(avg_spikes, self.W.T) + self.b.T  # Forward pass
        return outputs
    
    def train(self, train_loader, epochs):
        optimizer = optim.SGD([self.W, self.b], lr=self.lr)  # SGD optimizer
        loss_vec = []  # Store losses
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()  # Zero gradients
                predictions = self.forward(images)
                loss = self.loss_fn(predictions, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Optimizer step
                
                epoch_loss += loss.item() * images.size(0)  # Accumulate loss
            
            epoch_loss /= len(train_loader.dataset)  # Average loss per sample
            loss_vec.append(epoch_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        
        return loss_vec
    
    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = self.forward(images)
                _, predicted = torch.max(predictions, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total * 100
        print(f"Accuracy on test set: {accuracy:.2f}%")
        return accuracy


# [6] NETWORK VERIFICATION FIGURES
net = LeakyNetwork(num_inputs, num_outputs)

loss_vec = net.train(train_loader, epoch_size)
accuracy_vec = net.test(test_loader)

# Plotting
epoch_vec = np.arange(1, epoch_size + 1)
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(epoch_vec, accuracy_vec, marker='o', linestyle='-', color='b')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
plt.subplot(2, 1, 2)
plt.plot(epoch_vec, loss_vec, marker='o', linestyle='-', color='r')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.tight_layout(); plt.show()
