
import torch

learning_rate = 1e-4
batch_size =1
epochs = 500
print_every = 32

cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor