import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epochs = 50
embedding_dim = 300
hidden_dim = 300
data_dir = './data'
train_name = 'train'
test_name = 'test'
save_name = '-models.pth'