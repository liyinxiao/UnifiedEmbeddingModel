import torch
from torch import nn, utils
import torch.nn.functional as F
import numpy as np

# Hyper-parameters
num_epochs = 100
learning_rate = 0.2
margin = 1.0 # margin in triplet loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(utils.data.Dataset):
    def __init__(self, query, positive_document, negative_document):
        self.query = query
        self.positive_document = positive_document
        self.negative_document = negative_document

    def __getitem__(self, index):
        return self.query[index], self.positive_document[index], self.negative_document[index]

    def __len__(self):
        return len(self.query)


class QueryEncoder(nn.Module):
    def __init__(self, input_size):
        super(QueryEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 10)
        self.fc3 = nn.Linear(10, 8)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out


class DocumentEncoder(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=(100,), activation=('relu',), solver='adam'):
        super(DocumentEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.Linear(12, 8)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return out


class Trainer():
    def __init__(self, query_encoder, document_encoder, criterion, query_optimizer, document_optimizer):
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        self.criterion = criterion
        self.query_optimizer = query_optimizer
        self.document_optimizer = document_optimizer
        self.training_loss = []

    def train(self, data_loader, epoch):
        self.query_encoder.train()
        self.document_encoder.train()
        running_loss = 0
        for _, (query_inputs, positive_document_inputs, negative_document_inputs) in enumerate(data_loader):
            # Forward pass
            anchor = query_encoder(query_inputs)
            positive = document_encoder(positive_document_inputs)
            negative = document_encoder(negative_document_inputs)
            loss = triplet_loss(anchor, positive, negative)
            # Backward and optimize
            query_optimizer.zero_grad()
            document_optimizer.zero_grad()
            loss.backward()
            query_optimizer.step()
            document_optimizer.step()
            running_loss += loss.item()
        print('Epoch [{}], Loss: {:.4f}'.format(
            epoch+1, running_loss / len(data_loader)))
        self.training_loss.append(running_loss / len(data_loader))
    
    def save_model(self, query_model_path, document_model_path):
        torch.save(self.query_encoder, query_model_path)
        torch.save(self.document_encoder, document_model_path)


def load_dummy_data(query_input_size: int, document_input_size: int) -> utils.data.DataLoader:
    # Dummy Data
    query_inputs = np.random.rand(100, query_input_size).astype(np.float32)
    positive_document_inputs = np.random.rand(
        100, document_input_size).astype(np.float32)
    negative_document_inputs = np.random.rand(
        100, document_input_size).astype(np.float32)
    data_loader = utils.data.DataLoader(Dataset(torch.from_numpy(query_inputs), torch.from_numpy(positive_document_inputs), torch.from_numpy(negative_document_inputs)),
                                        batch_size=50, shuffle=True, num_workers=2)
    return data_loader


if __name__ == '__main__':
    # number of features for query encoder and document encoder
    query_input_size = 20
    document_input_size = 15
    # Encoder initialization
    query_encoder = QueryEncoder(query_input_size).to(device)
    document_encoder = DocumentEncoder(document_input_size).to(device)
    # Optimier initialization
    query_optimizer = torch.optim.Adam(
        query_encoder.parameters(), lr=learning_rate)
    document_optimizer = torch.optim.Adam(
        document_encoder.parameters(), lr=learning_rate)
    # Triplet loss
    triplet_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=margin)
    # Trainer initialization
    trainer = Trainer(query_encoder, document_encoder,
                      triplet_loss, query_optimizer, document_optimizer)
    # load dummy data
    data_loader = load_dummy_data(query_input_size, document_input_size)
    for epoch in range(num_epochs):
        trainer.train(data_loader, epoch)
