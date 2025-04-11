import torch.nn as nn
import torch.optim as optim
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate, lr):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  
        self.network = nn.Sequential(*layers)
        
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.network(x)

    def fit(self, train_loader, val_loader, epochs=10):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad() 
                y_pred = self(x_batch)  
                loss = self.criterion(y_pred, y_batch)  
                loss.backward()  
                optimizer.step()  
                
                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss / len(train_loader)}")
            
            self.evaluate(val_loader)

    def evaluate(self, val_loader):
        self.eval()  
        val_loss = 0.0
        with torch.no_grad():  
            for x_batch, y_batch in val_loader:
                y_pred = self(x_batch)
                loss = self.criterion(y_pred, y_batch)
                val_loss += loss.item()

        # print(f"Validation Loss: {val_loss / len(val_loader)}")
        return val_loss / len(val_loader)


