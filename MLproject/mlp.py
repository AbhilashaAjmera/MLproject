import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

# Step 1: Prepare the Dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize images to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize images
])

train_dataset = datasets.USPS(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.USPS(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3: Define Loss Function, Optimizer, and TensorBoard writer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
writer = SummaryWriter('./logs/cnn_usps')

# Step 4: Train the Model
def train_model(model, train_loader, criterion, optimizer, writer):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'Batch {i+1}/{len(train_loader)} Loss: {running_loss/100:.4f}')
            writer.add_scalar('training_loss', running_loss/100, len(train_loader) * epoch + i)
            running_loss = 0.0

# Step 5: Evaluate the Model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, conf_matrix
    probs = torch.softmax(outputs, dim=1)
    precision, recall, _ = precision_recall_curve(labels.cpu(), probs[:, 1].cpu())
    pr_auc = auc(recall, precision)

   

# Step 6: Main Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, writer)

    # Evaluate the model
    accuracy, precision, recall, conf_matrix = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)

    # Write metrics to TensorBoard
    writer.add_scalar('test_accuracy', accuracy, epoch)
    writer.add_scalar('test_precision', precision, epoch)
    writer.add_scalar('test_recall', recall, epoch)

# Step 7: Close TensorBoard writer
writer.close()
# Step 2: Load TensorBoard Extension
%load_ext tensorboard

# Step 3: Specify Log Directory
log_dir = './logs'

# Step 4: Start TensorBoard
%tensorboard --logdir $log_dir
