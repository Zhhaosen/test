import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# 假设你已经加载了数据到DataFrame中
df = pd.read_excel('total.xlsx')
print(df.columns)
# 选择特征列和标签列
features = ['mid-term',  'faculty', 'department']
label = 'Label'

# 数据清洗，例如删除缺失值
df = df.dropna()

# 划分训练集和测试集
X = df[features]
y = df[label].values
total_x = X
total_y = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征编码和缩放
numerical_cols = ['mid-term']
categorical_cols = ['faculty', 'department']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)
preprocessor.fit(df[features])
X_train_encoded = preprocessor.transform(X_train)
X_test_encoded = preprocessor.transform(X_test)
total_x = preprocessor.transform(total_x)

print(X_train_encoded.shape)
print(X_test_encoded.shape)

# 将编码后的数据转换为PyTorch张量
# 假设 X_train_encoded 是一个稀疏矩阵
# 首先，将其转换为密集矩阵
X_train_encoded_dense = X_train_encoded.toarray()
total_x = total_x.toarray()
# 然后，使用密集矩阵创建 PyTorch 张量
X_train_tensor = torch.tensor(X_train_encoded_dense, dtype=torch.float32)
total_x_tensor = torch.tensor(total_x,dtype=torch.float32)
# 重复相同的步骤来处理测试集数据
X_test_encoded_dense = X_test_encoded.toarray()
X_test_tensor = torch.tensor(X_test_encoded_dense, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

total_y_tensor = torch.tensor(total_y, dtype=torch.long)

print(X_train_tensor.shape)
print(X_test_tensor.shape)
X_test_tensor1 = X_test_tensor.reshape(-1,1,56,1)
X_train_tensor1 = X_train_tensor.reshape(-1,1,56,1)
# y_train_tensor1 = y_train_tensor.reshape(-1,1,1,1)
# y_test_tensor1 = y_test_tensor.reshape(-1,1,1,1)
# 创建TensorDataset和DataLoader
total_x_tensor1 = total_x_tensor.reshape(-1,1,56,1)
train_dataset = TensorDataset(X_train_tensor1, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor1, y_test_tensor)
total_dataset = TensorDataset(total_x_tensor1,total_y_tensor)
# train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

total_loader = DataLoader(dataset= total_dataset, batch_size=128, shuffle= False)
# 现在你可以使用train_loader和test_loader来训练和评估你的CNN模型

num_features = 56  # 获取编码后的特征数量

# class CNNModel(nn.Module):
#     def __init__(self, num_classes, num_features):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=1, padding=1)
#         # 调整池化层参数
#         self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)  # 使用ceil_mode确保输出尺寸不为0
#         # 调整全连接层的输入特征数
#         self.fc1 = nn.Linear(2048 , 512)  # 假设池化后的高度为6
#         self.fc2 = nn.Linear(512, num_classes)
#
#     # 其余代码保持不变
#
#     def forward(self, x):
#         # 第一个卷积层和池化层
#         x = self.pool(F.relu(self.conv1(x)))
#         # 第二个卷积层和池化层
#         x = self.pool(F.relu(self.conv2(x)))
#
#
#         # 计算展平后的尺寸
#         # 假设经过两次卷积和池化后，x的尺寸变为(N, C, H, W)
#         # 我们需要计算H和W的实际值，并用它们来计算展平后的尺寸
#         # 这里需要根据实际的输出尺寸来调整
#         N, C, H, W = x.size()  # 获取x的尺寸
#         flattened_size = C * H * W  # 计算展平后的尺寸
#         # print(N,C,H,W)
#         # print(flattened_size)
#         x = x.view(-1, flattened_size)  # 展平层
#         # 全连接层
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
# 将编码后的数据转换为NumPy数组，逻辑回归需要NumPy数组格式的输入
X_train_np = X_train_tensor.numpy()
X_test_np = X_test_tensor.numpy()
total_x_np = total_x_tensor.numpy()
def train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    return accuracy_knn

# 调用函数来训练和评估KNN
accuracy_knn = train_and_evaluate_knn(X_train_np, y_train, total_x_np, total_y)
print(f'Accuracy of the KNN model: {accuracy_knn * 100:.2f}%')
# 创建逻辑回归模型实例

# 如果你想封装成函数
def train_and_evaluate_lr(X_train, y_train, X_test, y_test):
    lr_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    return accuracy_lr

# 调用函数来训练和评估逻辑回归
accuracy_lr = train_and_evaluate_lr(X_train_np, y_train, total_x_np, total_y)
print(f'Accuracy of the Logistic Regression model: {accuracy_lr * 100:.2f}%')

# ...省略了数据预处理和加载的代码...

# 假设你已经有了训练集和测试集的张量X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

# 创建随机森林分类器实例
# 假设X_train_tensor和y_train_tensor是已经准备好的训练数据和标签
# 以及X_test_tensor和y_test_tensor是测试数据和标签
def train_and_evaluate_rf(X_train, y_train, X_test, y_test, n_estimators=100):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    return accuracy_rf

# 调用函数来训练和评估随机森林
accuracy_rf = train_and_evaluate_rf(X_train_encoded, y_train, total_x, total_y)
print(f'Accuracy of the Random Forest model: {accuracy_rf * 100:.2f}%')
# 定义SVM模型，使用核函数RBF

# 如果需要使用SVM进行分类，可以定义一个函数来封装这个过程
def train_and_evaluate_svm(X_train, y_train, X_test, y_test, kernel_type='rbf'):
    svm_model = SVC(kernel=kernel_type, gamma='scale')
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
accuracy = train_and_evaluate_svm(X_train_tensor, y_train_tensor, total_x_tensor, total_y_tensor)
print(f'Accuracy of the SVM model: {accuracy * 100:.2f}%')

class CNNModel(nn.Module):
    def __init__(self, num_classes, num_features):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # 调整池化层参数
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)  # 使用ceil_mode确保输出尺寸不为0
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        # 调整全连接层的输入特征数

        self.fc1 = nn.Linear(2880, 1024)  # 假设池化后的高度为56
        self.fc4 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    # 其余代码保持不变

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout2(x)


        # 计算展平后的尺寸
        # 假设经过两次卷积和池化后，x的尺寸变为(N, C, H, W)
        # 我们需要计算H和W的实际值，并用它们来计算展平后的尺寸
        # 这里需要根据实际的输出尺寸来调整
        N, C, H, W = x.size()  # 获取x的尺寸
        flattened_size = C * H * W  # 计算展平后的尺寸
        # print(N,C,H,W)
        # print(flattened_size)
        x = x.view(-1, flattened_size)  # 展平层
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc4(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x
# 实例化模型
num_classes = 5  # 假设您有5个类别
model = CNNModel(num_classes, num_features)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
best_accuracy = 0
best_model_state = None
best_model_state1 = None
best_loss = float('inf')
# 训练模型
# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 优化
        optimizer.step()

        # 计算损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        validation_loss = 0.0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
    current_accuracy = 100 * correct / total
    validation_loss /= len(test_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], validation_loss: {validation_loss}, Validation Accuracy: {current_accuracy:.2f}%')
    avg_loss = epoch_loss*0 + validation_loss*1
    # 检查是否是最佳准确率
    avg_acc = epoch_accuracy*0 + current_accuracy*1
    if avg_acc >= best_accuracy:

        best_accuracy = avg_acc
        torch.save(model.state_dict() ,'acc_model.pth')
        print("epoch:", epoch, "best:", best_accuracy)
    if avg_loss <= best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict() ,'loss_model.pth')
        print("epoch:", epoch, "best:", best_loss)
    model.train()
    # 计算平均损失和准确率




model.load_state_dict(torch.load('acc_model.pth'))
model.eval()
with torch.no_grad():
    predictions = []
    true_labels = []
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
conf_matrix = confusion_matrix(true_labels, predictions)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print(f'Accuracy: {accuracy * 100:.2f}%')
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Best Model on Test Set')
plt.show()

model.load_state_dict(torch.load('loss_model.pth'))
# 使用最佳模型在测试集上进行最终评估，并绘制混淆矩阵
model.eval()
with torch.no_grad():
    predictions = []
    true_labels = []
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
conf_matrix = confusion_matrix(true_labels, predictions)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print(f'Accuracy: {accuracy * 100:.2f}%')
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Best Model on Test Set')
plt.show()
