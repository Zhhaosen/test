import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设你已经加载了数据到DataFrame中
df = pd.read_excel('t1.xlsx')
print(df.columns)
# 选择特征列和标签列
features = ['mid-term', 'final', 'faculty', 'department']
label = 'Label'

# 数据清洗，例如删除缺失值
df = df.dropna()

# 划分训练集和测试集
X = df[features]
y = df[label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征编码和缩放
numerical_cols = ['mid-term', 'final']
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


print(X_train_encoded.shape)
print(X_test_encoded.shape)
# 将编码后的数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_encoded, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_encoded, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

X_test_tensor1 = X_test_tensor.reshape(-1,1,12,1)
X_train_tensor1 = X_train_tensor.reshape(-1,1,12,1)
# y_train_tensor1 = y_train_tensor.reshape(-1,1,1,1)
# y_test_tensor1 = y_test_tensor.reshape(-1,1,1,1)
# 创建TensorDataset和DataLoader

train_dataset = TensorDataset(X_train_tensor1, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor1, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=31, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=31, shuffle=False)

# 现在你可以使用train_loader和test_loader来训练和评估你的CNN模型

num_features = 12  # 获取编码后的特征数量
num_classes = 5
class CNNModel(nn.Module):
    def __init__(self, num_classes, num_features):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, padding=1)
        # 调整池化层参数
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)  # 使用ceil_mode确保输出尺寸不为0
        # 调整全连接层的输入特征数
        self.fc1 = nn.Linear(640, 512)  # 假设池化后的高度为6
        self.fc2 = nn.Linear(512, 6)

    # 其余代码保持不变

    def forward(self, x):
        # 第一个卷积层和池化层
        x = self.pool(F.relu(self.conv1(x)))
        # 第二个卷积层和池化层
        x = self.pool(F.relu(self.conv2(x)))


        # 计算展平后的尺寸
        # 假设经过两次卷积和池化后，x的尺寸变为(N, C, H, W)
        # 我们需要计算H和W的实际值，并用它们来计算展平后的尺寸
        # 这里需要根据实际的输出尺寸来调整
        N, C, H, W = x.size()  # 获取x的尺寸
        flattened_size = C * H * W  # 计算展平后的尺寸
        x = x.view(-1, flattened_size)  # 展平层

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
num_classes = 5  # 假设您有5个类别
model = CNNModel(num_classes, num_features)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')



# 保存模型
torch.save(model.state_dict(), 'cnn_model.pth')

# 加载模型
model.load_state_dict(torch.load('cnn_model.pth'))

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total}%')