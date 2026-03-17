import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# 예시 데이터 생성
drug_data = {
    'DrugID': ['Drug1', 'Drug2', 'Drug3'],
    'SMILES': ['CCO', 'CCN', 'CCO'],
    'Target': [1, 0, 1]  # 1: 부작용 발생, 0: 부작용 없음
}
output_data = pd.read_csv('./data/label.csv')  # 부작용 선택
smile_data = pd.read_csv('./data/smile_data_v3.csv')
smile_data = smile_data.sort_values(by='approval_year')
print(smile_data)
output_data = merged_df = pd.merge(output_data, smile_data["drug_id"], on='drug_id', )

df_smiles = smile_data['smiles']
df_drug_id = smile_data['drug_id']
df_output = output_data['']
df_drugs = pd.DataFrame(drug_data)

# SMILES를 그래프 데이터로 변환하는 함수
def smiles_to_graph(smiles):
    # 여기에서는 각 원자의 특성을 무작위로 생성하는 예시 코드를 작성했습니다.
    # 실제로는 RDKit 등을 사용하여 SMILES를 분자 그래프로 변환할 수 있습니다.
    num_atoms = len(smiles)
    atom_features = torch.randn(num_atoms, 5)  # 임의의 5차원 특성 벡터
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # 임의의 엣지 인덱스
    return Data(x=atom_features, edge_index=edge_index)

# 각 약물의 SMILES를 그래프 데이터로 변환하고 리스트에 저장
graph_data_list = []
for _, row in df_drugs.iterrows():
    graph_data = smiles_to_graph(row['SMILES'])
    graph_data.y = torch.tensor([row['Target']], dtype=torch.float32)
    graph_data_list.append(graph_data)

# 데이터를 훈련 및 검증용으로 분리
train_data, val_data = train_test_split(graph_data_list, test_size=0.2, random_state=42)

# DataLoader 생성
batch_size = 1  # 배치 크기를 1로 설정
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# 모델 정의
class MyModel(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_features, hidd_dim, kernel_size=(1, 5))
        self.fc1 = nn.Linear(hidd_dim, kge_dim)
        self.fc2 = nn.Linear(kge_dim, rel_total)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 및 손실 함수, 옵티마이저 정의
model = MyModel(1, 64, 64, 2)  # 예시 모델
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 훈련 및 검증 수행
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epoch in range(epochs):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x.unsqueeze(1).float())
        loss = criterion(out, data.y.long())
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            out = model(data.x.unsqueeze(1).float())
            loss = criterion(out, data.y.long())
            val_loss += loss.item()
            _, predicted = out.max(1)
            total += data.y.size(0)
            correct += predicted.eq(data.y).sum().item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {val_loss/len(val_loader)}, Accuracy: {100*correct/total}%')
