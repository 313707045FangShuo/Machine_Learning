## PreTrain

- 資料處理、Model Setting、Training 和 Validation
- 使用 early stopping (5) 來防止 overfitting
- 訓練結束後，保存最佳的模型權重同時製作 confusion matrix

#### 1. **ResNet50 模型設定**
使用 ResNet50 預訓練模型，根據資料集中的類別數，調整分類層的輸出。

python
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


#### 2. **凍結底層參數**
將模型中大部分層的權重凍結，僅在最上層（layer4 和 fc）進行訓練，以保留預訓練的權重，並著重在微調分類層。

python
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False


#### 3. **不同學習率的 Optimizer**
使用 **Adam optimizer**，並為不同的層設置了不同的學習率。

python
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters(), 'lr': 0.001}
])


#### 4. **訓練與驗證過程**

- **Training 階段**：在每個 batch 上計算損失，進行反向傳播並更新模型權重。
- **Validation 階段**：計算驗證集上的損失和準確率，並收集預測結果與真實標籤。

python
for epoch in range(num_epochs):
    # 訓練與驗證
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

## CNN

#### 1. **模型設定與凍結底層權重**
加載 ResNet50 預訓練模型，並凍結其底層權重（layer4 以前的層），僅訓練最後的卷積層和分類層。

python
model = models.resnet50(pretrained=True)
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

# 替換分類層
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

#### 2. **定義 Loss Function 與 Optimizer**
使用交叉熵損失函數作為辛普森分類的 loss function，並設置分層學習率，讓分類層的學習率較高，卷積層的學習率較低

python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters(), 'lr': 0.001}
])

#### 3. **訓練與驗證模型**

python
for epoch in range(num_epochs):
    # 訓練階段
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 驗證階段
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            epoch_true_labels.extend(labels.cpu().tolist())
            epoch_pred_labels.extend(outputs.argmax(dim=1).cpu().tolist())

#### 4. **保存最佳模型**

python
torch.save(model.state_dict(), "resnet50_simpsons_finetuned.pth")
print("Model saved as resnet50_simpsons_finetuned.pth")


#### 5. **Task2：Confusion Matrix**

python
cm = confusion_matrix(all_true_labels, all_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.dataset.classes)

fig, ax = plt.subplots(figsize=(20, 20))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches='tight')

## Testing (Task 1)

#### 1. **定義圖片預處理**
python
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 調整圖片大小
    transforms.ToTensor(),  # 轉換為 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
])


#### 2. **讀取訓練好的 ResNet50 model**
python
num_classes = len(train_dataset.dataset.classes)  # 類別數
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替換分類層
model.load_state_dict(torch.load("resnet50_simpsons_finetuned_earlystop.pth"))
model = model.to(device)
model.eval()  # 設置為評估模式


#### 3. **預測結果**
python
results = []
for i in range(1, 10792):  # 測試圖片範圍
    img_path = os.path.join(test_dir, f"{i}.jpg")
    try:
        image = Image.open(img_path).convert("RGB")  # 打開圖片
        input_tensor = transform(image).unsqueeze(0).to(device)  # 預處理
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        character = classes[predicted.item()]  # 取得預測類別名稱
        results.append({"id": i, "character": character})
        print(f"Processed {i}.jpg -> {character}")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")