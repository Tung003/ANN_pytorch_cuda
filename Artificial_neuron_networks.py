import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt


"""
dữ liệu MNIST 60000 mẫu dữ liệu chữ số viết tay
kích thước [1,28,28]=[1,784]
xây dựng mạng neuron nhân tạo với 5 lớp fullyconected
sử dụng hàm kích hoạt RELU và kiểm tra kết quả.
"""

transform=transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,),(0.5,))])

training_data_MNIST=datasets.MNIST(root="data",
                                   train=True,
                                   download=False,
                                   transform=transform
                                   )

testing_data_MNIST=datasets.MNIST(root="./data",
                                  train=False,
                                  download=False,
                                  transform=transform)

training_data_Loader=DataLoader(dataset=training_data_MNIST,
                             batch_size=32,
                             shuffle=True)
testing_data_Loader=DataLoader(dataset=testing_data_MNIST,
                               batch_size=32,
                               shuffle=True)

class Artificial_neuron_networks(nn.Module):
    def __init__(self):
        super().__init__()
        self.fl1=nn.Flatten()
        self.fc1=nn.Linear(in_features=784,out_features=512)
        self.act1=nn.ReLU()
        self.fc2=nn.Linear(in_features=512,out_features=256)
        self.act2=nn.ReLU()
        self.fc3=nn.Linear(in_features=256,out_features=128)
        self.act3=nn.ReLU()
        self.fc4=nn.Linear(in_features=128,out_features=64)
        self.act4=nn.ReLU()
        self.fc5=nn.Linear(in_features=64,out_features=10)

    def forward(self,x):
        x=self.fl1(x)
        x=self.fc1(x)
        x=self.act1(x)
        x=self.fc2(x)
        x=self.act2(x)
        x=self.fc3(x)
        x=self.act3(x)
        x=self.fc4(x)
        x=self.act4(x)
        x=self.fc5(x)
        return x

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(training_data_MNIST[0][0].shape)
    # datax, label = next(iter(training_data_Loader))
    # print(datax, label)
    model = Artificial_neuron_networks().to(device)# đẩy model lên trên GPU

    num_epochs=10
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.002,betas=(0.9,0.999))

    for epoch in range(num_epochs):
        model.train()# chuyển model về chế độ huấn luyện
        loss_total=0
        for image,label in training_data_Loader:
            image,label=image.to(device),label.to(device)
            optimizer.zero_grad()
            output=model(image)
            loss=criterion(output,label)
            loss.backward()
            optimizer.step()
            loss_total+=loss.item()
        print(f"Epoch: {epoch+1}, Loss: {loss_total/len(training_data_Loader):.4f}")
# đánh giá mô hình
    model.eval()# chuyển model về chế độ đánh giá
    correct, total=0,0
    with torch.no_grad():
        for image,label in testing_data_Loader:
            image,label=image.to(device),label.to(device)
            outputs=model(image)
            c,predicted=torch.max(outputs,1)
            total+=label.size(0)
            correct+=(predicted==label).sum().item()

    accuracy=100*correct/total
    print(f"Độ chính xác trên bộ test: {accuracy:.2f} %")

    model.eval()
    data_iter=iter(testing_data_Loader)
    image,label=next(data_iter)
    image,label=image[:10].to(device),label[:10].to(device)

    with torch.no_grad():
        outputs = model(image)
        z, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()

    for i in range(10):
        img = image[i].cpu().numpy().squeeze()  #chuyển ảnh từ tensor và gpu về numpy và cpu để hiển thị
        real_label = label[i].item()
        predicted_label = predicted[i].item()

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Nhãn thực: {real_label}\nNhãn dự đoán: {predicted_label}", fontsize=12)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
