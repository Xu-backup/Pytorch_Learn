from Deal_MNIST import Net
import torch
from torchvision import transforms
from PIL import Image
import os


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = '.\MNIST\digit.jpg'
    data_transform = transforms.Compose(
            [transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))])

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img.show()

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)  ##增加batch维度
    model = Net().to(device)

    # load model weights
    weights_path = ".\\MNIST\\MINST.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict_cla = torch.argmax(output).numpy()

    print_res = "class: {}   prob: {:.3}".format(str(predict_cla),
                                                 output[predict_cla].numpy())
    print(print_res)

if __name__ == '__main__':
    main()