import torch
from torchvision import transforms
from PIL import Image


def AIO_classifier(frame, model):
    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(size=128),
        transforms.CenterCrop(size=128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    device = torch.device("cpu")
    img = Image.fromarray(frame.astype('uint8'), 'RGB')

    # Perform prediction and plot results
    with torch.no_grad():
        # img = Image.open(image_path).convert('RGB')
        inputs = preprocess(img).unsqueeze(0).to(device)
        outputs = model(inputs)
        class_index = torch.max(outputs.data, 1)[1].numpy()[0]
        return class_index


if __name__ == '__main__':
    PATH = r'models/resnet50_0.pth'

    # Load the model for testing
    model = torch.load(PATH)
    model.eval()
    image_path = r'C:\Users\chend\PycharmProjects\E_Scooter_v1\Data\test\frame_13700.png'

    class_names = ['Bicyclists', 'Buildings', 'Crosswalk', 'Nature', 'Other_infrastructure', 'Pedestrians',
                   'Road', 'Sidewalks', 'Traffic_signals', 'Undetermined', 'Vehicles']

    object_class = class_names[AIO_classifier(image_path, model)]
    print(object_class)
