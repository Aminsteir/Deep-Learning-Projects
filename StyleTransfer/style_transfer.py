import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models


class StyleTransfer:
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.model = models.vgg19(pretrained=True).features
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model = self.model.to(device)

    def process_image(self, content, style, steps=2100, frames=700):
        content = content.to(self.device)
        style = style.to(self.device)

        content_features = get_features(content, self.model)
        style_features = get_features(style, self.model)

        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
        style_weights = {'conv1_1': 1.0, 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}

        content_weight = 1
        style_weight = 1e6

        target = content.clone().requires_grad_(True).to(self.device)

        optimizer = torch.optim.Adam([target], lr=0.003)

        height, width, channels = im_convert(target).shape
        image_array = np.empty(shape=(frames if steps % frames == 0 else frames + 1, height, width, channels))
        capture_frame = steps / frames
        counter = 0

        for step in range(1, steps + 1):
            target_features = get_features(target, self.model)
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
            style_loss = 0

            for layer in style_weights:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                style_gram = style_grams[layer]
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                _, d, h, w = target_feature.shape
                style_loss += layer_style_loss / (d * h * w)

            total_loss = content_loss * content_weight + style_loss * style_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % capture_frame == 0 or step == steps:
                print('Added frame {}'.format(counter))
                image_array[counter] = im_convert(target)
                counter += 1

        return target, image_array


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }

    features = {}

    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            features[layers[name]] = image

    return features


def img_to_pil_image(image):
    return Image.fromarray(np.uint8(image)).convert('RGB')


def transform_image(image, size):
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return in_transform(image).unsqueeze(0)


def load_image(img_path, max_size=1024, shape=None):
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    return transform_image(image, size)


def im_convert(tensor):
    # Converts the image to numpy array
    image = tensor.cpu().clone().detach().numpy()

    image = image.squeeze()

    image = image.transpose(1, 2, 0)

    # Apply some transformation
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))

    # Clips the image
    image = image.clip(0, 1)
    return image


def to_cv2_img(item):
    img = item
    img *= 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def write_video(path, images, frame_h, frame_w):
    vid = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (frame_w, frame_h))

    for i in range(0, len(images)):
        img = to_cv2_img(images[i])
        print(img)
        vid.write(img)

    vid.release()


def main():
    f_name = input('File Name: ')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transfer = StyleTransfer(device=device)

    content = load_image('../images/Ray.jpg')
    style = load_image('../images/Van_Gogh.jpg', shape=content.shape[-2:])

    steps = 100
    frames = 100

    target, image_array = transfer.process_image(content, style, steps, frames)

    print('Finished Processing Image')

    final_image = to_cv2_img(im_convert(target))
    cv2.imwrite('../outputs/{}.jpg'.format(f_name), final_image)

    frame_height, frame_width, _ = im_convert(target).shape
    write_video('../outputs/{}.mp4'.format(f_name), image_array, frame_height, frame_width)


if __name__ == '__main__':
    main()
