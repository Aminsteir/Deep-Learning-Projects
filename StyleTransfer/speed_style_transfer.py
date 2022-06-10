import os
import re
import time
from distutils import util as distutils

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from StyleTransfer.transform_net import TransformerNet
from StyleTransfer.vgg import Vgg16


def train(style_path, train_dataset_path, model_out_dir, model_out_name, max_style_size=1440, image_size=256,
          batch_size=4, shuffle_training=True, learning_rate=1e-3, epochs=6,
          content_weight=1e5, style_weight=1e10, log_interval=100, use_gpu_avail=True, normalize_input=False):
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu_avail else 'cpu')
    np.random.seed(123)
    torch.manual_seed(123)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder(train_dataset_path, transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle_training)

    transformer = TransformerNet().to(device)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)

    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    if normalize_input:
        style_transform = transforms.Compose([
            transforms.ToTensor(),
            # Learn only features from an image and not color ranges
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    else:
        style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

    style = load_image(style_path, max_style_size)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(normalize_batch(style))
    gram_style = [gram_matrix(x) for x in features_style]

    for epoch in range(1, epochs + 1):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0

        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % log_interval == 0:
                message = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), epoch, count, len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(message)
            torch.cuda.empty_cache()

    transformer.eval().cpu()
    save_model_path = os.path.join(model_out_dir, model_out_name + '.model')
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(content_image, model_path, input_255=False, gpu_if_avail=True,
            preserve_color=False, out_scheme=None):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if gpu_if_avail and torch.cuda.is_available() else 'cpu')

    if not input_255:
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    else:
        content_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model_path)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
    data = ttoi(output[0])
    if preserve_color:
        data = transfer_color(ttoi(content_image), data)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    if out_scheme is not None:
        return cv2.cvtColor(data, out_scheme)
    return data.clip(0, 255)


def transfer_color(src, dest):
    src, dest = src.clip(0, 255), dest.clip(0, 255)
    height, width, _ = src.shape
    dest = cv2.resize(dest, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    dest_gray = cv2.cvtColor(dest, cv2.COLOR_RGB2GRAY)  # 1 Extract the Destination's luminance
    src_yiq = cv2.cvtColor(src, cv2.COLOR_RGB2YCrCb)  # 2 Convert the Source from BGR to YIQ/YCbCr
    src_yiq[..., 0] = dest_gray  # 3 Combine Destination's luminance and Source's IQ/CbCr

    return cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR).clip(0, 255)  # 4 Convert new image from YIQ back to RGB


def img_to_pil_image(image):
    return Image.fromarray(image).convert('RGB')


def ttoi(tensor):
    ttoi_t = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensor = tensor.squeeze()
    img = ttoi_t(tensor)
    img = img.cpu().numpy()
    img = img.transpose(1, 2, 0)
    return img.clip(0, 255)


def upscale(image_array):
    rdn = RDN(weights='noise-cancel')
    up_image = rdn.predict(image_array, by_patch_of_size=50)
    return up_image


def load_image(img_path, shape=None):
    image = Image.open(img_path).convert('RGB')
    size = max(image.size)
    if shape is not None:
        size = min(shape, size)

    transform = transforms.Compose([
        transforms.Resize(size)
    ])

    return transform(image)


def gram_matrix(x):
    b, ch, h, w = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def main():
    if bool(distutils.strtobool(input('Train? (y/n): '))):
        training = [
            {'Name': 'Penn', 'Image': '../images/Watercolor_Penn.jpg', 'Normalized_Input': False}
        ]

        output_dir = '../saved_models/'
        train_dir = '../data/videos/'

        for model in training:
            train(model.get('Image'), train_dir, output_dir, model.get('Name'),
                  epochs=4, log_interval=10, normalize_input=model.get('Normalized_Input'))
            torch.cuda.empty_cache()
    else:
        overwrite_dir = bool(distutils.strtobool(input('Overwrite output directories? (y/n): ')))

        images = {
            'Sam Harris': load_image('../images/SamHarris.jpg'),
            'Sam Harris 2': load_image('../images/SamHarris_2.jpg', shape=1440),
            'Tulsi Gabbard': load_image('../images/Gabbard.jpg'),
            'Tulsi Gabbard 2': load_image('../images/Gabbard_2.jpg'),
            'Hessam': load_image('../images/hessam.jpg'),
            'Ray': load_image('../images/Ray.jpg'),
            'Roses': load_image('../images/Roses.jpg', shape=1440),
            'Miles': load_image('../images/Miles.jpg', shape=1440),
            'Spider Verse': load_image('../images/SpiderVerse2.jpg'),
            'Planet': load_image('../images/Planet.jpg'),
            'Moon': load_image('../images/Moon.jpg'),
            'Mars': load_image('../images/Mars.jpg'),
            'Amin': load_image('../images/SchoolPhoto.jpg'),
            'The Crew 2': load_image('../images/TheCrew2.png', shape=1440),
            'City': load_image('../images/City.jpg', shape=1440),
            'Minecraft': load_image('../images/Minecraft.png', shape=1440),
            'Amazing Spider-Man': load_image('../images/Amazing_Spider-Man.jpg', shape=1440),
            'Spider-Man PS4': load_image('../images/Spider-Man_PS4.jpg', shape=1440),
            'Art Landscape': load_image('../images/Artist_Landscape.jpg', shape=1440),
            'Lake': load_image('../images/lake.jpg', shape=1440),
            'Bean': load_image('../images/bean.jpg', shape=1440),
            'Funny Trump': load_image('../images/Trump.jpg', shape=1440),
            'Brain': load_image('../images/Brain.jpg', shape=1440),
            'Amber Doorway': load_image('../images/amber.jpg', shape=1440),
            'Kyran - Ghost': load_image('../images/Cryan.jpg', shape=1440),
            'Mona Lisa': load_image('../images/MonaLisa.jpg', shape=1440),
            'MIT': load_image('../images/MIT.jpg', shape=1440),
            'Iron Man': load_image('../images/Iron_Man.jpg', shape=1440),
            'Cyberpunk 2077': load_image('../images/Cyberpunk.jpg', shape=1440),
            'ASUS - CyberCity': load_image('../images/CyberCity.jpg', shape=1440),
            'Star Wars': load_image('../images/Star_Wars.jpg', shape=1440),
            'Laptop': load_image('../images/Laptop.jpg', shape=1440),
            'Exploration': load_image('../images/Explore.jpg', shape=1440),
            'Just Cause 4': load_image('../images/JC4.jpg', shape=1440),
            'Chess 1': load_image('../images/Chess1.jpg', shape=1440),
            'Chess 2': load_image('../images/Chess2.jpg', shape=1440),
            'Chess 3': load_image('../images/Chess3.jpg', shape=1440),
            'Kirito': load_image('../images/Kirito.jpg', shape=3840),
        }
        models = {
            'Starry Night': '../saved_models/starry.model', 'Mosaic': '../saved_models/mosaic.pth',
            'Udnie': '../saved_models/udnie.pth', 'Rain_Princess': '../saved_models/rain_princess.pth',
            'Candy': '../saved_models/candy.pth', 'Gold': '../saved_models/gold.model',
            'Persian': '../saved_models/Persian.model', 'Splashy': '../saved_models/Splashy.model',
            'Graffiti': '../saved_models/Graffiti.model', 'Gogh Field': '../saved_models/Gogh_Field.model',
            'Pastel': '../saved_models/Pastel.model', 'Starry Night at Rhone': '../saved_models/Starry_Rhone.model',
            'Money': '../saved_models/Money.model', 'Penn Watercolor': '../saved_models/Penn.model'
        }

        output_folder = '../outputs/speed_testing/PFP'

        for person in images:
            output_dir = '{}/{}'.format(output_folder, person)

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            # Write the original picture for comparison
            original_image = np.array(images.get(person))[:, :, ::-1].copy()
            cv2.imwrite('{}/Original.jpg'.format(output_dir), original_image)

            for model in models:
                model_dir = '{}/{}'.format(output_dir, model)

                if os.path.exists(model_dir) and not overwrite_dir:
                    continue
                elif not os.path.exists(model_dir):
                    os.mkdir(model_dir)

                cv2.imwrite('{}/Recolored.jpg'.format(model_dir, model),
                            stylize(images.get(person), models.get(model), gpu_if_avail=True,
                                    preserve_color=True, out_scheme=cv2.COLOR_RGB2BGR))
                cv2.imwrite('{}/Regular.jpg'.format(model_dir, model),
                            stylize(images.get(person), models.get(model), gpu_if_avail=True, preserve_color=False))
                torch.cuda.empty_cache()

            print('Finished {}'.format(person))


if __name__ == '__main__':
    main()
