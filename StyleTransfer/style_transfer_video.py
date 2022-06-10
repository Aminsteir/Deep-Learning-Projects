import os

import cv2
import numpy as np

from StyleTransfer import speed_style_transfer


def count_frames(video):
    total = round(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return total


def write_vid(video_path, output_path, style_path=None, color=True):
    capture = cv2.VideoCapture(video_path)

    width = 960
    height = 540
    shape = (width, height)

    fps = round(capture.get(cv2.CAP_PROP_FPS))
    count = 0

    vid = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, shape)

    frames = count_frames(capture)

    success, image = capture.read()

    while success:
        image = cv2.resize(image, shape)

        if style_path is not None:
            image = speed_style_transfer.stylize(speed_style_transfer.img_to_pil_image(image), style_path,
                                                 gpu_if_avail=True, preserve_color=color)

        image = image.astype(np.uint8)

        cv2.imshow('Frame', image)

        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

        count += 1
        print('{} out of {} images'.format(count, frames))
        vid.write(image)
        success, image = capture.read()

    capture.release()
    vid.release()
    cv2.destroyAllWindows()

    return None


def main():
    print('Video Style Transfer')

    overwrite = True

    styles = [
        {'name': 'Mosaic', 'model': '../saved_models/mosaic.pth', 'color': True},
        {'name': 'Gold', 'model': '../saved_models/gold.model', 'color': False},
        {'name': 'Starry', 'model': '../saved_models/starry.model', 'color': False},
        {'name': 'Udnie', 'model': '../saved_models/udnie.pth', 'color': True},
        {'name': 'Candy', 'model': '../saved_models/candy.pth', 'color': False},
        {'name': 'Wave', 'model': '../saved_models/Wave.model', 'color': True},
        {'name': 'Gogh_Field', 'model': '../saved_models/Gogh_Field.model', 'color': True},
        {'name': 'Persian', 'model': '../saved_models/Persian.model', 'color': True},
        {'name': 'Rain Princess', 'model': '../saved_models/rain_princess.pth', 'color': False},
        {'name': 'Graffiti', 'model': '../saved_models/Graffiti.model', 'color': False},
        {'name': 'Splashy', 'model': '../saved_models/Splashy.model', 'color': False},
        {'name': 'Pastel', 'model': '../saved_models/Pastel.model', 'color': False},
        {'name': 'Starry Rhone', 'model': '../saved_models/Starry_Rhone.model', 'color': True},
    ]

    files = [
        {
            'name': 'My Friend Pedro',
            'path': '../videos/Pedro.mp4',
            'models': [i for i in range(0, len(styles))],
        }
    ]

    base_out_dir = '../outputs/videos'

    for video in files:
        name = video.get('name')
        vid_path = video.get('path')

        vid_folder = '{}/{}'.format(base_out_dir, name)

        if os.path.exists(vid_folder) and not overwrite:
            continue
        elif not os.path.exists(vid_folder):
            os.mkdir(vid_folder)

        write_vid(vid_path, '{}/Original.mp4'.format(vid_folder))

        for model in video.get('models'):
            style = styles[model]
            model = style.get('name')

            write_vid(vid_path, '{}/{}.mp4'.format(vid_folder, model.replace(" ", "")),
                      style.get('model'), style.get('color'))


if __name__ == '__main__':
    main()
