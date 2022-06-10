import cv2
import numpy as np
import pyvirtualcam

from StyleTransfer import speed_style_transfer


def main():
    print('Style Transfer Live Camera')

    cv2.startWindowThread()

    cv2_type = None  # Sometimes, you can use cv2.CAP_DSHOW

    if cv2_type is None:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(0, cv2_type)

    width, height = (1920, 1080)

    models = [
        {'model': '../saved_models/mosaic.pth'},
        {'model': '../saved_models/gold.model'},
        {'model': '../saved_models/starry.model'},
        {'model': '../saved_models/udnie.pth'},
        {'model': '../saved_models/candy.pth'},
        {'model': '../saved_models/Wave.model'},
        {'model': '../saved_models/Gogh_Field.model'},
        {'model': '../saved_models/Persian.model'},
        {'model': '../saved_models/rain_princess.pth'},
        {'model': '../saved_models/Graffiti.model'},
        {'model': '../saved_models/Splashy.model'},
        {'model': '../saved_models/Pastel.model'},
        {'model': '../saved_models/Starry_Rhone.model'}
    ]
    current_idx = 0

    transfer = False
    color = False
    transfer_w, transfer_h = (600, 400)

    with pyvirtualcam.Camera(width=width, height=height, delay=0, fps=10) as cam:
        while True:
            ret, image = vid.read()

            if transfer:
                model = models[current_idx].get('model')
                image = cv2.resize(image, (transfer_w, transfer_h), cv2.INTER_AREA)
                image = speed_style_transfer.stylize(speed_style_transfer.img_to_pil_image(image), model,
                                                     gpu_if_avail=True, preserve_color=color,
                                                     out_scheme=cv2.COLOR_RGB2RGBA)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

            resized_frame = cv2.resize(image.astype(np.uint8), (width, height))

            cv2.imshow('Live Cam', resized_frame)

            cam.send(resized_frame)

            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                cam.close()
                cv2.destroyAllWindows()
                break
            elif key == 32:
                transfer = not transfer
            elif key & 0xFF == ord('n'):
                current_idx += 1
                if current_idx >= len(models):
                    current_idx = 0
            elif key & 0xFF == ord('p'):
                current_idx -= 1
                if current_idx < 0:
                    current_idx = len(models) - 1
            elif key & 0xFF == ord('c'):
                color = not color

            cam.sleep_until_next_frame()

    vid.release()
    exit()


if __name__ == '__main__':
    main()
