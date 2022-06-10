import cv2
import os


def count_frames(video):
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return total


def main():
    file_name = 'Nature'
    video_cap = cv2.VideoCapture('../videos/{}.mp4'.format(file_name))

    t_frames = count_frames(video_cap)
    add_frame = t_frames // int(input('{} total frames. Designated number of frames: '.format(t_frames)))

    folder_path = '../data/videos/{}'.format(file_name)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    success, image = video_cap.read()

    count = 0
    count_2 = 0

    while success:
        if (count_2 + 1) % add_frame == 0:
            count_2 = 0
            cv2.imwrite("../data/videos/{}/frame{}.jpg".format(file_name, count), image)  # save frame as JPEG file
            count += 1
        success, image = video_cap.read()
        count_2 += 1

    video_cap.release()


if __name__ == '__main__':
    main()
