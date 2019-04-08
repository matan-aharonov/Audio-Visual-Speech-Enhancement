import numpy as np
import cv2
import sys


from facedetection.face_detection import FaceDetector
from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader


def preprocess_video_sample(video_file_path, slice_duration_ms, mouth_height=128, mouth_width=128):
    print("preprocessing %s" % video_file_path)

    face_detector = FaceDetector()

    with VideoFileReader(video_file_path) as reader:
        frames = reader.read_all_frames(convert_to_gray_scale=True)
        mouth_cropped_frames = np.zeros(shape=(mouth_height, mouth_width, reader.get_frame_count()), dtype=np.float32)
        for i in range(reader.get_frame_count()):
            mouth_cropped_frames[:, :, i] = face_detector.crop_mouth(frames[i], bounding_box_shape=(mouth_width, mouth_height))

        frames_per_slice = int((float(slice_duration_ms) / 1000) * reader.get_frame_rate())
        n_slices = int(float(reader.get_frame_count()) / frames_per_slice)

        slices = [
            mouth_cropped_frames[:, :, (i * frames_per_slice):((i + 1) * frames_per_slice)]
            for i in range(n_slices)
        ]

    return np.stack(slices), reader.get_frame_rate()


def extractFrames(pathOut):
    count = 0
    vidcap = cv2.VideoCapture(0)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
        success,image = vidcap.read()
        print ('Read a new frame: ', success)

        frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)

        print(frameCount)
        print(frameWidth)
        print(frameHeight)
        print(fps)

        video_shape = (200, frameHeight, frameWidth, 3)
        frames[count,] = image
        frames = np.ndarray(shape=video_shape, dtype=np.uint8)


        # print(frames)
        print(vidcap.get(cv2.CAP_PROP_POS_MSEC))

      # cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file

        count = count + 1


if __name__ == '__main__':

    # Set up tracker.
    tracker = cv2.TrackerMedianFlow_create()

    # Read video
    video = cv2.VideoCapture(0)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, "MEDIANFLOW" + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
