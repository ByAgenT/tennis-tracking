import argparse
import queue
import pandas as pd
import imutils
from PIL import Image, ImageDraw
import cv2
import numpy as np
import time

from sktime.datatypes._panel._convert import from_2d_array_to_nested
from court_detection.court_detector import CourtDetector
from ball_tracking.tracknet import TrackNet
from players_tracking.trackplayers import *
from utils import get_video_properties
from detection import *
from pickle import load
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict

# Parse parameters
parser = argparse.ArgumentParser()

parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default="")
parser.add_argument("--court_detection", type=bool, default=False)
parser.add_argument("--minimap", type=bool, default=False)
parser.add_argument("--bounce", type=bool, default=False)

args = parser.parse_args()

input_video_path = args.input_video_path
output_video_path = args.output_video_path
court_detection = args.court_detection
minimap = args.minimap
bounce = args.bounce

# TrackNet configuration
n_classes = 256
save_weights_path = 'ball_tracking/model_tennis.h5'
width, height = 640, 360

# Get video FPS & size
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Input Video FPS : {fps}')
print(f'Frame count: {frame_count}')
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Try to determine the total number of frames in the video file
if imutils.is_cv2() is True:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
else:
    prop = cv2.CAP_PROP_FRAME_COUNT
total = int(video.get(prop))

# Load TrackNet model
modelFN = TrackNet
m = modelFN(n_classes, input_height=height, input_width=width)
m.compile(loss='categorical_crossentropy',
          optimizer='adadelta', metrics=['accuracy'])
m.load_weights(save_weights_path)

# Court Detector
court_detector = CourtDetector()

pose_model = YOLO("yolov8m-pose.pt")
object_model = YOLO("yolov8x.pt")
# object_model = YOLO("yolov8x-tennisball.pt")

# Initialize trajectory queue with 7 frames
q = queue.deque()
for i in range(0, 8):
    q.appendleft(None)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(
    output_video_path, fourcc, fps, (output_width, output_height))

# Get video properties
fps, length, v_width, v_height = get_video_properties(video)
print(
    f"Video properties: FPS:{fps} Length:{length} Width:{v_width} Height:{v_height}")

coords = []
frame_i = 0
frames = []
t = []
track_history = defaultdict(lambda: [])


def detect_or_track_court(frame_index: int, frame):
    '''Detects and tracks court and returns court lines'''
    if not court_detection:
        if frame_index == 1:
            print("Skipping court detection...")
        return

    if frame_index == 1:
        # In first video frame we try to detect the court
        print('Detecting the court...')
        return court_detector.detect(frame)
    else:
        # For all other frames we no longer need to detect court, only keep track
        return court_detector.track_court(frame)


def render_track_court(frame_index, lines, frame):
    '''Renders court lines on the frame'''
    if not court_detection:
        if frame_index == 1:
            print("Skipping court rendering...")
        return frame

    # Draw court lines in output video
    for i in range(0, len(lines), 4):
        x1, y1, x2, y2 = lines[i], lines[i+1], lines[i+2], lines[i+3]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
    return cv2.resize(frame, (v_width, v_height))


print("Starting player detection...")
for _ in tqdm(range(frame_count)):
    ret, frame = video.read()
    frame_i += 1

    if ret:
        lines = detect_or_track_court(frame_i, frame)

        results = pose_model.track(
            frame, persist=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(
                230, 230, 230), thickness=10)

        results = object_model.track(
            annotated_frame, persist=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes.xywh.cpu()
        annotated_frame = results[0].plot()

        new_frame = render_track_court(frame_i, lines, annotated_frame)
        frames.append(new_frame)
    else:
        break

video.release()

currentFrame = 0
video = cv2.VideoCapture(input_video_path)
frame_i = 0
last = time.time()  # start counting
for img in tqdm(frames):
    if currentFrame < 3:
        # Skip first 3 frames to slide buffer
        # TODO: We probably should not skip entire loop and instead don't do ball prediction
        currentFrame += 1
        continue

    frame_i += 1

    # detect the ball
    # img is the frame that TrackNet will predict the position
    # since we need to change the size and type of img, copy it to output_img
    output_img = img
    output_img_prev = frames[currentFrame - 1]
    output_img_prev_prev = frames[currentFrame - 2]

    # resize it
    output_img_prev_prev = cv2.resize(output_img_prev_prev, (width, height))
    output_img_prev = cv2.resize(output_img_prev, (width, height))
    img = cv2.resize(img, (width, height))

    # input must be float type
    output_img_prev_prev = output_img_prev_prev.astype(np.float32)
    output_img_prev = output_img_prev.astype(np.float32)
    img = img.astype(np.float32)

    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    X = np.concatenate([np.rollaxis(output_img_prev_prev, 2, 0), np.rollaxis(
        output_img_prev, 2, 0), np.rollaxis(img, 2, 0)], axis=0)
    # predict heatmap
    pr = m.predict(np.array([X]))[0]

    # since TrackNet output is ( net_output_height*model_output_width , n_classes )
    # so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
    pr = pr.reshape((height, width, n_classes)).argmax(axis=2)

    # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
    pr = pr.astype(np.uint8)

    # reshape the image size as original input image
    heatmap = cv2.resize(pr, (output_width, output_height))

    # heatmap is converted into a binary image by threshold method.
    ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

    # find the circle in image with 2<=radius<=7
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                               maxRadius=7)

    # output_img = mark_player_box(output_img, player1_boxes, currentFrame-1)
    # output_img = mark_player_box(output_img, player2_boxes, currentFrame-1)

    PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(PIL_image)

    # check if there have any tennis be detected
    if circles is not None:
        # if only one tennis be detected
        if len(circles) == 1:

            x = int(circles[0][0][0])
            y = int(circles[0][0][1])

            coords.append([x, y])
            t.append(time.time()-last)

            # push x,y to queue
            q.appendleft([x, y])
            # pop x,y from queue
            q.pop()

        else:
            coords.append(None)
            t.append(time.time()-last)
            # push None to queue
            q.appendleft(None)
            # pop x,y from queue
            q.pop()

    else:
        coords.append(None)
        t.append(time.time()-last)
        # push None to queue
        q.appendleft(None)
        # pop x,y from queue
        q.pop()

    # draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
    for i in range(0, 8):
        if q[i] is not None:
            draw_x = q[i][0]
            draw_y = q[i][1]
            bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
            draw = ImageDraw.Draw(PIL_image)
            draw.ellipse(bbox, outline='yellow')
            del draw

    # Convert PIL image format back to opencv image format
    opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

    output_video.write(opencvImage)

    # next frame
    currentFrame += 1

# Everything is done, release the video
video.release()
output_video.release()

if minimap:
    game_video = cv2.VideoCapture(output_video_path)

    fps1 = int(game_video.get(cv2.CAP_PROP_FPS))

    output_width = int(game_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(game_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('game ', fps1)
    output_video = cv2.VideoWriter(
        'VideoOutput/video_with_map.mp4', fourcc, fps, (output_width, output_height))

    print('Adding the mini-map...')

    # Remove Outliers
    x, y = diff_xy(coords)
    remove_outliers(x, y, coords)
    # Interpolation
    coords = interpolation(coords)
    create_top_view(court_detector, detection_model, coords, fps)
    minimap_video = cv2.VideoCapture('VideoOutput/minimap.mp4')
    fps2 = int(minimap_video.get(cv2.CAP_PROP_FPS))
    print('minimap ', fps2)
    while True:
        ret, frame = game_video.read()
        ret2, img = minimap_video.read()
        if ret:
            output = merge(frame, img)
            output_video.write(output)
        else:
            break
    game_video.release()
    minimap_video.release()

output_video.release()

for _ in range(3):
    x, y = diff_xy(coords)
    remove_outliers(x, y, coords)

# interpolation
coords = interpolation(coords)

# velocty
Vx = []
Vy = []
V = []
frames = [*range(len(coords))]

for i in range(len(coords)-1):
    p1 = coords[i]
    p2 = coords[i+1]
    t1 = t[i]
    t2 = t[i+1]
    x = (p1[0]-p2[0])/(t1-t2)
    y = (p1[1]-p2[1])/(t1-t2)
    Vx.append(x)
    Vy.append(y)

for i in range(len(Vx)):
    vx = Vx[i]
    vy = Vy[i]
    v = (vx**2+vy**2)**0.5
    V.append(v)

xy = coords[:]

if bounce:
    # Predicting Bounces
    test_df = pd.DataFrame(
        {'x': [coord[0] for coord in xy[:-1]], 'y': [coord[1] for coord in xy[:-1]], 'V': V})

    # df.shift
    for i in range(20, 0, -1):
        test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
    for i in range(20, 0, -1):
        test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
    for i in range(20, 0, -1):
        test_df[f'lagV_{i}'] = test_df['V'].shift(i, fill_value=0)

    test_df.drop(['x', 'y', 'V'], 1, inplace=True)

    Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
                  'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
                  'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
                  'lagX_2', 'lagX_1']]
    Xs = from_2d_array_to_nested(Xs.to_numpy())

    Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
                  'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
                  'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
                  'lagY_3', 'lagY_2', 'lagY_1']]
    Ys = from_2d_array_to_nested(Ys.to_numpy())

    Vs = test_df[['lagV_20', 'lagV_19', 'lagV_18',
                  'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
                  'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
                  'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
    Vs = from_2d_array_to_nested(Vs.to_numpy())

    X = pd.concat([Xs, Ys, Vs], 1)

    # load the pre-trained classifier
    clf = load(open('clf.pkl', 'rb'))

    predcted = clf.predict(X)
    idx = list(np.where(predcted == 1)[0])
    idx = np.array(idx) - 10

    if minimap == 1:
        video = cv2.VideoCapture('VideoOutput/video_with_map.mp4')
    else:
        video = cv2.VideoCapture(output_video_path)

    output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    print(fps)
    print(length)

    output_video = cv2.VideoWriter(
        'VideoOutput/final_video.mp4', fourcc, fps, (output_width, output_height))
    i = 0
    while True:
        ret, frame = video.read()
        if ret:
            # if coords[i] is not None:
            if i in idx:
                center_coordinates = int(xy[i][0]), int(xy[i][1])
                radius = 3
                color = (255, 0, 0)
                thickness = -1
                cv2.circle(frame, center_coordinates, 10, color, thickness)
            i += 1
            output_video.write(frame)
        else:
            break

    video.release()
    output_video.release()
