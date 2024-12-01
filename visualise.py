import torch
import cv2 as cv
import numpy as np
import os
from PIL import Image
import argparse
from torchvision import transforms
import src.preprocessor as preprocessor
from src.networks import AlexNet
from src.ds_transforms import *
from random import randint


def run_model(input_img, net, processor, transform):
    preds = []
    face_images, coords = processor(input_img)
    
    for img in face_images:
        img = Image.fromarray(img) # transform expects PIL image
        img = transform(img)
        pred = net.predict(img)
        preds.append(pred)
        
    return face_images, coords, preds


def visualise_model(input_img, coords, g_preds, a_preds, confidence_scores): 
    # darken image to make writing more visible:
    # cite this
    input_img = cv.addWeighted(input_img, 0.75, np.zeros(input_img.shape, input_img.dtype), 0, 0)

    for idx, e in enumerate(coords):
        face_x = e['face_x'].astype(int)
        face_y = e['face_y'].astype(int)
        face_w = e['face_w'].astype(int)
        face_h = e['face_h'].astype(int)

        thickness = 1 if face_w/input_img.shape[1] < 0.25 else 2
        
        input_img = cv.rectangle(input_img,
                                    (face_x, face_y),
                                    (face_x + face_w, face_y + face_h),
                                    color=(255, 255, 255),
                                    thickness=thickness)

        this_gender = g_preds[idx]
        this_age = a_preds[idx]

        text = 'Female' if this_gender == 0 else 'Male'
        text += ' ~' + str(int(this_age))   # '~' --> 
                                            # indicate that the age is a ROUGH approximation

        cv.putText(
            input_img,
            text=text,
            org=(face_x+1, face_y-5),
            fontFace=0,
            fontScale=max(face_w/150, 0.2),
            color=(255, 255, 255),
            thickness=thickness)
    
    return input_img

def resize_to_max(image, max_size=720):
    if image.shape[0] > image.shape[1]:
        new_width = int(image.shape[1]/image.shape[0]*max_size)
        new_height = max_size
        image = cv.resize(image, (new_width, new_height))
    else:
        new_width = max_size
        new_height = int(image.shape[0]/image.shape[1]*max_size)
        image = cv.resize(image, (new_width, new_height))
    return image


def visualise_image(image_path, g_net, g_processor, g_transform, a_net,
                       a_processor, a_transform, resize=720,
                       show_processed_faces=False, confidence_scores=False):
    in_image = cv.imread(image_path)
    if resize:
        in_image = resize_to_max(in_image, resize)

    face_images, coords, g_preds = run_model(in_image, g_net, g_processor, g_transform)
    _, _, a_preds = run_model(in_image, a_net, a_processor, a_transform)
    out_image = visualise_model(in_image, coords, g_preds, a_preds, confidence_scores)

    if show_processed_faces:
        # paste each image from face_images into frame
        for idx, face in enumerate(face_images):
            # transform expects PIL image
            face = Image.fromarray(face)
            face = g_transform(face)
            face = face.squeeze(0).numpy()
            if len(face.shape) == 2:
                face = cv.cvtColor(face, cv.COLOR_GRAY2RGB)
            else:
                face = np.transpose(face, (1,2,0))
            face_min, face_max = face.min(), face.max()
            face = (face - face_min) / (face_max - face_min) * 255
            face = cv.resize(face, (100, 100)) # resize to avoid clutter
            # convert to rgb if grayscale
            out_image[idx*face.shape[0]:(idx+1)*face.shape[0], 0:face.shape[1]] = face[:,:]

    cv.imshow(image_path, out_image)
    cv.waitKey(0)


def visualise_cam(g_net, g_processor, g_transform, a_net, a_processor,
                  a_transform, resize=720, cam_id=0,
                  show_processed_faces=False, confidence_scores=False):
    cam_input = cv.VideoCapture(cam_id)
    _, frame = cam_input.read() 
    win_title = 'Live (Camera Input)'
    cv.imshow(win_title, frame)
    frame_diff_threshold = 130  # start with highest threshold,
                                # lower later when we have more info
    # frame_diff_history = [0] * 10 # keep track of last 10 frames for calcs
    face_images, coords, g_preds , a_preds = [], [], [], []
    last_frame = []
    
    while cv.waitKey(1) < 0:
        # main loop
        _, frame = cam_input.read()
        if resize:
            frame = resize_to_max(frame, resize)

        if len(last_frame) > 0:
            frame_diff = abs(np.sum(frame - last_frame))/frame.size
            # if frame_diff > 1:
            #     # sometimes frame is copy of last due to lag etc.
            #     # so ignore that
            #     frame_diff_history.pop(0)
            #     frame_diff_history.append(frame_diff)
        else:
            frame_diff = 0
        last_frame = frame

        # frame_diff_avg = sum(frame_diff_history)/10 # avg movement from last 10 frames
        # frame_diff_threshold = frame_diff_avg * 0.2

        if frame_diff > frame_diff_threshold:
            face_images, coords, g_preds , a_preds = [], [], [], []
        else:
            if len(face_images) <= 0:   # only fetch predictions if don't already have
                                        # i.e. don't keep predicting for same person unnecessarily
                face_images, coords, g_preds = run_model(frame, g_net, g_processor, g_transform)
                _, _, a_preds = run_model(frame, a_net, a_processor, a_transform)

        out_image = visualise_model(frame, coords, g_preds, a_preds, confidence_scores)

        if show_processed_faces:
            # paste each image from face_images into frame
            for idx, face in enumerate(face_images):
                # transform expects PIL image
                face = Image.fromarray(face)
                face = g_transform(face)
                face = face.squeeze(0).numpy()
                if len(face.shape) == 2:
                    # convert to rgb if grayscale
                    face = cv.cvtColor(face, cv.COLOR_GRAY2RGB)
                else:
                    face = np.transpose(face, (1,2,0))
                face_min, face_max = face.min(), face.max()
                face = (face - face_min) / (face_max - face_min) * 255
                out_image[idx*face.shape[0]:(idx+1)*face.shape[0], 0:face.shape[1]] = face[:,:]
                
        cv.imshow(win_title, out_image)


if __name__ == '__main__':
    # parse command line arguments
    # https://docs.python.org/3/tutorial/stdlib.html#command-line-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str)
    parser.add_argument('--show-processed', action='store_true', default=False)
    parser.add_argument('--confidence-scores', action='store_true', default=False)
    parser.add_argument('--frame-diff-threshold', type=float, default=255.0)
    args = parser.parse_args()

    # load models
    # todo: make this accessible via cmd
    g_path = './models/AlexNet-2_genderEq_83.pt'
    g_processor = preprocessor.process(crop='mid')
    g_transform = alexnet_transform(size=224)

    a_path = './models/AlexNet-1_ageNoEq_7.pt'
    a_processor = preprocessor.process(crop='mid')
    a_transform = alexnet_transform(size=224)

    # infer model architecture from path
    # e.g. 'LeNet-2_xyz.pt' -> 'LeNet(2)'
    g_net = None
    g_architecture = os.path.basename(g_path).split('_')[0].replace('-', '(') + ')'
    exec('g_net = ' + g_architecture)
    g_net.load_state_dict(torch.load(g_path,
        map_location=torch.device('cpu'))) # ...
    g_net.eval()

    a_net = None
    a_architecture = os.path.basename(a_path).split('_')[0].replace('-', '(') + ')'
    exec('a_net = ' + a_architecture)
    a_net.load_state_dict(torch.load(a_path,
        map_location=torch.device('cpu')))
    a_net.eval()

    if args.image_path:
        visualise_image(args.image_path,
                        g_net, g_processor, g_transform,
                        a_net, a_processor, a_transform,
                        show_processed_faces=args.show_processed,
                        confidence_scores=args.confidence_scores)
    else:
        visualise_cam(g_net, g_processor, g_transform,
                      a_net, a_processor, a_transform,
                      show_processed_faces=args.show_processed,
                      confidence_scores=args.confidence_scores)