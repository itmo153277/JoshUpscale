#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate batches for dataset."""

import argparse
import csv
import numpy as np
import cv2

# pylint: disable=no-member

parser = argparse.ArgumentParser(
    description="Generate batches for dataset")
parser.add_argument("video",
                    help="Video file",
                    type=str)
parser.add_argument("scenes",
                    help="Scenes file",
                    type=str)
args = parser.parse_args()
video_file = args.video
scenes = args.scenes
cap = cv2.VideoCapture(video_file)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
start_frames = []
with open(scenes, "rt") as f:
    scenes = csv.reader(f)
    for i, row in enumerate(scenes):
        if i > 1:
            start_frames.append(int(row[1]))
start_frames = np.array(start_frames)
r_starts = []
for _ in range(3000):
    r_start = None
    while True:
        r_start = int(np.random.uniform(frames - 10))
        scene_end = np.where(start_frames > r_start)[0]
        if not scene_end.any():
            break
        if start_frames[scene_end[0]] >= r_start + 10:
            break
    r_starts.append(r_start)
print("vid = last")
print("\nlast + ".join(["vid.trim(%d, -10)" % x for x in r_starts]))
