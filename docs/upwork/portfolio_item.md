# Upwork portfolio item

**Title:** court-vision - volleyball player tracking in metres from one fixed camera (no training)

**Role:** Computer vision engineer (solo project)

**Skills / tags:** Computer Vision, OpenCV, Camera Calibration, Homography, Object Tracking, YOLO,
Kalman Filter, Python, Sports Analytics, Gradio

**Description**

A pipeline that turns amateur fixed-camera volleyball footage into per-player statistics: distance
covered, top and average speed, jumps with estimated height, court heatmaps and front-row /
back-row occupancy - each with an uncertainty band.

The problem: clubs and venues have a camera on a balcony or a drone, not a tracking system. They
have no labelled data, mixed court markings, and wide-angle lenses.

What I built:

- Court registration from painted lines alone. Line segments are detected, combinations of four
  lines are matched against the official court dimensions, and the best match is refined on every
  line pixel. It picks the volleyball court out of overlapping lines from other sports and keeps
  tracking through camera drift.
- Lens distortion estimated from the same straight lines (plumb-line method), no checkerboard.
- Player detection and tracking with pretrained YOLO + ByteTrack, team assignment from net side and
  jersey colour, positions projected to the court plane and smoothed with a Kalman filter in metres.
- Jump detection from geometry: a jump is a short ballistic bump of the foot point along the
  projected vertical; while a player is airborne the ground position is interpolated instead of
  projected, and height is estimated from the recovered camera.
- Uncertainty for every number via Monte Carlo over the homography estimate.
- A validation harness that renders synthetic courts with known camera, distortion, clutter lines,
  player trajectories and jumps, and reports the errors and where the method breaks.

Outputs: annotated side-by-side MP4 with a live top-down minimap, per-player stat cards, CSV/JSON,
and a Gradio page (upload clip -> results). Runs on CPU; GPU optional.

**Media to attach:** annotated MP4, `assets/demo.gif`, `assets/best_frame.png`,
`assets/card_summary.png`, one player card.

**Project link:** GitHub repo URL
