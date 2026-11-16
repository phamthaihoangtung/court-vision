# Upwork Project Catalog listing

**Title (max 75 chars)**
You will get volleyball player tracking & stats from your fixed-camera video

**Category:** Data Science & Analytics > Machine Learning / Computer Vision
**Search tags:** computer vision, sports analytics, player tracking, camera calibration, OpenCV

## Summary (shown under the title)

Send me footage from a fixed camera (tripod, balcony, ceiling mount, hovering drone) and get back
positions in metres for every player: distance covered, top and average speed, jumps with height
estimates, court heatmaps, front-row / back-row occupancy, plus an annotated video with a live
top-down minimap. No sensors, no wearables, no model training on your data.

The method is classical computer vision and 3D geometry around a pretrained person detector, so it
works on the first clip you send. Every number comes with an uncertainty band, and the pipeline is
validated on synthetic courts with known ground truth (results tables are in the public repo).

## Tiers

| | **Starter - Stats report** | **Standard - Automated pipeline** | **Advanced - Edge deployment** |
|---|---|---|---|
| Who it is for | A coach or player with one match | A club or venue that films every week | A venue with installed cameras |
| What you get | Stats report from your footage: annotated MP4 with minimap, per-player stat cards (PNG), CSV/JSON | The pipeline set up for your court and camera: one command or a web upload page that turns each new clip into the same outputs | The pipeline running on a device at your venue (Jetson / mini-PC with GPU) on your camera streams |
| Footage | Up to 20 min, one camera | Unlimited clips, one court and camera position | Up to 2 cameras / courts |
| Court setup | Done by me | Calibrated and locked for your hall, incl. overlapping basketball/badminton lines | Per-camera calibration, lens undistortion |
| Custom stats | - | 1 custom metric (e.g. serve zone, rotation check) | 3 custom metrics + export to your format |
| Source code | - | Yes | Yes |
| Docs & handover call | - | README + 30 min call | Runbook + 60 min call + 14 days support |
| Delivery | 3 days | 10 days | 21 days |
| Revisions | 1 | 2 | 3 |
| Suggested price | $150 | $900 | $2,800 |

Prices are a starting suggestion: set them against your hourly rate and what comparable CV
listings charge when you publish.

## Add-ons

- Extra 20 min of footage: +$60
- Second sport template (beach volleyball, padel, 5/7-a-side football): +$200
- Pose-based jump analysis (ankle keypoints instead of box cues): +$150
- 24 h rush (Starter only): +$80

## What I need from the buyer (requirements form)

1. A sample clip (30 s is enough to confirm feasibility before you order).
2. Sport and court type (indoor 18x9 m, beach 16x8 m, other).
3. Is the camera fixed for the whole recording? Roughly how high and where is it?
4. Which numbers matter most to you (distance, speed, jumps, heatmaps, something else)?
5. Standard/Advanced: where should it run (your PC, a server, an edge device) and how do clips arrive?

## FAQ

**Does the camera have to be perfectly still?**
It should not pan or zoom. Slow drift (a hovering drone, a shaky balcony mount) is tracked frame by frame.

**Our hall has basketball and badminton lines all over the court.**
The court finder is built for that: it searches line combinations for the one that matches volleyball
dimensions. Halls where every sport uses the same line colour are harder; send a frame and I will
tell you before you order.

**How accurate is it?**
On synthetic courts with known ground truth the court position error is around a centimetre and
distance errors are a few percent; the repo README has the full tables, including the cases where it
degrades. Jump height is the least precise number and depends on camera angle - a low, oblique
camera sees jumps well, a camera looking almost straight down cannot measure them, and the report
says so instead of guessing.

**Can it identify players by name or jersey number?**
Players get consistent labels within a clip (A1...A6, B1...B6). Mapping labels to names is a short
manual step; jersey-number reading is possible as custom work.

**Do you train a model on my footage?**
No. Nothing is trained, and your footage is only used to produce your deliverables.

**Ball tracking / touches / scoring?**
Not part of this listing. Ask me - it is a separate piece of work.
