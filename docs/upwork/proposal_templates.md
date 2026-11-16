# Proposal templates

Fill the [brackets]. Keep each under ~150 words: clients read the first two lines in the preview.

## 1. "Sports player tracking / analytics" jobs

Hi [name],

You want [their goal in their words: e.g. per-player distance and speed from your match recordings]
from [their camera setup]. I have built exactly this for volleyball: one fixed camera in, player
positions in metres out, with distance, speed, jumps and heatmaps - no wearables and no training
on your footage.

Demo (annotated video + live top-down minimap): [GIF/MP4 link]
Code and accuracy tables: [repo link]

For [their sport] I would reuse the court registration and tracking and add [the one thing specific
to their job]. The first thing I would check is [the real risk: camera angle / line visibility /
occlusion], because that decides the accuracy you can expect.

If you send a 30-second sample clip I will run it and show you the result before you commit to
anything.

Two questions: [is the camera fixed for the whole recording?] [which numbers matter most?]

[Your name]

## 2. "Camera calibration / top-down view" jobs

Hi [name],

You need [their goal: e.g. a bird's-eye view of the pitch / pixel-to-metre mapping] from [their
camera]. I do this from the markings already on the ground - no checkerboard and no site visit:
detect the lines, match them to the known dimensions, and fit the image-to-ground mapping; lens
distortion is estimated from the same straight lines.

Example: [link to minimap GIF] - the right panel is the top-down view computed from the left one.
Method and error tables on synthetic ground truth: [repo link]

For your case I would deliver [homography / calibration file + a script that maps any pixel to
metres + an overlay image so you can check it by eye]. Accuracy depends mostly on [how much of the
marked area is visible / camera height], which I can assess from a single frame.

Send one frame and the real-world dimensions of anything visible in it, and I will reply with the
overlay.

[Your name]
