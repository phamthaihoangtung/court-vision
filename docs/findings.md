- **Court registration** finds the court in every baseline scene with about half a centimetre of
  median error. That holds with basketball and badminton lines in other colours, with heavy image
  noise, and on the 4-line beach court. The weak case is a hall where every sport's lines are
  white.
- **Distance and speed** are within about 1% and 0.1 m/s. Very abrupt movement (8 m/s^2) raises
  the distance error to about 4%, and the 90% band then rarely contains the truth, because the
  smoother cuts corners. The bands assume normal movement.
- **Lens**: uncorrected barrel distortion of -0.10 costs about 11 cm of court error; the
  plumb-line correction brings it back to about 1 cm.
- **Jumps** are the weakest output. Precision is high and recall is moderate: small jumps
  (20-30 cm) are the ones missed. A low, oblique camera sees jumps well. A steep camera (65
  degrees) cannot separate a jump from a quick step, and most jumps there are dropped rather than
  guessed.
- These synthetic detections have clean boxes. Real boxes also change with posture and occlusion,
  which is why jump precision on real footage is lower (see Limitations).
