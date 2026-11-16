- **Court registration** finds the court in every baseline scene with about half a centimetre of
  median error, including with basketball and badminton lines in other colours, heavy image noise
  and the 4-line beach court. The weak case is a hall where every sport's lines are white: 62%.
- **Distance and speed** are within about 1% and 0.1 m/s. Very abrupt movement (8 m/s^2) raises
  the distance error to 3.7% and the 90% band then contains the truth only 22% of the time,
  because the smoother cuts corners.
- **Lens**: uncorrected barrel distortion of -0.10 costs about 11 cm of court error; the
  plumb-line correction brings it back to about 1 cm.
- **Jumps** are the weakest output. With the conservative default thresholds, precision is 86%
  and recall 54% at baseline (height error about 6 cm), 99% / 79% from a low camera, and 50% / 18%
  from a steep 65-degree camera. Missed jumps are mostly the small ones (20-30 cm).
- **Known defect on real footage**: jump counts on the near-overhead clip (`topdown`, 18 jumps up
  to 76 cm) and for one player in the `diagonal` clip (7 jumps) are not credible. From overhead the
  focal length cannot be recovered from the court plane, so the observability gate is computed from
  a wrong camera. Treat jump output as experimental unless the camera is low and oblique.
