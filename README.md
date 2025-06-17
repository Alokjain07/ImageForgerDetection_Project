# ImageForgerDetection_Project
This project implements a comprehensive image forgery detection system in MATLAB that compares an original image with a potentially forged version to detect tampering. It integrates multiple analysis techniques to improve detection accuracy.
key Features
Pixel Difference Analysis – Detects tampered regions by comparing pixel-wise intensity changes.
Error Level Analysis (ELA) – Highlights recompressed areas in JPEG images that may indicate forgery.
Noise Variance Analysis – Uses local standard deviation to spot noise inconsistencies from edits.
Copy-Move Forgery Detection – Uses VLFeat (SIFT + RANSAC) to identify duplicated regions within the forged image.
