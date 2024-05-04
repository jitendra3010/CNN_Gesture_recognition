This dataset has 5 gestures having 30 frames in sequence for each sample.

Each gesture corresponds to the following specific command:

Thumbs up: Increase the volume
Thumbs down: Decrease the volume
Left swipe: 'Jump' backward 10 seconds
Right swipe: 'Jump' forward 10 seconds
Stop: Pause the movie

The data contains a 'train' and a 'val' folder with two CSV files for the two folders. 

These folders are in turn divided into subfolders where each subfolder represents a video of a particular gesture. Each subfolder, i.e. a video, contains 30 frames (or images). 

Note that all images in a particular video subfolder have the same dimensions but different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 (depending on the webcam used to record the videos).
Each row of the CSV file represents one video and contains three main pieces of information - the name of the subfolder containing the 30 images of the video sample (gesture), the name of the gesture, and the numeric label (between 0-4) of the video.
