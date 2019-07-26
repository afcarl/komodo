import numpy as np
import tensorflow as tf

def collect_frames(queue, nframes):
    queue.fill(nframes)
    return np.stack(queue.queue, axis=-1)

def rgb_to_grayscale(frame):
    frame_grayscale = tf.image.rgb_to_grayscale(frame)
    frame_cropped = tf.image.crop_to_bounding_box(frame_grayscale, 34, 0, 160, 160)
    frame_resized = tf.image.resize_images(frame_cropped, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.squeeze(frame_resized)
