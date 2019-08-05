import gym
import numpy as np
import tensorflow as tf
from cv2 import resize, INTER_NEAREST
from versterken.queue import Queue

class AtariEnvironment():

    def __init__(self, env, nframes=4, nrepeats=1):
        """
            `nframes`: number of *observed* frames per state
            `nrepeats`: number of *observed* frames to repeat each action

                # Example (nframes = 4, nrepeats = 4)

                - observations:  o1  --------------------------->  o2
                     - actions:  a1  ---> a1 ---> a1 ---> a1 --->  a2
                      - states: [o1, ---------------------------> [o2,
                                 o1,                               o1,
                                 o1,                               o1,
                                 o1]                               o1]
        """
        self.env = env
        self.nframes = nframes
        self.frame_queue = Queue(self.nframes)

    def reset(self):
        """Reset environment. The initial 'state' is the first frame repeated `nframes` times."""
        frame = self.env.reset()
        self.frame_queue = Queue(self.nframes)
        self.frame_queue.fill(preprocess(frame))
        return collect_frames(self.frame_queue)

    def step(self, action):
        """Perform an action. The new 'state' is the new frame appened to previous `nframes` - 1."""
        frame, reward, done, info = self.env.step(action)
        self.frame_queue.push(preprocess(frame))
        state = collect_frames(self.frame_queue)
        return state, reward, done, info

def collect_frames(queue):
    queue.fill()
    return np.stack(queue.queue, axis=-1)

def preprocess_tf(frame):
    frame_grayscale = tf.image.rgb_to_grayscale(frame)
    frame_cropped = tf.image.crop_to_bounding_box(frame_grayscale, 34, 0, 160, 160)
    frame_resized = tf.image.resize_images(frame_cropped, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.squeeze(frame_resized)

def preprocess(frame):
    """Preprocess frame *without* TensorFlow."""
    frame_grayscale = rgb_to_grayscale(frame)
    frame_cropped = crop_to_bounding_box(frame_grayscale, 34, 0, 160, 160)
    frame_resized = resize_image(frame_cropped, (84, 84))
    return frame_resized

def rgb_to_grayscale(frame, rgb_weights = [0.2989, 0.5870, 0.1140]):
    """Convert an RGB frame to grayscale.

        # Input
        - `frame`: [height, width, 3] uint with pixel values in [0, 255].

        # Returns
        - `frame_gs`: [height, width] unit8 with pixel values in [0, 255].
    """
    return np.uint8(np.dot(frame, rgb_weights))

def crop_to_bounding_box(frame, offset_height, offset_width, target_height, target_width):
    return frame[offset_height:offset_height + target_height, offset_width:offset_width + target_width]

def resize_image(frame, size):
    return resize(frame, size, interpolation = INTER_NEAREST)
