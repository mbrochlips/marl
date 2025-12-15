import imageio


class VideoRecorder:
    def __init__(self, fps=10):
        self.fps = fps
        self.frames = []

    def reset(self):
        self.frames = []

    def record_frame(self, env):
        frame = env.unwrapped.render(mode="rgb_array")
        self.frames.append(frame)

    def save(self, filename):
        imageio.mimsave(filename, self.frames, fps=self.fps)
