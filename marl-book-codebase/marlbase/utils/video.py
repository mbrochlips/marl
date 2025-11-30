import imageio


class VideoRecorder:
    def __init__(self, fps=10):
        self.fps = fps
        self.frames = []

    def reset(self):
        self.frames = []

    def record_frame(self, env):
        #frame = env.unwrapped.render()
        frame = env.unwrapped.render(mode="rgb_array")
        #print(f"Frame type: {type(frame)}, dtype: {getattr(frame, 'dtype', None)}, shape: {getattr(frame, 'shape', None)}")

        # Convert boolean frames to uint8 for imageio compatibility
        # if hasattr(frame, 'dtype') and frame.dtype == bool:
        #     frame = frame.astype('uint8') * 255
        
        self.frames.append(frame)

    def save(self, filename):
        imageio.mimsave(f"{filename}", self.frames, fps=self.fps)
