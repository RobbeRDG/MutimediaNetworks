import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # Python Imaging Library https://pypi.org/project/Pillow/


class ImageSource:

    def __init__(self):
        self.pixel_seq = None
        self.img_path = None
        self.img = None
        self.mode = None
        self.num_of_channels = None
        self.width = None
        self.height = None
        self.bitmap = None
        self.channels = None

    def load_from_file(self, img_path):
        """Loads the image from a file via PIL

        Arguments:
            img_path {str} -- Absolute path to the image file

        Returns:
            self -- Returns self to do the following: ImageSource().load_from_file()
        """

        self.img_path = img_path
        img = Image.open(self.img_path)
        self.img = img
        self.mode = img.mode
        self.num_of_channels = len(img.getbands())
        self.width = img.width
        self.height = img.height

        # "bitmap" is a height x width x num_of_channels numpy array
        self.bitmap = np.array(img).astype('uint8')
        if self.num_of_channels > 1:
            self.channels = [self.bitmap[:, :, i]
                             for i in range(self.num_of_channels)]
        else:
            self.channels = [self.bitmap]
        return self

    def get_bitmap(self):
        return self.bitmap

    def get_pixel_seq(self):
        if self.pixel_seq is None:
            self.pixel_seq = self.bitmap.ravel()
        return self.pixel_seq

    def to_bitmap(self, pixel_seq=None):
        if pixel_seq is None:
            pixel_seq = self.get_pixel_seq()
        return pixel_seq.reshape((self.height, self.width, self.num_of_channels))

    def show(self):
        self.img.show()

    def show_color_hist(self):
        colors = [(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5)]
        for i, channel in enumerate(self.channels):
            plt.hist(channel.ravel(), bins=range(255), fc=colors[i])
        plt.show()
    
    def _clear(self):
        self.bitmap = None
        self.img = None
        self.pixel_seq = None
        self.img_path = None

    def from_bitmap(self, bitmap):
        self._clear()
        self.pixel_seq = bitmap.ravel()
        self.bitmap = self.pixel_seq.reshape((self.height, self.width, self.num_of_channels))
        self.img = Image.fromarray(self.bitmap, self.mode)
        self.img_path = "bitmap"

    def __str__(self):
        return f"""
        Location: {self.img_path}
        Mode: {self.mode}
        Size: {self.width} x {self.height}
        """
