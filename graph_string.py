#!/usr/bin/env python3

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class GraphString:
    ''' A class for storing graphable strings. '''
    CHAR_LINES = {
        ' ': [np.array([[0,0], [1,0]])],
        '/': [
            np.array([[0,0], [0.7,1], [0.9,1], [1,0]]),
            np.array([[0,0], [0.2,0], [0.9,1], [1,0]]),
        ],
        '\\': [
            np.array([[0,0], [0.1,1], [0.3,1], [1,0]]),
            np.array([[0,0], [0.1,1], [0.8,0], [1,0]]),
        ],
        '!': [
            np.array([
				[0.00, 0.00],
				[0.22, 1.00],
				[0.55, 1.00],
				[0.6 , 0.00],
				[1.00, 0.00],
			]), np.array([
				[0.00, 0.00],
				[0.22, 0.99],
				[0.27, 0.3 ],
				[0.49, 0.3 ],
				[0.55, 0.99],
				[0.6 , 0.00],
				[1.00, 0.00],
			]), np.array([
				[0.00, 0.00],
				[0.26, 0.21],
				[0.51, 0.21],
				[0.59, 0.00],
				[1.00, 0.00],
			]), np.array([
				[0.00, 0.00],
				[0.26, 0.21],
				[0.3 , 0.00],
				[0.46, 0.00],
				[0.51, 0.21],
				[0.59, 0.00],
				[1.00, 0.00],
			]), 
        ],
        'A': [
            np.array([
                [0   , 0   ],
                [0.35, 1   ],
                [0.65, 1   ],
                [1   , 0   ],
            ]), np.array([
                [0   , 0   ],
                [0.13, 0.38],
                [0.4 , 0.38],
                [0.5 , 0.75],
                [0.6 , 0.38],
                [0.85, 0.38],
                [1   , 0   ],
            ]), np.array([
                [0   , 0   ],
                [0.13, 0.38],
                [0.85, 0.38],
                [1   , 0   ],
            ]), np.array([
                [0   , 0   ],
                [0.3 , 0   ],
                [0.33, 0.16],
                [0.66, 0.16],
                [0.7 , 0   ],
                [1   , 0   ],
            ]),
        ],
        'B': [
            np.array([
                [0   , 0   ],
                [0.02, 1   ],
                [0.65, 1   ],
                [0.87, 0.9 ],
                [0.92, 0.76],
                [0.95, 0.38],
                [1   , 0   ],
            ]), np.array([
                [0   , 0   ],
                [0.01, 0.56],
                [0.33, 0.6 ],
                [0.34, 0.8 ],
                [0.53, 0.8 ],
                [0.58, 0.76],
                [0.6 , 0.7 ],
                [0.73, 0.53],
                [0.82, 0.56],
                [0.9 , 0.62],
                [0.92, 0.74],
                [0.97, 0.3 ],
                [1   , 0   ],
            ]), np.array([
				[0   , 0   ],
				[0.01, 0.51],
				[0.33, 0.61],
				[0.52, 0.61],
				[0.58, 0.64],
				[0.61, 0.69],
				[0.73, 0.53],
				[0.81, 0.52],
				[0.90, 0.46],
				[0.96, 0.36],
				[0.97, 0.29],
				[1.00, 0   ],
            ]), np.array([
				[0   , 0   ],
				[0.01, 0.21],
				[0.33, 0.22],
				[0.34, 0.42],
				[0.53, 0.41],
				[0.62, 0.39],
				[0.65, 0.31],
				[0.97, 0.28],
				[1.00, 0   ],
            ]), np.array([
				[0   , 0   ],
				[0.01, 0.21],
				[0.33, 0.22],
				[0.55, 0.22],
				[0.61, 0.24],
				[0.64, 0.30],
				[0.97, 0.27],
				[1.00, 0   ],
            ]), np.array([
				[0   , 0   ],
				[0.56, 0   ],
				[0.77, 0.03],
				[0.84, 0.06],
				[0.90, 0.11],
				[0.95, 0.19],
				[0.97, 0.27],
				[1.00, 0   ],
			]), 
        ],
        # C
        'D': [
            np.array([
				[0.00, 0.00],
				[0.01, 0.99],
				[0.56, 0.99],
				[0.71, 0.95],
				[0.81, 0.88],
				[0.88, 0.80],
				[0.93, 0.67],
				[0.95, 0.51],
				[1.00, 0.01],
			]), np.array([
				[0.00, 0.00],
				[0.01, 0.17],
				[0.32, 0.23],
				[0.33, 0.76],
				[0.47, 0.76],
				[0.57, 0.72],
				[0.61, 0.64],
				[0.63, 0.55],
				[0.64, 0.48],
				[0.94, 0.42],
				[1.00, 0.01],
			]), np.array([
				[0.00, 0.00],
				[0.01, 0.20],
				[0.32, 0.23],
				[0.46, 0.24],
				[0.54, 0.26],
				[0.59, 0.31],
				[0.62, 0.39],
				[0.63, 0.47],
				[0.94, 0.47],
				[0.99, 0.01],
			]), np.array([
				[0.00, 0.00],
				[0.54, 0.01],
				[0.65, 0.04],
				[0.75, 0.08],
				[0.84, 0.14],
				[0.90, 0.25],
				[0.94, 0.35],
				[0.95, 0.48],
				[1.00, 0.00],
			]),
        ],
        # EFG
        'H': [
            np.array([
				[0.00, 0.00],
				[0.01, 0.99],
				[0.32, 0.99],
				[0.33, 0.65],
				[0.66, 0.65],
				[0.67, 0.99],
				[0.99, 0.99],
				[1.00, 0.00],
			]), np.array([
				[0.01, 0.01],
				[0.31, 0.01],
				[0.33, 0.41],
				[0.66, 0.41],
				[0.67, 0.01],
				[0.99, 0.00],
			]),
        ],
        'I': [
            np.array([
				[0.00, 0.00],
				[0.01, 0.67],
				[0.02, 0.99],
				[0.98, 1.00],
				[0.99, 0.67],
				[1.00, 0.00],
			]), np.array([
				[0.00, 0.00],
				[0.01, 0.67],
				[0.40, 0.68],
				[0.65, 0.68],
				[0.99, 0.68],
				[1.00, 0.01],
			]), np.array([
				[0.00, 0.00],
				[0.02, 0.33],
				[0.40, 0.33],
				[0.41, 0.68],
				[0.65, 0.68],
				[0.66, 0.33],
				[0.99, 0.33],
				[1.00, 0.01],
			]), np.array([
				[0.01, 0.00],
				[0.51, 0.01],
				[0.99, 0.01],
			]),
        ],
        # JK
        'L': [
            np.array([
				[-0.00, 0.00],
				[0.01, 0.99],
				[0.39, 0.98],
				[0.40, 0.25],
				[0.99, 0.25],
				[1.0 , 0.00],
			]), np.array([
				[0.00, 0.00],
				[0.51, 0.00],
				[0.99, 0.00],
			]),
        ],
        # MN
        'O': [
            np.array([
				[0.01, -0.00],
				[0.02, 0.48],
				[0.04, 0.63],
				[0.09, 0.77],
				[0.20, 0.90],
				[0.33, 0.96],
				[0.48, 0.99],
				[0.64, 0.98],
				[0.77, 0.92],
				[0.87, 0.84],
				[0.94, 0.73],
				[0.98, 0.55],
				[1.00, -0.00],
			]), np.array([
				[-0.00, -0.00],
				[0.03, 0.46],
				[0.31, 0.47],
				[0.32, 0.60],
				[0.35, 0.68],
				[0.41, 0.74],
				[0.49, 0.77],
				[0.59, 0.75],
				[0.65, 0.69],
				[0.68, 0.59],
				[0.69, 0.48],
				[0.98, 0.47],
				[1.00, -0.00],
			]), np.array([
				[0.00, -0.00],
				[0.03, 0.47],
				[0.31, 0.48],
				[0.32, 0.38],
				[0.34, 0.31],
				[0.40, 0.25],
				[0.48, 0.22],
				[0.56, 0.23],
				[0.63, 0.27],
				[0.67, 0.33],
				[0.69, 0.44],
				[0.70, 0.48],
				[0.97, 0.48],
				[1.00, -0.00],
			]), np.array([
				[0.00, -0.00],
				[0.03, 0.47],
				[0.03, 0.35],
				[0.07, 0.24],
				[0.14, 0.15],
				[0.24, 0.06],
				[0.37, 0.02],
				[0.51, 0.01],
				[0.66, 0.03],
				[0.77, 0.07],
				[0.86, 0.15],
				[0.92, 0.24],
				[0.96, 0.35],
				[0.98, 0.46],
				[1.00, -0.00],
			]), 
        ],
        'P': [
            np.array([
				[0.01, 0.01],
				[0.02, 0.99],
				[0.68, 0.99],
				[0.83, 0.94],
				[0.94, 0.85],
				[0.99, 0.72],
				[1.00, -0.00],
			]), np.array([
				[0.01, 0.01],
				[0.02, 0.54],
				[0.37, 0.57],
				[0.38, 0.79],
				[0.52, 0.79],
				[0.59, 0.77],
				[0.63, 0.73],
				[0.65, 0.68],
				[0.98, 0.67],
				[1.00, -0.00],
			]), np.array([
				[0.01, 0.02],
				[0.02, 0.54],
				[0.37, 0.57],
				[0.52, 0.57],
				[0.59, 0.60],
				[0.63, 0.63],
				[0.65, 0.68],
				[0.98, 0.67],
				[1.00, -0.00],
			]), np.array([
				[0.01, 0.02],
				[0.36, 0.02],
				[0.38, 0.38],
				[0.62, 0.38],
				[0.74, 0.40],
				[0.83, 0.43],
				[0.91, 0.49],
				[0.96, 0.57],
				[0.99, 0.64],
				[1.00, -0.00],
			]), 
        ],
        # Q
        'R': [
            np.array([
				[0.00, 0.00],
				[0.01, 1.00],
				[0.64, 0.99],
				[0.82, 0.92],
				[0.88, 0.85],
				[0.91, 0.72],
				[1.00, 0.00],
			]), np.array([
				[0.  , 0.00],
				[0.01, 0.57],
				[0.32, 0.60],
				[0.32, 0.79],
				[0.51, 0.79],
				[0.57, 0.77],
				[0.60, 0.70],
				[0.64, 0.44],
				[0.73, 0.46],
				[0.81, 0.51],
				[0.86, 0.57],
				[0.90, 0.64],
				[0.91, 0.70],
				[1.00, 0.01],
			]), np.array([
				[0.01, 0.00],
				[0.01, 0.56],
				[0.32, 0.60],
				[0.47, 0.60],
				[0.55, 0.61],
				[0.59, 0.64],
				[0.60, 0.69],
				[0.64, 0.45],
				[0.71, 0.42],
				[0.77, 0.39],
				[0.81, 0.34],
				[0.85, 0.26],
				[0.99, 0.00],
			]), np.array([
				[0.02, 0.01],
				[0.31, 0.01],
				[0.32, 0.41],
				[0.39, 0.40],
				[0.45, 0.35],
				[0.64, 0.01],
				[0.99, 0.01],
			]), 
        ],
        # S
        'T': [
            np.array([
				[0.00, 0.00],
				[0.01, 0.75],
				[0.02, 0.99],
				[0.97, 0.98],
				[0.98, 0.73],
				[1.00, -0.01],
			]), np.array([
				[0.00, 0.00],
				[0.02, 0.75],
				[0.33, 0.74],
				[0.34, 0.00],
				[0.65, -0.00],
				[0.66, 0.73],
				[0.97, 0.74],
				[1.00, -0.01],
			]), 
        ],
        # UVWX
        'Y': [
            np.array([
				[0.00, -0.00],
				[0.02, 1.01],
				[0.32, 1.01],
				[0.50, 0.68],
				[0.67, 1.00],
				[0.99, 1.00],
				[1.00, -0.00],
			]), np.array([
				[0.00, -0.00],
				[0.01, 1.01],
				[0.35, 0.42],
				[0.36, 0.00],
				[0.64, 0.00],
				[0.65, 0.41],
				[0.99, 1.00],
				[1.00, -0.00],
			]), 
        ],
        # Z
    }
    def __init__(self, string, char_width=1, char_height=1.5, char_spacing=0.1,
                 char_lines={}, resolution=1e-2):
        ''' Initialise a GraphString from a normal character string.

        'string' is the string of characters to be graphed.
        'char_lines' is a dictionary of {'char':line_points} to supplement the
            class CHAR_LINES, or to overwrite default characters with an
            alternative implementation.

        '''
        self.string       = string
        self.resolution   = resolution
        self.char_width   = char_width
        self.char_height  = char_height
        self.char_spacing = char_spacing
        self._char_lines  = {**self.CHAR_LINES, **char_lines}
        self._determine_num_lines()
        self._create_graph_lines()

    def _determine_num_lines(self):
        ''' Determine the number of lines required to display self.string. '''
        self._num_lines = 0
        for char in self.string:
            try:
                lines = self._char_lines[char]
            except KeyError as err:
                raise err(f'{char} not supported by default - specify line'
                          ' values as char_lines on initialisation.')
            num_lines = len(lines)
            if num_lines > self._num_lines:
                self._num_lines = num_lines

    def _create_graph_lines(self):
        ''' Create the set of lines describing the full stored string. '''
        self._graph_lines = [[[],[]] for line in range(self._num_lines)]
        offset = 0
        for char in self.string:
            self._add_char_lines(char, offset)
            offset += self.char_spacing + self.char_width

    def _add_char_lines(self, char, offset):
        ''' Adds the lines from char at the specified offset. '''
        char_lines = self._char_lines[char]
        num_lines  = len(char_lines)
        new_line   = [[]*2]
        for index, line in enumerate(self._graph_lines):
            if index < num_lines:
                # get line data from char_lines, scale and offset accordingly
                new_line = char_lines[index] * [self.char_width,
                                                self.char_height] + [offset, 0]
            # else: use line data from previous line (generally bottom line)

            for axis_index, axis in enumerate(line):
                axis.extend(new_line[:,axis_index])

    def plot(self, interpolation=None, resolution=None, **kwargs):
        ''' Display the string on a graph. '''
        self._plot_lines = []
        for index, line in enumerate(self._graph_lines):
            x,y = line
            if interpolation:
                x,y = self.interpolate(x, y, interpolation, resolution)

            self._plot_lines.append(plt.plot(x, y, **kwargs)[0])
        plt.axis('off')
        plt.axis('equal')
        plt.tight_layout()

    def interpolate(self, x, y, kind, resolution):
        ''' Interpolate the line data, return interpolated x and y. '''
        # interpolate linearly first, to avoid huge deviations from
        #   intended path
        linear_res = 5 * resolution
        f = interpolate.interp1d(x, y, kind='linear', fill_value=0,
                                 bounds_error=False)
        x = np.arange(x[0], x[-1]+linear_res, linear_res)
        y = f(x)
        # now apply desired interpolation
        f = interpolate.interp1d(x, y, kind=kind, fill_value='extrapolate')
        x = np.arange(x[0], x[-1]+resolution, resolution)
        y = f(x)
        return x, y

    def animate(self, display_chars='all', taper=None, filename=None, speed=10,
                interpolation='cubic', resolution=1e-2, **kwargs):
        ''' Create and play an animation.

        'display_chars' is the number of characters to show at a time. If left
            as 'all', displays all the characters simultaneously.
        'taper' specifies if the ends should be tapered to ensure a periodic
            boundary (useful for circular=True).
        'filename' specifies an optional file to save the animation to.

        '''
        fig = plt.figure('GraphString')
        self.plot(interpolation=interpolation, resolution=resolution)
        x = self._graph_lines[0][0]
        full_width = x[-1] - x[0]
        if display_chars == 'all':
            crop_width = full_width
        else:
            crop_width = display_chars * (self.char_width + self.char_spacing)\
                - self.char_spacing

        # double each line to animate more easily
        lines = []
        for index, line in enumerate(self._graph_lines):
            data = []
            x, y = line
            first = x[0]
            last = x[-1]
            x += [val+last-first for val in x]
            y += y
            x,y = self.interpolate(x,y,interpolation,resolution)
            data = np.array([x,y])
            lines.append(data)
            self._plot_lines[index].set_data(data[:, data[0] < crop_width])

        ax = fig.axes[0]
        ax.set_xlim(left=0, right=crop_width)
        plt.axis('equal')

        num_frames = int((full_width) // (speed * resolution))
        if taper:
            logistic = 1 / (1 + np.exp(-8/taper*(np.arange(taper)-taper/2)))
            reverse = logistic[::-1]

        def update(frame):
            start = (frame % (num_frames)) * speed * resolution
            end = start + crop_width
            x = lines[0][0]
            region = (start <= x) & (x <= end)
            for index, line in enumerate(self._plot_lines):
                x,y = lines[index]
                y_out = y[region]
                if taper:
                    y_out[:taper] *= logistic
                    y_out[-taper:] *= reverse
                line.set_data(x[region]-start, y_out)
            return self._plot_lines

        self.anim = FuncAnimation(fig, update, frames=num_frames, interval=30,
                                  blit=True)

        if filename:
            self.anim.save(filename, fps=33, extra_args=['-vcodec', 'libx264'])


if __name__ == '__main__':
    message = ' HAPPY BIRTHDAY OLI! '
    test = GraphString(' '*len(message) + message)
    test.animate(display_chars=21, taper=200, filename='HB.mp4')
    plt.show()


