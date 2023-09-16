"""Signal To Image Visualizations.

Contains code for visualizations of the signal to image transformations.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
    GitHub Copilot.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyts.image import (GramianAngularField, MarkovTransitionField,
                        RecurrencePlot)

from matplotlib.animation import FuncAnimation

X_LIM = 2 * np.pi


def generate_sine_wave(frames: int,
                       time_delay: float,
                       samples: int) -> np.ndarray:
    """Produces a sine wave.

    Produces a sine wave of amplitude +- 1 at a wavelength of 2 pi..
    When played as an animation, the wave will look like it is moving
    to the left over time.

    Returns:
        Array with shape [frames, num_samples, 2] where 2 is the x and y vals.
    """
    wave = np.zeros((frames, samples, 2))
    for i in range(frames):
        start = i * time_delay
        x = np.linspace(start, X_LIM + start, samples)
        wave[i, :, 0] = np.linspace(0, X_LIM, samples)
        wave[i, :, 1] = np.sin(x)

    return wave


def gaf_polar_transform(array: np.ndarray) -> np.ndarray:
    """Performs the polar transformation."""
    polar = np.zeros_like(array)
    # r values are t_i / N
    polar[:, :, 1] = array[:, :, 0] / array.shape[1]
    # phi values are arccos(x)
    polar[:, :, 0] = np.arccos(array[:, :, 1])

    return polar


def make_anim(fig, array, animate, save_name):
    """Renders the animation."""
    save_fp = str(Path(__file__).parent.parent / "figures" / f'{save_name}.mp4')

    anim = FuncAnimation(fig, animate, frames=array.shape[0], interval=20,
                         blit=True)

    anim.save(save_fp, fps=30, extra_args=["-vcodec", "libx264"])


def plot_animation_cartesian(array: np.ndarray, title: str,
                             save_name: str):
    """Plots the array as an animation where dim 0 is the time dimension."""
    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.set_xlim(0, X_LIM)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(title)
        return ax.plot(array[i, :, 0], array[i, :, 1])

    make_anim(fig, array, animate, save_name)
    plt.clf()


def plot_gaf_polar(array: np.ndarray, title: str, save_name: str):
    """Creates polar plot for gaf visualization."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    def animate(i):
        ax.clear()
        ax.set_rmax(1)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.grid(True)
        ax.set_title(title)
        return ax.plot(array[i, :, 0], array[i, :, 1])

    make_anim(fig, array, animate, save_name)


def plot_image(array: np.ndarray, title: str, save_name: str):
    """Creates the image plots."""

    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.set_title(title)
        return [ax.imshow(array[i], cmap='viridis', origin='lower')]

    make_anim(fig, array, animate, save_name)


def main():
    # Generate sine wave
    print("Generating sine wave...")
    wave = generate_sine_wave(120, 0.1, 15)

    print("Calculating polar transform...")
    polar = gaf_polar_transform(wave)

    print("Calculating GAF transform...")
    gaf = GramianAngularField()
    gaf_plot = gaf.transform(wave[:, :, 1])

    print("Calculating MTF transform...")
    mtf = MarkovTransitionField(n_bins=4)
    mtf_plot = mtf.transform(wave[:, :, 1])

    print("Calculating RP transform...")
    rp = RecurrencePlot()
    rp_plot = rp.transform(wave[:, :, 1])

    # Now animating
    print("Plotting cartesian animation...")
    plot_animation_cartesian(wave, 'Sine Wave', 'sine_wave')

    print("Plotting polar animation...")
    plot_gaf_polar(polar, "Polar Transformation",
                   "gaf_transformed_polar.mp4")

    print("Plotting GAF animation...")
    plot_image(gaf_plot, "Gramian Angular Field",
               "gaf_transformed.mp4")

    print("Plotting MTF animation")
    plot_image(mtf_plot, "Markov Transition Field",
               "mtf_transformed.mp4")

    print("Plotting RP animation")
    plot_image(rp_plot, "Recurrence Plot",
               "rp_transformed.mp4")


if __name__ == '__main__':
    main()
