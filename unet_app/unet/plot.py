import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def get_training_plot(metric='Loss', include_training=True):
    history_files = glob.glob('model/history/*')

    train_results = []
    val_results = []
    for hfile in history_files:
        if metric == 'Loss':
            train_results = np.concatenate([train_results,np.load(hfile, allow_pickle=True)[()]['loss']])
            val_results = np.concatenate([val_results,np.load(hfile, allow_pickle=True)[()]['val_loss']])
        if metric == 'Dice':
            train_results = np.concatenate([train_results,np.load(hfile, allow_pickle=True)[()]['dice_score']])
            val_results = np.concatenate([val_results,np.load(hfile, allow_pickle=True)[()]['val_dice_score']])

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.set_title(f'{metric} curves')
    ax.set_xlabel(f'Epoch')
    ax.set_ylabel(f'{metric}')

    # x = np.arange(len(results)) + 1
    # y = results

    ax.grid('on')

    if include_training:
        ax.plot(np.arange(len(train_results)) + 1, train_results, label='Training', color='#1f77b4')
    ax.plot(np.arange(len(val_results)) + 1, val_results, label='Validation', color='#ff7f0e')
    ax.legend()
    
    fig.tight_layout(pad=0.5)
    fig.set_dpi(300)
    
    canvas.draw()       # draw the canvas, cache the renderer

    plot_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return plot_image