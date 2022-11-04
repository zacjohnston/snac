# -------------------------------
#   ____  _   _    _    ____ 
#  / ___|| \ | |  / \  / ___|
#  \___ \|  \| | / _ \| |    
#   ___) | |\  |/ ___ \ |___ 
#  |____/|_| \_/_/   \_\____|
# -------------------------------
"""
Main SNAC class for the Simulation object.

A Simulation instance represents a single set of SNEC run.
It can load model data files, manipulate/extract that data,
and plot.

Info
-------------------
    Setup arguments
    ---------------
    model: Name of the SNEC set (i.e. the directory name).

    Data structures
    ---------------
    dat: Integrated time-series quantities found in the `.dat` files.
    profile: Lagrangian profiles as extracted from `.xg` files.
    solo_profile: Lagrangian profile taken at a particular day
    scalars: Scalar quantities, e.g., time of shock breakout. Dictionary.

"""

import os
import time
import numpy as np
# import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# snac
from . import load
from . import paths
from . import plot_tools
from . import tools


class Simulation:
    """
    Main class containing data for one simulation.
    """

    def __init__(self, model, config='snec',
                 output_dir='Data', verbose=True, load_all=True,
                 reload=False, save=True, load_profiles=True):
        """
        Object representing a SNEC simulation

        parameters
        ----------
        model : str
            The name of the main model directory
        config : str
            Base name of config file to use, e.g. 'snec' for 'config/snec.ini'
        output_dir : str
            name of subdirectory containing model output files
        load_all : bool
            load all model data (profiles, dat)
        reload : bool
            load from raw data, not saved pickle files (slow)
        save : bool
            save extracted data to pickle files (for faster loading)
        verbose : bool
            print information to terminal
        load_profiles : bool
            do, or do not, load mass profiles
        """
        t0 = time.time()
        self.verbose = verbose
        self.model = model

        self.model_path = paths.model_path(model=model)
        self.output_path = os.path.join(self.model_path, output_dir)

        self.config = None  # model-specific configuration; see load_config(). Dict
        self.dat = None  # integrated data from .dat; see load_dat()
        self.profiles = None  # mass profile data for each timestep
        self.solo_profile = None  # profile at one timestep
        self.scalars = None  # scalar quantities: time of shock breakout..
        self.vfe = None  # Holds v_Fe(t)
        self.tau = None  # Hold tau_sob

        self.load_config(config=config)

        if load_all:
            self.load_all(reload=reload, save=save, load_profiles=load_profiles)

        t1 = time.time()
        self.printv(f'Model load time: {t1 - t0:.3f} s')

    # =======================================================
    #                      Setup/init
    # =======================================================
    def printv(self, string, verbose=None, **kwargs):
        """
        Verbose-aware print
        parameters
        ----------
        string : str
            string to print if verbose=True
        verbose : bool
            override self.verbose setting
        **kwargs
            args for print()
        """
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print(string, **kwargs)

    def load_config(self, config='snec'):
        """
        Load config parameters from file
        parameters
        ----------
        config : str
        """
        self.config = load.load_config(name=config, verbose=self.verbose)

    def load_all(self, reload=False, save=True, load_profiles=False):
        """
        Load all model data
        parameters
        ----------
        reload : bool
        save : bool
        load_profiles :bool
        """
        self.load_dat(reload=reload, save=save)
        self.get_scalars()

        if load_profiles:
            self.load_all_profiles(reload=reload, save=save)
            self.get_profile_day()

    # =======================================================
    #                   Loading Data
    # =======================================================
    def load_dat(self, reload=False, save=True):
        """
        Load .dat file
        parameters
        ----------
        reload : bool
        save : bool
        """
        self.dat = load.get_dat(
            model=self.model,
            cols=self.config['dat_quantities']['fields'], reload=reload,
            save=save, verbose=self.verbose)

    def load_all_profiles(self, reload=False, save=True):
        """
            Load profiles

            parameters
            ----------
            reload : bool
            save : bool
            """
        config = self.config['profiles']

        self.profiles = load.get_profiles(
            model=self.model,
            fields=config['fields'],
            reload=reload, save=save, verbose=self.verbose)

    def get_scalars(self):
        """
        Compute all necessary SNEC scalar quantities
        """
        self.scalars = load.get_scalars(model=self.model)

    # =======================================================
    #                   Manipulation
    # =======================================================

    def clear_vfe(self):
        """
        Clear the self.vfe array.
        """
        if self.vfe is not None:
            self.vfe = None
        else:
            print("vfe is already empty.")

    def get_dat_day(self, day=50.0):
        """
        Isolate dat quantities at a specific day post shock breakout, 50 by default

        Parameters:
        day : float
        """
        t = self.dat['time']
        if day >= 0.0:
            ind = np.max(np.where(t <= day * 86400.))
        else:
            ind = 0

        for col in self.dat:
            if col == 'time':
                continue

            label = col + '_solo'
            self.scalars[label] = self.dat[col][ind]

    def get_profile_day(self, day=0.0):
        """
        Isolate mass profiles at a specific day, move from dict to DataFrame.

        Parameters:
        -----------
        day : float
            0 indicates shock breakout. Pass -1 for initial profile.
        post_breakout : bool
            if True, day is assumed to be with respend to shock breakout.
        """

        cols = self.config['profiles']['fields']

        times = np.array([*self.profiles['rho']])  # returns dictionary keys - the times.
        # This isolates the key for the data just before [day] days.

        if day == 0.0:  # If day = 0, add some padding so we're just through shock breakout.
            t = times[np.max(np.where(times <= day * 86400.)) + 1]
        elif day == -1:
            t = times[0]
        else:
            t = times[np.max(np.where(times <= day * 86400.)) + 0]

        df = pd.DataFrame()
        # All profiles in the dict contain mass as first column. Just grab from one
        df['mass'] = self.profiles['rho'][t][:, 0]
        for col in cols:
            df[col] = self.profiles[col][t][:, 1]

        df.time = t / 86400  # Actual timestamp, post shock breakout.
        df.day = day  # time looking for.

        self.solo_profile = df

    # =======================================================
    #                      Plotting
    # =======================================================
    def plot_profiles(self, t, y_var_list, x_var='mass', y_scale=None, x_scale=None,
                      max_cols=2, sub_figsize=(6, 5), legend=False):
        """Plot one or more profile variables
        parameters
        ----------
        t : int
            timestamp to plot
        y_var_list : str or [str]
            variable(s) to plot on y-axis (from Simulation.profile)
        x_var : str
            variable to plot on x-axis
        y_scale : {'log', 'linear'}
        x_scale : {'log', 'linear'}
        legend : bool
        max_cols : bool
        sub_figsize : tuple
        """
        y_var_list = tools.ensure_sequence(y_var_list)
        n_var = len(y_var_list)
        fig, ax = plot_tools.setup_subplots(n_var, max_cols=max_cols,
                                            sub_figsize=sub_figsize, squeeze=False)

        for i, y_var in enumerate(y_var_list):
            row = int(np.floor(i / max_cols))
            col = i % max_cols

            self.plot_profile(t, y_var=y_var, x_var=x_var, y_scale=y_scale,
                              x_scale=x_scale, ax=ax[row, col],
                              legend=legend if i == 0 else False)
        return fig

    def plot_profile(self, t, y_var, x_var='mass', y_scale=None, x_scale=None,
                     ax=None, legend=False, title=True,
                     figsize=(8, 6), label=None,
                     linestyle='-', marker=''):
        """Plot given profile variable
        parameters
        ----------
        t : float
            time
        y_var : str
            variable to plot on y-axis (from Simulation.profile)
        x_var : str
            variable to plot on x-axis
        y_scale : {'log', 'linear'}
        x_scale : {'log', 'linear'}
        ax : Axes
        legend : bool
        title : bool
        figsize : [width, height]
        label : str
        linestyle : str
        marker : str
        """
        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        self._set_ax_title(ax, t=t, title=title)
        self._set_ax_scales(ax, y_var, x_var=x_var, y_scale=y_scale, x_scale=x_scale)
        self._set_ax_labels(ax, x_var=x_var, y_var=y_var)

        x = self.profiles[y_var][t][:, 0]
        y = self.profiles[y_var][t][:, 1]

        ax.plot(x, y, ls=linestyle, marker=marker, label=label)

        if legend:
            ax.legend()

        return fig

    def plot_slider(self, y_var, x_var='mass', y_scale=None, x_scale=None,
                    figsize=(8, 6), title=True, legend=True,
                    linestyle='-', marker=''):

        """
        Plot interactive slider of profile for given variable

        parameters
        ----------
        y_var : str
        x_var : str
        y_scale : {'log', 'linear'}
        x_scale : {'log', 'linear'}
        figsize : [width, height]
        title : bool
        legend : bool
        linestyle : str
        marker : str
        """
        fig, profile_ax, slider_ax = self._setup_slider_fig(figsize=figsize)
        t_max, t_min = self._get_slider_bounds()

        slider = Slider(slider_ax, 'time', t_min, t_max, valinit=t_max, valstep=1)

        self.plot_profile(t_max, y_var=y_var, x_var=x_var, y_scale=y_scale,
                          x_scale=x_scale, ax=profile_ax, legend=legend,
                          title=title, figsize=figsize,
                          linestyle=linestyle, marker=marker)

        def update(t):
            self.get_profile_day(day=t / 86400)
            profile = self.solo_profile
            y_profile = profile[y_var]

            profile_ax.lines[0].set_ydata(y_profile)
            profile_ax.lines[0].set_xdata(profile[x_var])
            self._set_ax_title(profile_ax, t=t, title=title)

            fig.canvas.draw_idle()

        slider.on_changed(update)
        return fig, slider

    def plot_dat(self, y_var, y_scale='log', display=True, ax=None, figsize=(8, 6),
                 linestyle='-', marker=''):
        """Plot quantity from dat file
        parameters
        ----------
        y_var : str
        y_scale : {'log', 'linear'}
        figsize : [width, height]
        display : bool
        ax : Axes
        linestyle : str
        marker : str
        """
        fig, ax = self._setup_fig_ax(ax=ax, figsize=figsize)
        ax.plot(self.dat['time'], self.dat[y_var], linestyle=linestyle, marker=marker)

        ax.set_yscale(y_scale)
        self._set_ax_labels(ax, x_var='$t$ (s)', y_var=y_var)

        if display:
            plt.show(block=False)

        return fig, ax

    # =======================================================
    #                      Plotting Tools
    # =======================================================
    def get_label(self, key):
        """Return formatted string for plot label
        parameters
        ----------
        key : str
            parameter key, e.g. 'r', 'temp', 'dens'
        """
        return self.config['plotting']['labels'].get(key, key)

    def _set_ax_scales(self, ax, y_var, x_var, y_scale, x_scale):
        """Set axis scales (linear, log)
        parameters
        ----------
        ax : Axes
        y_var : str
        x_var : str
        y_scale : {'log', 'linear'}
        x_scale : {'log', 'linear'}
        """
        if x_scale is None:
            x_scale = self.config['plotting']['ax_scales'].get(x_var, 'log')
        if y_scale is None:
            y_scale = self.config['plotting']['ax_scales'].get(y_var, 'log')

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

    def _set_ax_title(self, ax, t, title):
        """Set axis title
        parameters
        ----------
        ax : Axes
        title : bool
        """
        if title:
            ax.set_title(f't = {t:.3f} s')

    def _set_ax_labels(self, ax, x_var, y_var):
        """Set axis labels
        parameters
        ----------
        ax : Axes
        x_var : str
        y_var : str
        """
        ax.set_xlabel(self.get_label(x_var))
        ax.set_ylabel(self.get_label(y_var))

    def _set_ax_legend(self, ax, legend, loc=None):
        """Set axis labels
        parameters
        ----------
        ax : Axes
        legend : bool
        """
        if legend:
            ax.legend(loc=loc)

    def _setup_fig_ax(self, ax, figsize):
        """Setup fig, ax, checking if ax already provided
        parameters
        ----------
        ax : Axes
        figsize : [width, height]
        """
        fig = None

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        return fig, ax

    def _setup_slider_fig(self, figsize):
        """Setup fig, ax for slider
        parameters
        ----------
        figsize : [width, height]
        """
        fig = plt.figure(figsize=figsize)
        profile_ax = fig.add_axes([0.1, 0.2, 0.8, 0.65])
        slider_ax = fig.add_axes([0.1, 0.05, 0.8, 0.05])

        return fig, profile_ax, slider_ax

    def _get_slider_bounds(self):
        """
        Return t_max, t_min
        """
        t_max = [*self.profiles['rho']][-1]
        t_min = 0.0
        return t_max, t_min
