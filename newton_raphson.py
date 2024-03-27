import numpy as np
import matplotlib.pyplot as plt
import porespy as ps
import cmath
from matplotlib.widgets import RectangleSelector
from prettytable import PrettyTable


class NewtonsFractal:
    """
    Newton's Fractal class. Based on https://scipython.com/book2/chapter-8-scipy/examples/the-newton-fractal/
    """
    def __init__(self, f, f_prime, region, tolerance=1e-8):
        self.f = f
        self.f_prime = f_prime
        self.region = region

        self.newton_fractal_window, self.nfw_axes = plt.subplots()
        self.box_counting_gradient_window, self.bcgw_axes = plt.subplots(1, 1, figsize=(7, 4))
        self.newton_iterations_window, self.niw_axes = plt.subplots()
        self.niw_cax = self.niw_axes.inset_axes([1.03, 0, 0.1, 1])    # Holds the colour bar

        self.num_values = 500
        self.max_iterations = 1000
        self.tolerance = tolerance

        self.nfw_axes.set_title("Press 'r' to reset the zoom")
        self.nfw_axes.set_xlabel("Re", fontstyle='italic')
        self.nfw_axes.set_ylabel("Im", fontstyle='italic')
        self.niw_axes.set_xlabel("Re", fontstyle='italic')
        self.niw_axes.set_ylabel("Im", fontstyle='italic')

        self.newton_fractal_window.canvas.mpl_connect('key_press_event', self.reset_display)
        self.selector = RectangleSelector(
            self.nfw_axes, self._select_callback, useblit=True, button=[1, 3], spancoords='pixels', interactive=True
        )

    def newton_raphson(self, z0, f, f_prime, max_iterations):
        """Apply Newton-Raphson method to f(z)"""
        guess = z0
        for iteration in range(max_iterations):
            delta_z = f(guess) / f_prime(guess)
            if abs(delta_z) < self.tolerance:
                return guess, iteration
            guess = guess - delta_z
        return None, None

    def _get_region_by_size(self, size):
        """Returns a region based on the size specified"""
        return -size, size, -size, size

    def _select_callback(self, event_mouse_click, event_mouse_release):
        """
        This method is called when the mouse is used to select an area on the fractal. The mouse coordinates
        are recorded, and the values are used to set the region of the fractal to zoom in on

        We then call the _plot_newton_fractal method with the new region to re-draw the area selected with the mouse
        """
        x_min, y_min = event_mouse_click.xdata, event_mouse_click.ydata
        x_max, y_max = event_mouse_release.xdata, event_mouse_release.ydata
        zoom_region = (x_min, x_max, y_min, y_max)

        print(f"Zooming to {zoom_region}")
        self._plot_newton_fractal(
            self.f, self.f_prime, region=zoom_region, num_values=self.num_values, max_iterations=self.max_iterations,
        )

    def reset_display(self, event):
        """
        This method is called when any key is pressed on the fractal. If the 'r' key was pressed, we call
        plot_newton_fractal, which in turn calls the _plot_newton_fractal method with the original region set
        when the program started running. This effectively 'resets' the zoom of the display back to the initial
        state
        """
        if event.key == 'r':
            print("Resetting zoom...")
            self.plot_newton_fractal(self.num_values, self.max_iterations)

    def plot_newton_fractal(self, num_values=500, max_iterations=1000):
        """
        This method calls _plot_newton_fractal to compute the roots and draw the fractal with the region specified
        at the start of the program run (no zoom)
        """
        self.num_values = num_values
        self.max_iterations = max_iterations

        self._plot_newton_fractal(
            self.f, self.f_prime, region=self.region, num_values=num_values, max_iterations=max_iterations,
        )

    def _set_plot_labels(self):
        """
        Set various labels on the diagrams
        """
        self.bcgw_axes.set_xlabel("log 1/s")
        self.bcgw_axes.set_ylabel("log N(S)")

    def _get_gradients(self, x_axes, y_axes):
        """
        This method calculates the average slope for our plot of log N(s) vs log 1/s, and the slope between
        adjacent points in the plot and returns them
        """
        point_gradients = []
        average_gradient = (y_axes[0] - y_axes[len(y_axes) - 1]) / (x_axes[0] - x_axes[len(x_axes) - 1])

        i = 0
        j = 1
        array_lim = len(x_axes)
        while j < array_lim:
            point_gradients.append(
                (y_axes[i] - y_axes[j]) / (x_axes[i] - x_axes[j])
            )
            i += 1
            j += 1
        point_gradients.append("")  # Add an empty value for the last item in the list of points

        return average_gradient, point_gradients

    def _plot_box_counts(self, data):
        """
        Calculate the values required to plot a graph of log N(s) vs log 1/s. The data we use for this comes from using
        the porespy library to do the box counting on the fractal
        """
        box_sizes = []   # Box size - s
        box_counts = []  # Box count - N(s)
        one_div_s = []   # 1/s
        slope = []       # slope of graph

        i = 0
        for count in data.count:
            if count > 0:
                box_counts.append(data.count[i])
                box_sizes.append(data.size[i])
                slope.append(data.slope[i])
                one_div_s.append(1 / data.size[i])
            i += 1

        log_one_div_s = np.log(one_div_s)    # calculate log 1/s
        log_box_counts = np.log(box_counts)  # calculate log N(s)
        avg_gradient, point_gradients = self._get_gradients(log_one_div_s, log_box_counts)  # slope values

        # Plot log N(s) vs log 1/s
        self.bcgw_axes.clear()
        self._set_plot_labels()
        self.bcgw_axes.plot(log_one_div_s, log_box_counts, '-o', linewidth=1, markersize=4)
        self.bcgw_axes.set_title(f"Average Gradient: {avg_gradient}")

        # Display average gradient, and the values used to compute it as a table
        print(f"Average Gradient: {avg_gradient}")
        self._display_boxcount_data_as_table(
            log_one_div_s, log_box_counts, box_counts, one_div_s, box_sizes, point_gradients, slope
        )

        coeffs_1 = np.polyfit(log_one_div_s, log_box_counts, 1)
        print(f"Coeffs 1: {coeffs_1}")

    def _display_boxcount_data_as_table(
            self, log_one_div_s, log_box_counts, box_counts, one_div_s, box_sizes, gradients, slope
    ):
        """
        This method displays the box counting data as a table
        """
        table = PrettyTable()
        table.add_column("s", box_sizes)
        table.add_column("N(s)", box_counts)
        table.add_column("log N(s)", log_box_counts)
        table.add_column("1/s", one_div_s)
        table.add_column("log 1/s", log_one_div_s)
        table.add_column("gradients", gradients)
        table.add_column("slope", slope)

        print(table)
        print("")

    def _get_root_index(self, root, roots):
        try:
            return np.where(np.isclose(roots, root, atol=self.tolerance))[0][0]
        except IndexError:
            roots.append(root)
            return len(roots) - 1

    def _plot_newton_fractal(self, f, f_prime, region=None, num_values=500, max_iterations=1000):
        """
        Plot a Newton fractal by finding the roots of f(z)
        """
        roots = []
        if region is None:
            region = self._get_region_by_size(1)
        x_min, x_max, y_min, y_max = region
        cr = np.zeros((num_values, num_values))
        cv = np.zeros((num_values, num_values))

        for i_x, x in enumerate(np.linspace(x_min, x_max, num_values)):
            for i_y, y in enumerate(np.linspace(y_min, y_max, num_values)):
                z0 = x + y*1j
                current_root, num_iterations = self.newton_raphson(z0, f, f_prime, max_iterations)
                if current_root is not None:
                    cr[i_y, i_x] = self._get_root_index(current_root, roots)
                    cv[i_y, i_x] = num_iterations

        print(f"region is {region}")
        for root in roots:
            print(f"Root: {root}")

        # https://matplotlib.org/stable/users/explain/colors/colormaps.html
        self.nfw_axes.imshow(cr, cmap='viridis', extent=region, origin='lower')
        img = self.niw_axes.imshow(cv, cmap='rainbow', extent=region, origin='lower')

        # Add the colour bar to the iterations window
        self.niw_cax.clear()
        colour_bar = self.newton_iterations_window.colorbar(img, cax=self.niw_cax, label="number of iterations")
        colour_bar.minorticks_on()

        # Use the porespy library to perform box-counting on the fractal
        # https://porespy.org/examples/metrics/tutorials/computing_fractal_dim.html
        data = ps.metrics.boxcount(cr)

        # Refresh/update the window panels and display
        self.newton_fractal_window.canvas.draw_idle()
        self._plot_box_counts(data)
        self.box_counting_gradient_window.canvas.draw_idle()
        self.newton_iterations_window.canvas.draw_idle()
        plt.show()


def main():
    #f = lambda x: 6 * x ** 4 - 2 * x ** 3 + 7 * x ** 2 + 5 * x + 25
    #f_prime = lambda x: 24 * x ** 3 - 6 * x ** 2 + 14 * x + 5

    f = lambda x: 24 * x ** 2 - 16 * x + 7
    f_prime = lambda x: 48 * x - 16

    #f = lambda x: x ** 3 - 1
    #f_prime = lambda x: 3 * x ** 2

    #f = lambda x: x ** 4 - 1
    #f_prime = lambda x: 4 * x ** 3

    region = (-2, 2, -2, 2)

    nf = NewtonsFractal(f, f_prime, region)
    nf.plot_newton_fractal()


main()




