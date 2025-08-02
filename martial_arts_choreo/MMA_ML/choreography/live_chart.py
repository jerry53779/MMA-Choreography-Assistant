# choreography/live_chart.py
import collections
import threading
import time
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation

MAX_POINTS = 50

class LiveChart:
    """
    A class to create and manage a live-updating chart for MMA pose data.
    """
    def __init__(self, title: str, x_label: str, y_label: str):
        self.data_queue = queue.Queue()
        self.x_data = collections.deque(maxlen=MAX_POINTS)
        self.y_data = collections.deque(maxlen=MAX_POINTS)

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.x_data, self.y_data)
        
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.grid(True)
        
        self.anim = None

    def add_data(self, x_value: float, y_value: float):
        """Adds a new data point to the internal queue for plotting."""
        self.data_queue.put((x_value, y_value))

    def _update_plot(self, frame):
        """The update function for the Matplotlib animation."""
        while not self.data_queue.empty():
            x_val, y_val = self.data_queue.get()
            self.x_data.append(x_val)
            self.y_data.append(y_val)
        
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        
        return self.line,

    def start_chart(self, interval_ms: int = 200):
        """Starts the live chart animation."""
        self.anim = animation.FuncAnimation(
            self.fig, 
            self._update_plot, 
            interval=interval_ms, 
            blit=True
        )
        plt.show(block=False)

    def stop_chart(self):
        """Stops the chart animation and closes the plot window."""
        if self.anim:
            self.anim.event_source.stop()
        plt.close(self.fig)