from searching import a_star
from labyrinth import load_from_img, load_from_map_file,\
    NeighborsGenerator, normalize
from collections import defaultdict
from PIL import Image, ImageTk
from tkinter.tix import Tk
from tkinter.ttk import Button, Label, Entry
from math import sqrt
from datetime import datetime
import time
import threading
import os
import sys
import numpy as np

# window scaling for small images
SCALE = 1
# time delay between each draw of the path
PATH_DELAY_TIME = 0.
# loading function
LOAD_FUN = load_from_img
# file path
NAME = "D:/labyrinth/lab4.bmp"
#NAME = "D:/labyrinth/map/den000d.map"
# cache of sqrt(2) value
SQRT_2 = sqrt(2)


def manhattan(goal, start):
    """
    Computes the Manhattan distance between the node and the goal.
    Example:

    | O |   |   |
    |   |   |   |
    |   |   | X |

    The distance is 4, because the shortest paths between
    (0,0) and (2,2) (like [(0,0)->(0,1)->(0,2)->(1,2)->(2,2)])
    have a cost of 4.
    """

    def wrapper(node):
        return (abs(node[0] - goal[0]) +
                abs(node[1] - goal[1])) * 1.001  # break ties

    return wrapper


def heur_diag(goal, start):
    """
    Computes the minimum distance between the node
    and the goal considering diagonal moves.

    For example:

    | O |   |   |   |
    |   |   |   |   |
    |   |   |   | X |

    The distance is 2 * sqrt(2) + 1, because the shortest path between
    (0,0) and (2,3) (like [(0,0)->(1,1)->(2,2)->(2,3)) has a cost of 4.
    """

    def wrapper(node):
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        m, M = dx, dy
        if m > M:
            m, M = M, m
        return (m * (SQRT_2 - 1) + M) * 1.001  # break ties

    return wrapper


class GUI(Tk):
    def __init__(self, master=None):
        Tk.__init__(self, master)
        self.b1 = Button(self, text="Start A*", command=self.start_alg)
        self.b1.grid(row=0, column=0)
        self.b1.configure(state="disabled")
        self.load_thread = threading.Thread(target=self.initImage)
        self.load_thread.start()

    def initImage(self):
        """
        Loads the image from the file, using the appropriate function.
        """
        imgpath = str(NAME)
        print("Reading labirinth from {}...".format(imgpath))
        self.labirinth, self.im = LOAD_FUN(imgpath)
        self.pix = self.im.load()
        print("Read!")
        self.b1.configure(state="normal")

    def setImage(self, im):
        im2 = im.resize(self.newsize, Image.NEAREST)
        self.image = ImageTk.PhotoImage(im2)
        self.panel.configure(image=self.image)

    def gui_callback(self, queue, path=None):
        """
        Draws the A* queue, and eventually the path computed.
        """
        if path is None:
            path = []
        if queue is None:
            queue = []
        for node in queue:
            i, j = [int(x) for x in node[2]]  # convert numpy.int32 to int
            h = self.heur_goal((i, j))
            x = h / self.max_heur
            r = int(x * 255)
            g = int((1 - x) * 255)
            self.pix[j, i] = r, g, 0
        for node in path:
            i, j = node
            self.pix[j, i] = 0, 0, 255
        self.pix[self.labirinth.start[1], self.labirinth.start[0]] = 255, 0, 0
        self.setImage(self.im)

    def start_alg(self):
        children = list(self.children.values())
        for child in children:
            child.destroy()  # removes the button
        self.panel = Label(self)
        self.panel.pack()
        self.newsize = tuple([int(i * SCALE) for i in self.im.size])
        self.geometry("{}x{}+200+200".format(*self.newsize))
        self.update()
        threading.Thread(target=self.compute).start()  # start the computation

    def compute(self):
        if not self.labirinth.start or not self.labirinth.goal:
            raise ValueError("Start or goal not found")

        self.children_gen = NeighborsGenerator(self.labirinth)
        self.heur = heur_diag
        self.heur_goal = self.heur(self.labirinth.goal, self.labirinth.start)
        self.max_heur = int(self.heur_goal(self.labirinth.start))

        print("Start detected:\t{}".format(self.labirinth.start))
        print("Goal detected:\t{}".format(self.labirinth.goal))
        print("Starting search...")

        make_eq_func = lambda p1: lambda p2: p1 == p2
        time = datetime.now()
        path, *_, info = a_star(self.labirinth.start,
                                make_eq_func(self.labirinth.goal),
                                self.heur_goal, self.children_gen,
                                self.gui_callback)
        time = (datetime.now() - time).microseconds / 1000

        if path:
            path = self.reconstruct_path(path)

        print(path)

        print("Search ended")
        print("Time:", time)
        print("Nodes searched:", info.nodes)
        print("Maximum list size:", info.maxl)

        if path is None:
            print("Path not found")
        else:
            print("Found path of", len(path), "nodes")
            print("Path generation completed")
            print("*" * 100)
            if PATH_DELAY_TIME:
                for i in range(len(path)):
                    self.gui_callback(None, path[:i])
                    time.sleep(PATH_DELAY_TIME)
            else:
                self.gui_callback(None, path)

    def reconstruct_path(self, path):
        """
        Reconstructs the complete path inserting nodes which
        were "jumped" by the generating function. This is possible
        because JPS generates jump only in straight or diagonal lines.

        For example:
        [(0,0),(4,4)] -> [(0,0),(1,1),(2,2),(3,3),(4,4)].
        """
        expanded_path = [self.labirinth.start, tuple(int(x) for x in path[0])]
        for cur, next in zip(path, path[1:]):
            cur = np.array(cur)
            next = np.array(next)
            dir = normalize(next - cur)
            while not np.array_equal(cur, next):
                cur = tuple(int(x) for x in cur + dir)
                expanded_path.append(cur)
        return expanded_path


if __name__ == '__main__':
    GUI().mainloop()
