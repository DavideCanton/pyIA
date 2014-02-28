from pyIA.searching import *
from labirinth import *
from collections import defaultdict
from PIL import Image, ImageTk
from tkinter.tix import Tk
from tkinter.ttk import Button, Label, Entry
from datetime import datetime
import time
import threading
import os
import sys

SCALE = 1
TIME = 0.
LOAD_FUN = load_from_img
NAME = "D:/labirinth/lab5.png"
#NAME = "D:/labirinth/map/ost000a.map"
SQRT_2 = sqrt(2)


def manhattan(goal, start):
    def wrapper(x):
        return (abs(x[0] - goal[0]) +
                abs(x[1] - goal[1]))

    return wrapper


def heur_diag(goal, start):
    def wrapper(node):
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        m, M = dx, dy
        if m > M:
            m, M = M, m
        return (m * (SQRT_2 - 1) + M) * 1.001

    return wrapper


class GUI(Tk):
    def __init__(self, master=None):
        Tk.__init__(self, master)
        self.HMAX = -1
        b1 = Button(self, text="Start A*", command=self.start_alg)
        b1.grid(row=0, column=0)
        self.initImage()

    def initImage(self):
        imgpath = str(NAME)
        self.labirinth, self.im = LOAD_FUN(imgpath)
        self.pix = self.im.load()
        print("Reading labirinth from {}...".format(imgpath))

    def setImage(self, im):
        im2 = im.resize(self.newsize, Image.NEAREST)
        self.image = ImageTk.PhotoImage(im2)
        self.panel.configure(image=self.image)

    def gui_callback(self, L, path=None):
        if path is None:
            path = []
        if L is None:
            L = []
        for n in L:
            *_, (i, j) = n
            h = self.heur_goal((i, j))
            x = h / self.HMAX
            r = int(x * 255)
            g = int((1 - x) * 255)
            self.pix[j, i] = r, g, 0
        for n in path:
            i, j = n
            self.pix[j, i] = 0, 0, 255
        self.pix[self.labirinth.start[1], self.labirinth.start[0]] = 255, 0, 0
        self.setImage(self.im)

    def start_alg(self):
        children = list(self.children.values())
        for ch in children:
            ch.destroy()
        self.panel = Label(self)
        self.panel.pack()
        self.newsize = tuple([int(i * SCALE) for i in self.im.size])
        self.geometry("{}x{}+200+200".format(*self.newsize))
        self.update()
        threading.Thread(target=self.compute).start()

    def compute(self):
        if not self.labirinth.start or not self.labirinth.goal:
            raise ValueError("Start or goal not found")

        self.gen = NeighborsGeneratorPruning(self.labirinth)
        #self.gen = NeighboursGeneratorDiag(self.labirinth)
        self.heur = heur_diag
        self.heur_goal = self.heur(self.labirinth.goal, self.labirinth.start)
        self.HMAX = int(self.heur_goal(self.labirinth.start))

        print("Start detected:\t{}".format(self.labirinth.start))
        print("Goal detected:\t{}".format(self.labirinth.goal))
        print("Starting search...")

        make_eq = lambda p1: lambda p2: p1 == p2
        time = datetime.now()
        path, *_, info = a_star(self.labirinth.start,
                                make_eq(self.labirinth.goal),
                                self.heur_goal, self.gen, self.gui_callback)
        time = (datetime.now() - time).microseconds / 1000

        if path:
            path = self.reconstruct_path(path)

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
            if TIME:
                for i in range(len(path)):
                    self.gui_callback(None, path[:i])
                    time.sleep(TIME)
            else:
                self.gui_callback(None, path)

    def reconstruct_path(self, path):
        expanded_path = [self.labirinth.start]
        for a, b in zip(path, path[1:]):
            expanded_path.append(tuple(int(x) for x in a))
            a = np.array(a)
            b = np.array(b)
            dir = b - a
            f = dir[0] if dir[0] else dir[1]
            dir /= abs(f)
            while not np.array_equal(a, b):
                a = a + dir
                expanded_path.append(tuple(int(x) for x in a))
        return expanded_path


if __name__ == '__main__':
    GUI().mainloop()
