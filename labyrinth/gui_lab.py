import itertools as it
import threading
import time
from tkinter import Tk
from tkinter.ttk import Button, Label

import numpy as np
from PIL import Image, ImageTk

from gen_maze import maze
from labyrinth import NeighborsGeneratorJPS, heur_diag, \
    load_from_img, load_from_map_file, normalize, lab_to_im
from searching import a_star

# window scaling for small images
SCALE = 1
# time delay between each draw of the path
PATH_DELAY_TIME = 0.
# file path
MAZE_SIZE = 801, 601
SIZE = 10
# LAB_PATH = None
LAB_PATH = "img/lab7.png"


# LAB_PATH = "img/map/ost100d.map"


class GUI(Tk):
    def __init__(self, master=None):
        Tk.__init__(self, master)

        self.labyrinth = None
        self.pix = None
        self.im = None
        self.image = None
        self.panel = None
        self.newsize = None
        self.children_gen = None
        self.heur = None
        self.heur_goal = None
        self.max_heur = None

        self.b1 = Button(self, text="Start A*", command=self.start_alg)
        self.b1.grid(row=0, column=0)
        self.b1.configure(state="disabled")
        self.load_thread = threading.Thread(target=self.init_image)
        self.load_thread.start()

    def init_image(self):
        """
        Loads the image from the file, using the appropriate function.
        """
        if LAB_PATH is None:
            print("Generating labyrinth...")
            self.labyrinth = maze(*MAZE_SIZE, size=SIZE)
            self.im = lab_to_im(self.labyrinth)
        else:
            print("Reading labyrinth from {}...".format(LAB_PATH))
            if LAB_PATH.endswith("map"):
                load_fun = load_from_map_file
            else:
                load_fun = load_from_img
            self.labyrinth, self.im = load_fun(LAB_PATH)

        self.pix = self.im.load()
        print("Read!")
        self.b1.configure(state="normal")

    def set_image(self, im):
        im2 = im.resize(self.newsize, Image.NEAREST)
        self.image = ImageTk.PhotoImage(im2)
        self.panel.configure(image=self.image)

    def gui_callback(self, queue, path=None):
        """
        Draws the A* queue, and eventually the path computed.
        """
        path = path or []
        queue = queue or []

        for node in queue:
            x, y = [int(x) for x in node[1]]  # convert numpy.int32 to int
            h = self.heur_goal((x, y))
            v = h / self.max_heur
            r = int(v * 255)
            g = int((1 - v) * 255)
            self.pix[x, y] = r, g, 0
        for node in path:
            x, y = node
            self.pix[x, y] = 0, 0, 255
        start = self.labyrinth.start
        self.pix[start[0], start[1]] = 255, 0, 0
        self.set_image(self.im)

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
        if not self.labyrinth.start or not self.labyrinth.goal:
            raise ValueError("Start or goal not found")

        self.children_gen = NeighborsGeneratorJPS(self.labyrinth)
        self.heur = heur_diag
        self.heur_goal = self.heur(self.labyrinth.goal, self.labyrinth.start)
        self.max_heur = int(self.heur_goal(self.labyrinth.start))

        print("Start detected:\t{}".format(self.labyrinth.start))
        print("Goal detected:\t{}".format(self.labyrinth.goal))
        print("Starting search...")

        c_time = time.perf_counter()
        path, *_, info = a_star(self.labyrinth.start,
                                lambda p: np.array_equal(p, self.labyrinth.goal),
                                self.heur_goal, self.children_gen,
                                self.gui_callback)
        c_time = (time.perf_counter() - c_time)

        if path:
            path, cost = self.reconstruct_path(path)
        else:
            cost = float("inf")

        # print(path)

        print("Search ended")
        print("Time:", round(c_time, 2), "s")
        print("Nodes searched:", info.nodes)
        print("Maximum list size:", info.maxl)
        print("Path cost:", cost)

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

        def pairwise(iterable):
            a, b = it.tee(iterable)
            next(b, None)
            return zip(a, b)

        cost = 0
        expanded_path = [path[0]]
        sqrt_2 = 2 ** 0.5
        for cur_node, next_node in pairwise(path):
            cur_node = np.array(cur_node)
            next_node = np.array(next_node)
            direction = normalize(next_node - cur_node)
            cost_unit = sqrt_2 if np.all(direction) else 1
            while not np.array_equal(cur_node, next_node):
                cur_node = tuple(int(x) for x in cur_node + direction)
                expanded_path.append(cur_node)
                cost += cost_unit
        return expanded_path, cost


def main():
    GUI().mainloop()


if __name__ == '__main__':
    main()
