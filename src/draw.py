#!/usr/bin/env python3

"""
This file allows drawing of single graphs.

Shortcuts:
<- ->  (arrow keys) navigate between images
g      goto to specific picture
c      cluster highlight specific cluster
t      output current graph as tikz to figures/ folder
R      reset zooming and dragging
p      start and stop slideshow
n      toggle drawing of nodes
b      toggle drawing of background images
Esc    quit program
"""

import argparse
import tkinter
import tkinter.simpledialog
import logging
import os
from typing import List, Dict, Any
from pathlib import Path

import networkx as nx
import PIL.Image
import PIL.ImageTk
from draw import COLORS, tikz
from common import LOG_FORMAT, read_gpickle


class Navigator:
    def __init__(self, graphs: List[Path], gui: Dict[str, Any], start: int, backgroundlist: List[Path]):
        self.graphs = graphs
        self.gui = gui
        self.canvas = gui['canvas']
        self.top = gui['top']
        self.current_graph_index = start
        self.current_graph = read_gpickle(graphs[start])
        self.graph_count = len(graphs)
        self.background_list = backgroundlist
        self.current_cluster = 0  # this id is never assigned
        self.playback_speed = 50  # ms between frames

        self.top.title('Graph {}/{}'.format(start + 1, self.graph_count))
        self.zoom = 1
        self.x = 0
        self.y = 0
        self.playing = False
        self.draw_nodes = True
        self.draw_background = True

        self.canvas.bind('<Button-4>', self.mouse_wheel)  # linux
        self.canvas.bind('<Button-5>', self.mouse_wheel)  # linux
        self.canvas.bind('<MouseWheel>', self.mouse_wheel)  # windows
        self.canvas.bind("<ButtonPress-1>", self.scroll_start)
        self.canvas.bind("<B1-Motion>", self.scroll_move)

    def scroll_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def scroll_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def mouse_wheel(self, event):
        if event.num == 5 or event.delta == -120:
            self.zoom *= 0.9
        if event.num == 4 or event.delta == 120:
            self.zoom *= 1.1
        self.x = event.x
        self.y = event.y
        self.repaint()

    def play(self):
        self.current_graph_index = min(self.current_graph_index + 1, self.graph_count - 1)
        self.top.title('Graph {}/{}'.format(self.current_graph_index + 1, self.graph_count))
        self.repaint()
        if self.playing:
            self.top.after(self.playback_speed, self.play)

    def __call__(self, *args, **kwargs):
        event = args[0]
        key = event.keysym

        if key == 'Escape':
            self.top.quit()
        elif key == 'Right':
            self.current_graph_index = min(self.current_graph_index + 1, self.graph_count - 1)
        elif key == 'Left':
            self.current_graph_index = max(self.current_graph_index - 1, 0)
        elif key == 'c':
            cluster_id = tkinter.simpledialog.askinteger('Cluster', 'Specify number cluster id')
            self.highlight(cluster_id)
        elif key == 't':
            logging.info('generating TikZ-Code')
            graph = read_gpickle(self.graphs[self.current_graph_index])
            background = self.background_list[self.current_graph_index]
            tex_path = Path('figures').joinpath('{}.tex'.format(graph.name))

            with tex_path.open('w') as tex_file:
                print(tikz.graph2tikz(graph, background), file=tex_file)
        elif key == 'n':
            self.draw_nodes = not self.draw_nodes
        elif key == 'b':
            self.draw_background = not self.draw_background
        elif key == 'R':
            self.zoom = 1
            self.x = 0
            self.y = 0
            self.canvas.scan_dragto(0, 0)
        elif key == 'p':
            if not self.playing:
                self.playing = True
                self.play()
            else:
                self.playing = False
        elif key == 'g':
            goto = tkinter.simpledialog.askinteger('Goto', 'Specify number (zero indexed)',
                                                   initialvalue=self.current_graph_index,
                                                   minvalue=0,
                                                   maxvalue=self.graph_count)
            if goto is not None:
                self.current_graph_index = goto
        else:
            logging.debug('Button without action pressed: "{}"'.format(key))
            return

        self.top.title('Graph {}/{}'.format(self.current_graph_index + 1, self.graph_count))
        self.repaint()

    def node_info(self, item):
        tags = self.canvas.gettags(item)
        node_id = int([tag for tag in tags if tag.startswith('n')][0][1:])
        node = self.current_graph.nodes[node_id]

        cc_size = len(nx.node_connected_component(self.current_graph, node_id))
        node_info = '\n'.join('{}: {}'.format(k, v) for (k, v) in node.items())

        return 'node_id: {}\ncc size: {}\n{}'.format(node_id, cc_size, node_info)

    def highlight(self, cluster_id):
        self.gui['node_info'].set('... (select a node)')

        self.canvas.itemconfig('Node', fill='')

        cluster_tag = 'c{}'.format(cluster_id)
        items = self.canvas.find_withtag(cluster_tag)
        assert len(items) <= 1

        for item in items:
            node_id = next(nid for nid, ndata
                           in self.current_graph.nodes(data=True)
                           if ndata['cluster'] == cluster_id)

            for cc_node in nx.node_connected_component(self.current_graph, node_id):
                self.canvas.itemconfig('n{}'.format(cc_node), fill='light cyan')
            self.canvas.itemconfig(item, fill='turquoise4')

            self.current_cluster = cluster_id
            self.gui['node_info'].set(self.node_info(item))

    def add_graph(self, graph: nx.Graph) -> None:
        """Draws nodes and edges from graph onto the given canvas

        :param graph: graph whose nodes and edges will be drawn
        :return: None
        """
        for node in graph.nodes():
            x = graph.nodes[node]['x']
            y = graph.nodes[node]['y']
            r = graph.nodes[node]['r']

            degree = min(graph.degree(node), len(COLORS) - 1)

            # always draw degree 0, as it is not visible in 'edge only' mode
            if not self.draw_nodes and degree > 0:
                continue

            cid = graph.nodes[node].get('cluster', 0)  # 0 means no cluster assigned
            ctag = 'c{}'.format(cid)
            ntag = 'n{}'.format(node)

            tags = ('Node', ntag,)
            if cid != 0:
                tags += (ctag, )

            draw_id = self.canvas.create_oval(x - r, y - r, x + r, y + r, outline=COLORS[degree], width=2,
                                              activefill='turquoise1',
                                              tags=tags)
            self.canvas.tag_bind(draw_id, '<ButtonPress-1>', lambda event, cid=cid: self.highlight(cid))

        dash = (4, 1) if self.draw_nodes else None

        for n1, n2 in graph.edges:
            n1_x = graph.nodes[n1]['x']
            n1_y = graph.nodes[n1]['y']
            n2_x = graph.nodes[n2]['x']
            n2_y = graph.nodes[n2]['y']

            self.canvas.create_line(n1_x, n1_y, n2_x, n2_y, dash=dash)

    def paint_graph(self, graph: nx.Graph, bg_image: Path = None):
        degree_histogram = nx.degree_histogram(graph)
        degree_text = '\n'.join('{:>2}: {:>4} ({:.4f})'.format(degree, count, count / graph.number_of_nodes())
                                for degree, count
                                in enumerate(degree_histogram))
        graph_data = '\n'.join(str(k) + ": " + str(v)[:48] for k, v in graph.graph.items())
        self.gui['graph_info'].set('''
{}/{}
{}

# additional graph info
{}

# degree counts
{}

# connected components
number: {}
mean cc size: {:.2f}
largest: {}
'''.format(self.current_graph_index + 1,
           self.graph_count,
           nx.info(graph),
           graph_data,
           degree_text,
           nx.number_connected_components(graph),
           graph.number_of_nodes() / nx.number_connected_components(graph),
           max(len(cc) for cc in nx.connected_components(graph))
           ))
        self.canvas.delete(tkinter.ALL)
        if bg_image and self.draw_background:
            image = PIL.Image.open(bg_image)
            ow, oh = image.size
            size = int(ow * self.zoom), int(oh * self.zoom)

            photo = PIL.ImageTk.PhotoImage(image.resize(size))
            self.canvas.create_image(0, 0, image=photo, anchor=tkinter.NW, tags=('background',))
            self.canvas.image = photo  # keep reference to avoid premature gc
        else:
            self.canvas.image = None
        self.add_graph(graph)

    def repaint(self):
        self.current_graph = read_gpickle(self.graphs[self.current_graph_index])
        self.paint_graph(self.current_graph, self.background_list[self.current_graph_index])
        self.highlight(self.current_cluster)

        self.canvas.scale(tkinter.ALL, self.x, self.y, self.zoom, self.zoom)
        self.canvas.configure(scrollregion=self.canvas.bbox(tkinter.ALL))


def setup_window() -> Dict[str, Any]:
    top = tkinter.Tk()

    if os.name == 'nt':
        top.state('zoomed')  # maximize window on windows
    else:
        top.attributes('-zoomed', True)  # maximise window on linux (and maybe others)

    top.minsize(1820, 1024)
    top.title('Loading...')

    top_frame = tkinter.Frame(top)
    top_frame.pack(fill=tkinter.BOTH, expand=tkinter.YES)

    info_frame = tkinter.Frame(top_frame)
    info_frame.pack(side=tkinter.LEFT, fill=tkinter.Y, expand=tkinter.NO)

    graph_info = tkinter.StringVar(value='... graph info')
    node_info = tkinter.StringVar(value='... node info (select a node)')

    graph_info_label = tkinter.Label(info_frame, textvariable=graph_info, justify=tkinter.LEFT, font=('monospace', 10))
    graph_info_label.pack(anchor=tkinter.W)

    node_info_label = tkinter.Label(info_frame, textvariable=node_info, justify=tkinter.LEFT, font=('monospace', 10))
    node_info_label.pack(anchor=tkinter.W)

    canvas_frame = tkinter.Frame(top_frame)
    canvas_frame.pack(side=tkinter.RIGHT, fill=tkinter.BOTH, expand=tkinter.YES)

    canvas = tkinter.Canvas(canvas_frame, bg='white')
    canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)
    canvas.update()

    return {
        'canvas': canvas,
        'graph_info': graph_info,
        'node_info': node_info,
        'top': top
    }


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path',
                        help='single pickle file to be displayed or folder containing multiple pickle files')
    parser.add_argument('--start', default=0, type=int,
                        help='first file to be displayed')
    parser.add_argument('--background', default=None,
                        help='path to a folder containing images (*.tif) to be used as background')
    parser.add_argument('--debug', action='store_true',
                        help='more detailed output')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    else:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    gui = setup_window()

    input_path = Path(args.path)

    if input_path.is_file():
        gpickles = [input_path]
    else:
        gpickles = sorted(input_path.glob('*.pickle'))

    logging.info('Found {} pickles in {}'.format(len(gpickles), input_path))
    if len(gpickles) == 0:
        logging.error('There has to be at least one file! (Did you specify the correct folder?)')
        exit(1)

    if args.background is not None and Path(args.background).is_dir():
        background_list = sorted(Path(args.background).glob('*.tif'))
    else:
        background_list = [None] * len(gpickles)

    navigator = Navigator(gpickles, gui, min(args.start, len(gpickles) - 1), background_list)
    top = gui['top']
    top.bind('<Key>', navigator)
    navigator.repaint()
    top.mainloop()


if __name__ == '__main__':
    main()
