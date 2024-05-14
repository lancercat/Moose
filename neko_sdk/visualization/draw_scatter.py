import random

import numpy as np;
import pyqtgraph as pg;
import tqdm

from neko_sdk.visualization.neko_show_chart import neko_qt_chart_gui


class scatter_plot:
    def __init__(this, whitelist=None):
        this.whitelist = whitelist;
        pass;

    def set_points(this, points, labels, colors):
        this.points = points;
        this.labels = labels;
        this.colors = colors;
        pdict = {};
        for l in set(labels):
            pdict[l] = [];
        for i in range(len(labels)):
            pdict[this.labels[i]].append(points[i]);
        this.centers = {}
        for l in set(labels):
            this.centers[l] = np.mean(np.array(pdict[l]), axis=0);

        pass;

    def plot(this, wcenter=False):
        w = pg.ScatterPlotItem();
        if (this.labels is not None and this.whitelist is not None):
            for i in range(len(this.labels)):
                if this.labels[i] not in this.whitelist:
                    print("???");
                    continue;
                w.addPoints(
                    pos=this.points[i], pen=pg.mkPen(
                        color=this.colors[i],
                        width=this.sizes[i]
                    ),symbol=this.symbols[i]
                );
        elif this.labels is not None:
            for i in tqdm.tqdm(range(len(this.points))):
                w.addPoints(
                    pos=[this.points[i]], pen=pg.mkPen(
                        color=this.colors[int(this.labels[i])], size=this.sizes[i]//2,
                        width=this.sizes[i]
                    ), symbol=this.symbols[i]
                );
        else:
            for i in range(len(this.colors)):
                w.addPoints(
                    pos=[this.points[i]], pen=pg.mkPen(
                        color=this.colors[i], size=6,
                        width=5
                    ), symbol="+"
                );
        if (wcenter):
            for l in this.centers:
                w.addPoints(
                    pos=[this.centers[l]], pen=pg.mkPen(
                        color="#ffffff",
                        width=15
                    )
                )

        return w;

    @classmethod
    def random_color(cls, names):
        cdict = {};
        colors = [];
        for n in names:
            if n not in cdict.keys():
                cdict[n] = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255));
            colors.append(cdict[n]);
        return colors;

    def visualize(this, points, labels, colors,sizes=None,widths=None,symbols=None,dpath=None):
        this.set_points(points, labels, colors);
        if(sizes is None):
            this.sizes=np.zeros_like(labels)+5;
        else:
            this.sizes=sizes;
        if (widths is None):
            this.widths = this.sizes + 0;
        else:
            this.widths = widths;
        if(symbols is None):
            this.symbols=["o" for _ in this.sizes];
        else:
            this.symbols=symbols;
        wid = this.plot();
        neko_qt_chart_gui.show(wid,path=dpath);


