import numpy as np
import matplotlib.pyplot as plt


class Graph():
    def __init__(self, X, Y, lim_x, lim_y, figure_name):
        self.X_fpr = X
        self.Y_tpr = Y
        self.limit_X = lim_x
        self.limit_Y = lim_y
        self.figure_name = figure_name
        
    
    def x2y(self, x_lable, y_lable, x_size, y_size, lw, is_grid, fmt, line_name):
        plt.figure()
        plt.figure(figsize=(x_size, y_size))
        if is_grid:
            plt.grid()

        line_num = len(self.Y_tpr)
        if len(fmt) != line_num:
            print('Y list and fmt paramte number are different!')

        for i in range(line_num):
            plt.plot(self.X_fpr[i], self.Y_tpr[i], 
                     color=fmt[i][0], marker=fmt[i][1], ls=fmt[i][2],
                     lw=lw, label=line_name[i])
        plt.xlim(self.limit_X)
        plt.ylim(self.limit_Y)

        font2 = {
                'weight': 'normal',
                'size': 30,
                }

        plt.xlabel(x_lable, font2)
        plt.ylabel(y_lable, font2)
        plt.legend(loc="lower right",prop={'size':18})
        plt.tick_params(labelsize=23)


        plt.show()
        plt.savefig(self.figure_name)
