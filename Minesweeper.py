import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as im

class minesweeper():

    def __init__(self, n, m, mineWeight):
        self.rows = n
        self.cols = m
        self.grid = np.zeros((n,m))
        self.mineWeight = mineWeight*n*m
        self.click = []
        self.history = []
        self.mineCount = len(self.grid[self.grid == -1])
        self.nonCount = self.rows*self.cols - self.mineCount
        self.found = 0
        self.view = -9*np.ones((self.rows, self.cols)).astype(np.int16)
        self.pix = np.copy(self.view)
        self.wins = 0
        self.num_aided = 0

        # Create game layout
        for i,_ in np.ndenumerate(self.grid):
            if rd.randint(0,np.size(self.grid)) <= self.mineWeight:
                self.grid[i] = -1
        for i,_ in np.ndenumerate(self.grid):
            if self.grid[i] != -1:
                r,c = i
                sub = self.grid[r-1:r+2,c-1:c+2]
                if np.size(sub) == 0:
                    sub = self.grid[r:r+2,c-1:c+2]
                    if np.size(sub) == 0:
                        sub = self.grid[r-1:r+2,c:c+2]
                        if np.size(sub) == 0:
                            sub = self.grid[r:r+2,c:c+2]
                count = len(sub[sub == -1])
                self.grid[i] = count
        self.grid = self.grid.astype(int)

    def onclick(self,event):
        self.click = [round(event.xdata), round(event.ydata)]
        plt.close()

    def showGrid(self, grid, interactive = True):
        fig = plt.figure()
        lim = max(self.rows,self.cols)
        ax = plt.gca()
        s = 1
        color = {-9: 'silver', -1: 'm', 0: 'slategrey', 1: 'b', 2: 'g', 3: 'r', 4: 'darkblue', 5: 'darkred', 6: 'c', 7: 'gold', 8: 'k'}
        for index,val in np.ndenumerate(grid):
            coord = (index[1]-s/2,index[0]-s/2)
            ax.add_artist(Rectangle(coord, s, s, fc = color[val], ec = 'k'))
        plt.xlim([0-s/2,lim-s/2])
        plt.ylim([0-s/2,lim-s/2])
        plt.xticks([i for i in range(0,lim)])
        plt.yticks([i for i in range(0,lim)])
        plt.title('Mine: magenta, 0: slate grey, 1: blue, 2: green, 3: red, 4: dark blue,\n 5: dark red, 6: cyan, 7: gold, 8: black')
        if interactive:
            fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()
    
    def transform(self, grid):
        self.pix = grid.copy()
        for i,p in np.ndenumerate(grid):
            if p == -1:
                self.pix[i] = 255
            else:
                self.pix[i] = 25*abs(p)
        self.pix = self.pix.astype(np.uint8)

    def impix(self, grid):
        self.transform(grid)
        img = im.fromarray(self.pix)
        img.show()

    def aid(self):
        for i,_ in np.ndenumerate(self.grid):
            if rd.randint(0,np.size(self.grid)) <= self.mineWeight/1.5 and self.grid[i] != -1:
                self.view[i] = self.grid[i]
                self.history.append(i)
                self.num_aided += 1
                self.found += 1

    def first_game(self):
        self.aid()
        self.transform(self.view)
        return self.pix
        
    def move(self, coord):
        terminal = False
        reward = 0
        self.history.append(coord)
        
        if self.grid[coord] == -1 or self.history.count(coord) > 1:
            # print('\nGame Over :(\n')
            reward = -1
            terminal = True
        # elif self.history.count(coord) > 1:
        #     # print('Duplicate entry. Try again :/\n')
        #     reward = -.1*self.history.count(coord)
        #     terminal = True
        else:
            self.view[coord] = self.grid[coord]
            self.found += 1
            # reward = .5
            reward += .1*(self.found - self.num_aided - 1)
        if self.found == self.nonCount:
            # print('You Win! :)\n')
            reward += 1
            terminal = True
        
        self.transform(self.view)
        img = np.copy(self.pix)
        
        if terminal:
            self.__init__(self.rows, self.cols, self.mineWeight/(self.rows*self.cols))
            self.aid()
        
        return img, reward, terminal

    def choose(self):
        return (rd.choice(range(self.rows)),rd.choice(range(self.cols)))

    def play(self, mode = 'interactive', show = True, aid = False):
        if aid:
            for i,_ in np.ndenumerate(self.grid):
                if rd.randint(0,np.size(self.grid)) <= self.mineWeight/2 and self.grid[i] != -1:
                    self.view[i] = self.grid[i]
                    self.history.append(i)
        else:
            while self.found < self.nonCount:
                if show:
                    if mode == 'interactive':
                        self.showGrid(self.view, interactive = True)
                    else:
                        self.showGrid(self.view, interactive = False)
                if mode == 'interactive':
                    inp = tuple(self.click[::-1])
                elif mode == 'boring':
                    inp = tuple([int(i) for i in input('Enter coorinate to unveil: ').split(',')])[::-1]
                elif mode == 'auto':
                    inp = self.choose()
                    print(f'Choice: {inp}')
                else:
                    print('Invalid Game Mode')
                    break
                self.history.append(inp)
                if self.grid[inp] == -1:
                    print('BOOM!\nGame Over')
                    if mode == 'auto':
                        print(f'Found: {self.found}')
                    self.view[inp] = self.grid[inp]
                    break
                elif self.history.count(inp) > 1:
                    print('Duplicate entry. Try again')
                    continue
                else:
                    self.view[inp] = self.grid[inp]
                    self.found += 1
            if self.found == self.nonCount:
                self.wins += 1
                print('You Win!')

if __name__ == '__main__':
    g = minesweeper(10,10,.175)
    # g.aid()
    # g.showGrid(g.view, interactive = False)
    # g.impix(g.view)
    # g.showGrid(g.grid, interactive = False)
    data = []
    iters = 10000
    for i in range(iters):
        print(i)
        coord = g.choose()
        g.move(coord)
        # print(img)
        # print(g.view)
        # print(f'\nFound: {g.found-g.num_aided} \nCoord: {coord} \nIteration: {i+1}')
        data.append(g.found-g.num_aided)
    plt.scatter([j for j in range(iters)], data)
    plt.show()
    print(np.mean(data))
    # img, _, _ = g.move((0,0))
    # print(torch.from_numpy(img))
    #  score = []
    # for i in range(30):
    #     g = minesweeper(5,5,.175)
    #     g.play(mode = 'auto', show = False)
    #     if g.found > 15:
    #         g.impix(g.view)
    #         break
        # score.append(g.found)
    # print(f'Max score: {max(score)}\nWins: {g.wins}')
    # plt.hist(score,20)
    # plt.show()
