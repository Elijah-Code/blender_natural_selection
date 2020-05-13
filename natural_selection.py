import math
import random
import time
import sys
 
def normalize_vector(vec, fillvalue=1):
    new_vec = []
    for val in vec:
        if val > 0:
            new_vec.append(fillvalue)
        elif val == 0:
            new_vec.append(0)
        elif val < 0:
            new_vec.append(-fillvalue)
 
    return new_vec

class Box:
    def __init__(self, coords):
        self.occupants = []
        self.coords = coords

    def add(self, obj):
        self.occupants.append(obj)

    def remove(self, obj):
        self.occupants.remove(obj)

 
class GridObj:
    def __init__(self, box=None):
        self.box = None
        self.move(to_box=box)

    def move(self, to_box):
        if self.box:
            self.box.remove(self)
        to_box.add(self)
        self.box = to_box
        
    @property
    def x(self):
        return self.box.coords[0]
 
    @property
    def y(self):
        return self.box.coords[1]

    @property
    def coords(self):
        return self.box.coords
 
class Patate(GridObj):
    def __init__(self, box=None, speed=1):
        super().__init__(box)
        self.speed  = speed
        self.eaten_count = 0
 
    def find_nearest_candy(self, board):
        min_distance = 999
        nearest_candy = None

        for candy in board.candies:
            distance = math.sqrt((self.x - candy.x)**2 + (self.y - candy.y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_candy = candy
 
        return nearest_candy
 
    def next_step(self, board):
        nearest_candy = self.find_nearest_candy(board)
        if not nearest_candy:
            return False

        vec_to_candy  = [nearest_candy.x - self.x, nearest_candy.y - self.y]
        norm_vec      = normalize_vector(vec_to_candy, fillvalue=self.speed)
 
        if norm_vec[0] > vec_to_candy[0] and norm_vec[1] > vec_to_candy[1]:
            return vec_to_candy
 
        return norm_vec
 
    def can_eat(self, board):
        for candy in board.candies:
            if candy.coords == self.coords:
                return candy
        return False
 
 
class Candy(GridObj):
    def __init__(self, box=None):
        super().__init__(box)
 
class Board:
 
    def __init__(self, size, init_potatoes_nb=5, init_candies_nb=10):
        """
        Board object, hold squares and pawns
        """
        self.size = size
        self.grid = []
        self.potatoes = []
        self.candies  = []
        self.init_potatoes_nb = init_potatoes_nb
        self.init_candies_nb  = init_candies_nb
        self.init_grid()
 
    def init_grid(self):
        self.grid = []
        for y in range(self.size[1]):
            line = []
            for x in range(self.size[0]):
                line.append(Box([x,y]))
 
            self.grid.append(line)
 
    def add_potato(self, x, y):
        box = self.grid[y][x]
        potato = Patate(box)
        self.potatoes.append(potato)
 
    def add_candy(self, x, y):
        box = self.grid[y][x]
        candy = Candy(box)
        self.candies.append(candy)
 
    def del_candy(self, candy):
        candy.box.remove(candy)
        self.candies.remove(candy)
 
    def del_potato(self, potato):
        potato.box.remove(potato)
        self.potatoes.remove(potato)
 
    def move_potato(self, potato, new_pos):
        new_box = self.grid[new_pos[1]][new_pos[0]]
        potato.move(new_box)
 
    def first_gen(self):
        self.init_grid()
        # Create a list of all possible places on the map
        possible_places = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                possible_places.append([i,j])
        random.shuffle(possible_places)
 
        # Put 10 candies at random places
        for _ in range(self.init_candies_nb):
            candy_loc = possible_places.pop()
            self.add_candy(*candy_loc)
 
        for _ in range(self.init_potatoes_nb):
            potato_loc = possible_places.pop()
            self.add_potato(*potato_loc)
 
    def next_gen(self):
        self.init_grid()
        # Create a list of all possible places on the map
        possible_places = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                possible_places.append([i,j])
 
        random.shuffle(possible_places)
 
        # Put 10 candies at random places
        for _ in range(self.init_candies_nb):
            candy_loc = possible_places.pop()
            self.add_candy(*candy_loc)
 
        for potato in self.potatoes:
            # Delete dead potatoes
            if potato.eaten_count == 0:
                self.del_potato(potato)
 
            # Spawn new potatoes
            elif potato.eaten_count > 1:
                potato_loc = possible_places.pop()
                self.add_potato(*potato_loc)
 
            # Reset eaten count
            potato.eaten_count = 0
 
 
    def __repr__(self):
        box_size = 4
        border   = 1
        # First line = x coords
        msg = "\t"
        for xcoord in range(len(self.grid[0])):
            msg += str(xcoord).center(box_size)
        msg += "\n"
        # Line 2 : border up
        msg += "\t" + "#"*(self.size[0]*box_size + border*2)
        msg += "\n"
        for row_ix, row in enumerate(self.grid):
            msg += "{}\t".format(row_ix)+"#"*border
            for val in row:
                if not val.occupants:
                    printed = " "
                elif type(val.occupants[0]) == Patate:
                    printed = "P"
                elif type(val.occupants[0]) == Candy:
                    printed = "O"
                else:
                    print(type(val))
                    sys.exit(0)
                msg += " {} #".format(printed)
 
            msg += "#"*border
            msg += "\n"

        msg += "\t" + "#"*(self.size[0]*box_size + border*2)
 
        return msg
 
 
def natural_selection(board, nb_gen=10):
 
    board.first_gen()
    for gen_ix in range(nb_gen):
        stop_gen = False
        while not stop_gen:
            for potato in board.potatoes:
                # Find next step and move potato
                next_step = potato.next_step(board)
                if not next_step:
                    break
                potato_pos = [
                    potato.coords[0] + next_step[0],
                    potato.coords[1] + next_step[1]
                             ]
                board.move_potato(potato, potato_pos)
 
                # Potatoes can eat
                eatable = potato.can_eat(board)
                if eatable:
                    potato.eaten_count += 1
                    board.del_candy(eatable)
 
            # No candies left
            if len(board.candies) == 0:
                stop_gen = True
 
            print(board)
            print("Gen:", gen_ix)
            print("Number of potatoes:", len(board.potatoes))
            print("Number of candies:", len(board.candies))
            time.sleep(.1)
 
        board.next_gen()
        print("Gen {}, {} potatoes alive".format(gen_ix, len(board.potatoes)))
 
 
board = Board((10,10))
natural_selection(board)
