import sys
sys.stdout = open('/Users/elijah/Desktop/blender_log.txt', 'w')
sys.stderr = open('/Users/elijah/Desktop/err_log.txt', 'w')
import math
import time
import random
import bpy
import numpy as np
import mathutils
from itertools import chain
import copy
 
 
 
##### NATURAL SELECTION MODULE ######
#!/usr/local/bin/python3
#
# natural_selection.py
# Patate
"""
Problems:
    - two things can't be at the same place
"""
 
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
        self.coords = coords
        self.occupants = []
 
    def add(self, obj):
        self.occupants.append(obj)
 
    def remove(self, obj):
        self.occupants.remove(obj)
 
    @property
    def x(self):
        return self.coords[0]
 
    @property
    def y(self):
        return self.coords[1]
 
class GridObj:
 
    def __init__(self, box=None):
        self.box = None
        self.move(to_box=box)
 
    @property
    def x(self):
        return self.box.coords[0]
 
    @property
    def y(self):
        return self.box.coords[1]
 
    @property
    def coords(self):
        return self.box.coords
 
    def move(self, to_box):
        if self.box:
            self.box.remove(self)
        self.box = to_box
        to_box.add(self)
 
 
class Patate(GridObj):
 
    objs_count = 0
 
    def __init__(self, box=None, speed=1):
        super().__init__(box)
        self.speed  = speed
 
        self.is_dead = False
        self.eaten_count = 0
 
        self.id = Patate.objs_count
        Patate.objs_count += 1
 
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
 
 
 
class Bonbon(GridObj):
    objs_count = 0
    def __init__(self, box):
        super().__init__(box)
 
        self.id = Bonbon.objs_count
        Bonbon.objs_count += 1
 
class Board:
 
    def __init__(self, size, init_potatoes_nb=5, init_candies_nb=10):
        """
        Board object, hold squares and pawns
        """
 
        self.size = size
 
        self.grid = []
        self.potatoes = []
        self.candies  = []
        self.eaten_candies = []
 
        self.init_potatoes_nb = init_potatoes_nb
        self.init_candies_nb  = init_candies_nb
 
        self.init_grid()
 
#    def get_objects(self):
#        objs = []
#        for row in self.grid:
#            for box in row:
#                for occupant in box.occupants:
#                    if occupant:
#                        objs.append(occupant)
#
#        return objs
 
    def get_objects(self):
        objs = []
        for row in self.grid:
            for box in row:
                objs.append(box)
 
        return objs

    def clear(self):
        self.init_grid()
        self.potatoes = []
        self.candies = []
        self.eaten_candies = []
 
    def init_grid(self):
 
        self.grid = []
 
        for y in range(self.size[1]):
            line = []
            for x in range(self.size[0]):
                line.append(Box([x,y]))
 
            self.grid.append(line)
 
    def add_potato(self, x, y):
        box    = self.grid[y][x]
        potato = Patate(box=box)
        self.potatoes.append(potato)
 
    def add_candy(self, x, y):
        box   = self.grid[y][x]
        candy = Bonbon(box=box)
        self.candies.append(candy)
 
    def del_candy(self, candy):
        candy.box.remove(candy)
        self.candies.remove(candy)
        self.eaten_candies.append(candy)
 
    def del_potato(self, potato):
        potato.box.remove(potato)
        self.potatoes.remove(potato)
        potato.is_dead = True
 
    def move_potato(self, potato, new_pos):
        new_box = self.grid[new_pos[1]][new_pos[0]]
        potato.move(new_box)
 
    def new_gen(self, potatoes_nb, candy_nb):
        self.clear()
 
        # Create a list of all possible places on the map
        possible_places = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                possible_places.append([i,j])
 
        random.shuffle(possible_places)
 
        # Put 10 candies at random places
        for _ in range(candy_nb):
            candy_loc = possible_places.pop()
            self.add_candy(*candy_loc)
 
        for _ in range(potatoes_nb):
            potato_loc = possible_places.pop()
            self.add_potato(*potato_loc)

    def first_gen(self):
        self.new_gen(self.init_potatoes_nb, self.init_candies_nb)

    def next_state(self):
        self.eaten_candies = []
 
    def next_gen(self):
        next_gen_potatoes_nb = 0
        next_gen_candies_nb = self.init_candies_nb
 
        for potato in self.potatoes:
            # Delete dead potatoes
            if potato.eaten_count == 0:
                pass
 
            elif potato.eaten_count == 1:
                next_gen_potatoes_nb += 1
 
            # Spawn new potatoes
            elif potato.eaten_count > 1:
                next_gen_potatoes_nb += 2

        self.new_gen(next_gen_potatoes_nb, next_gen_candies_nb)
 
 
    def state(self, infos=[]):
 
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
 
        infos.insert(0, "") # Insert empty line at beginning
        infos_iter = iter(infos)
 
        for row_ix, row in enumerate(self.grid):
            try:
                next_info = next(infos_iter)
            except StopIteration:
                next_info = ""
 
            msg += "{}\t".format(row_ix)+"#"*border
            for val in row:
                if len(val.occupants) == 0:
                    printed = " "
                elif type(val.occupants[0]) == Patate:
                    printed = "P"
                elif type(val.occupants[0]) == Bonbon:
                    printed = "O"
                msg += " {} #".format(printed)
 
 
            msg += "#"*border
            # Add info
            msg += "\t{}".format(next_info)
 
            # New line
            msg += "\n"
 
        msg += "\t" + "#"*(self.size[0]*box_size + border*2)
 
        return msg
 
    def log_state(self, infos={}):
        potatoes_state = {potato.id: potato.coords for potato in board.potatoes}
        candies_state  = {candy.id: candy.coords for candy in board.candies}
 
        log = {
            "potatoes_state": potatoes_state,
            "candies_state": candies_state
        }
 
        log.update(infos)
 
        return log

 
def natural_selection(board, nb_gen=10, max_days=25, print_state=False, log_path=""):
    state = []
    board.first_gen()
    for gen_ix in range(nb_gen):
 
        stop_gen = False
        days_nb  = 0
 
        while not stop_gen:
 
            days_nb += 1
 
            # Move every potato
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
 
            # Check if gen is over
            if len(board.candies) == 0:
                stop_gen = True
 
            if days_nb == max_days:
                stop_gen = True
 
            yield (board, gen_ix)

            board.next_state()
 
        # Check if there are still potatoes
        if not len(board.potatoes):
            break
 
        board.next_gen()
 
    if log_path:
        with open(log_path, 'w') as f:
            json.dump(state, f)
 
 
    print("{} gens completed".format(gen_ix+1))
    print("{} Potatoes at the beginning".format(board.init_potatoes_nb))
    print("{} Potatoes at the end".format(len(board.potatoes)))
 
 

GENS = 3
DAYS = 2
board = Board((10,10), init_potatoes_nb=2, init_candies_nb=5)
n = natural_selection(board, nb_gen=GENS, max_days=DAYS)
for x in n:
    print("Gen nb:", x[1], "Board:", x[0].potatoes)

##### END OF NATURAL SELECTION #####
 
# TODO: Merge the potatoes parts into one object
# Change spawn_potatoes (currently selecting all children)
 
def deselect_everything():
    for obj in bpy.context.selected_objects:
        obj.select_set(state=False)
 
deselect_everything() # run it once in case of
 
def get_all_children(obj, children=[]):
    if not children:
        children = [obj]
    for child in obj.children:
        children.append(child)
        child_children = get_all_children(child, children=children)
        children.extend(child_children)
 
    return children
 
def select_all_children(obj, prev_children=[]):
    obj.select_set(state=True)
    for child in obj.children:
        select_all_children(child, prev_children=prev_children)
 
def spawn_from_model(model, x, y, z):
    """
    Returns parent object
    """
    deselect_everything()
 
    select_all_children(model)
    bpy.ops.object.duplicate()
 
    # Because the duplicated object is automatically selected
    for object in bpy.context.selected_objects:
        if object.name.startswith(model.name):
            break
 
    object.location = (x,y,z)
 
    return object
 
def del_object(obj):
    bpy.data.objects.remove(obj, do_unlink=True)
 
def hide_object(obj, frame=None):
    
    obj.hide_viewport = True
    if frame is not None:
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)
 
def unhide_object(obj, frame=None):
    obj.hide_viewport = False
    if frame is not None:
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)
 
 
class BlenderObjWrapper:

    def __init__(self):
        self.born_frame = None
        self.death_frame = None
 
    @classmethod
    def spawn(cls, x, y, z=0, name='auto'):
 
        model = bpy.data.objects[cls.MODEL_NAME]
 
        object = spawn_from_model(model, x, y, z)
 
        if name == 'auto':
            object.name = "{}.{}".format(cls.__name__, len(cls.objects))
        else:
            object.name = name
 
        cls_obj = cls(object)
        return cls_obj
 
 
    def center(self):
        self.obj.location = (0,0,0)
 
    def unhide_at_frame(self, frame_n):
        unhide_object(self.obj, frame=frame_n)
 
    def hide_at_frame(self, frame_n):
        hide_object(self.obj, frame=frame_n)
 
    def move(self, x, y, z=0, frame=None):
        self.obj.location = (x,y,z)
 
        if frame is not None:
            self.obj.keyframe_insert(data_path="location", frame=frame)
 
class Candy(BlenderObjWrapper):
    MODEL_NAME = "Candy base model"
    objects = []
 
    def __init__(self, obj):
        super().__init__()
        Candy.objects.append(self)
        self.obj = obj
 
 
class Potato(BlenderObjWrapper):
 
    MODEL_NAME = "Potato base model"
    objects    = []
 
    def __init__(self, obj):
        """
        obj is the base circle of the potato
        """
        super().__init__()
        Potato.objects.append(self)
        self.obj = obj
 
    def get_candy(self, candy, frame_start, frame_end):
        self.obj.keyframe_insert(data_path="location", frame=frame_start)
        self.obj.location = candy.location
        self.obj.keyframe_insert(data_path="location", frame=frame_end)
 
    def look_toward_candy(self, candy, frame_start, frame_end):
 
        assert self.obj.location != candy.location, "Potato and candy are at the same location"
 
        # Vecteur regard
        look_v      = self.get_face_global_normal()[:-1] # Remove Z axis
        # Vecteur vers le bonbon
        toward_v    = (
            candy.location[0] - self.obj.location[0],
            candy.location[1] - self.obj.location[1]
        )
 
        # Angle entre les deux
        angle       = np.arccos(
            np.dot(look_v, toward_v) / (np.linalg.norm(look_v)*np.linalg.norm(toward_v))
        )
 
        # Rotation
        self.obj.keyframe_insert(data_path="rotation_euler", frame=frame_start)
 
        self.obj.select_set(state=True)
        bpy.ops.transform.rotate(value=angle, orient_axis="Z")
 
        self.obj.keyframe_insert(data_path="rotation_euler", frame=frame_end)
 
        return
 
    def eat_candy(self, candy):
        deselect_all()
        candy.select_set(state=True)
        bpy.ops.object.delete()
 
    def get_face_local_normal(self):
        """
        Get the normal vector (look vector) in local coordinates
        """
        # Trouver 3 points du plan (v1, v2 et v3)
        face_keypoints = self.get_face_keypoints()
        v1, v2, v3 = face_keypoints[0].co, face_keypoints[1].co, face_keypoints[2].co
 
        # Trouver deux vecteurs du plan
        vec1       = np.subtract(v2, v1)
        vec2       = np.subtract(v2, v3)
 
        normal     = mathutils.Vector(np.cross(vec1, vec2))
        return normal
 
    def get_face_global_normal(self):
        local_look_v  = self.get_face_local_normal()
        world_mat     = self.obj.matrix_world.to_3x3()
        global_look_v = world_mat @ local_look_v
 
        return global_look_v
 
    def get_face_keypoints(self):
        grp         = self.get_grp("Face_keypoints")
        vertices    = []
 
        for v_ix, v in enumerate(self.body.data.vertices):
            try:
                grp.weight(v_ix)
                vertices.append(v)
            except:
                pass
        return vertices
 
    def get_grp(self, grp_name):
        for grp in self.body.vertex_groups:
            if grp.name == grp_name:
                return grp
        return False
 
    @property
    def body(self):
        return self.obj.children[0].children[1]
 
    @property
    def eyes(self):
        return self.obj.children[0].children[0]
 
 
class Animation:
 
    def __init__(self,
                 nb_gens,
                 max_days,
                 board_size,
                 init_potatoes_nb,
                 init_candies_nb,
                 scale,
                 fps=1,
                ):
 
        self.nb_gens    = nb_gens
        self.max_days   = max_days
        self.scale      = scale
        self.fps        = fps
 
        self.map        = {'Potatoes':{}, 'Candies':{}}
 
 
        self.board      = Board(size=board_size,
                                                  init_potatoes_nb=init_potatoes_nb,
                                                  init_candies_nb=init_candies_nb)
 
        self.ns = natural_selection(self.board, self.nb_gens, self.max_days)
 
    def scale_coords(self, coords):
        return [coords[0]*self.scale, coords[1]*self.scale]
 
    def init_gen_objs(self, state, frame_ix):
        for box in state.get_objects():
            for occupant in box.occupants:
                # Creer un objet blender
                if type(occupant) == Patate:
                    blender_cls = Potato
                    dict_name   = "Potatoes"
                elif type(occupant) == Bonbon:
                    blender_cls = Candy
                    dict_name   = "Candies"
 
                x_scale, y_scale = self.scale_coords(occupant.coords)
                blender_obj = blender_cls.spawn(x=x_scale, y=y_scale)

                blender_obj.born_frame = frame_ix

 
                # Map it
                self.map[dict_name][occupant.id] = blender_obj
 
    def next_gen(self, state, frame_ix):
        for box in state.get_objects():
            for occupant in box.occupants:
                # Fetch obj
                if type(occupant) == Patate:
                    dict_name   = "Potatoes"
                elif type(occupant) == Bonbon:
                    dict_name   = "Candies"
  
 
                blender_obj = self.map[dict_name][occupant.id]
 
                # Hide it from now until the end
                blender_obj.death_frame = frame_ix
 
    def apply_state(self, state, prev_state, frame_ix):
        
        prev_candies = []
        for box in prev_state.get_objects():
            for occupant in box.occupants:
                if type(occupant) == Bonbon:
                    prev_candies.append(occupant)

        for box in state.get_objects():
            for occupant in box.occupants:
                if type(occupant) == Bonbon:
                    try:
                        prev_candies.remove(occupant)
                    except:
                        print("EXCEPTION ON CANDY", occupant.id)

        for candy in prev_candies:
            blender_obj = self.map["Candies"][candy.id]
            blender_obj.death_frame = frame_ix

        
        for box in state.get_objects():
            for occupant in box.occupants:
                # Fetch obj
                if type(occupant) == Patate:
                    dict_name   = "Potatoes"
                elif type(occupant) == Bonbon:
                    dict_name   = "Candies"

 
                # Retrieve blender obj
                blender_obj = self.map[dict_name][occupant.id]
 
                # Move blender obj
                x_scale, y_scale = self.scale_coords(occupant.coords)
                blender_obj.move(x_scale, y_scale, frame=frame_ix)



    def run(self):

        frame_ix = 0
        prev_gen_ix = -1
        prev_state  = None

        for board, gen_ix in self.ns:
            print("===Frame:",frame_ix)

            print("=======GEN:", gen_ix)
            # First state of the generation ?
            if gen_ix != prev_gen_ix:
                # Init new objects
                self.init_gen_objs(board, frame_ix)
                if gen_ix > 0:
                    self.next_gen(prev_state, frame_ix)
            else:
                self.apply_state(board, prev_state, frame_ix)

            # Increment frame
            frame_ix    += 30*(1 / self.fps)
            prev_gen_ix  = gen_ix
            prev_state   = copy.deepcopy(board)

        for dic in chain(self.map.values()):
            for blender_obj in dic.values():
                if blender_obj.death_frame is None:
                    blender_obj.death_frame = frame_ix
                blender_obj.hide_at_frame(0)
                blender_obj.unhide_at_frame(blender_obj.born_frame)
                blender_obj.hide_at_frame(blender_obj.death_frame)




def clean_objects():
    start = ["Candy.", "Potato."]
    for obj in bpy.data.objects:
        for s in start:
            try:
                if obj.name.startswith(s):
                    del_object(obj)
            except:
                pass
 
 
if __name__ == "__main__":
 
 
    clean_objects()
 
    animation = Animation(
                 nb_gens=4,
                 max_days=50,
                 board_size=(10,10),
                 init_potatoes_nb=10,
                 init_candies_nb=10,
                 scale=5,
                 fps=3,
    )
    animation.run()