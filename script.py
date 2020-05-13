import bpy
import sys
import numpy as np
import mathutils

sys.stdout = open('/Users/elijah/Desktop/out-log.txt', 'w')
sys.stderr = open('/Users/elijah/Desktop/error-log.txt', 'w')

def deselect_everything():
    for obj in bpy.context.selected_objects:
        obj.select_set(state=False)
        
deselect_everything()
        
def select_all_children(obj):
    obj.select_set(state=True)
    for child in obj.children:
        select_all_children(child)
        
def spawn_from_model(model, x, y, z):
    deselect_everything()
    
    select_all_children(model)
    bpy.ops.object.duplicate()
    
    for object in bpy.context.selected_objects:
        if object.name.startswith(model.name):
            break
        
    object.location = (x,y,z)
        
    return object


class Patate:
    
    MODEL_NAME = "Potato base model"
    objects    = []

    def __init__(self, obj):
        """
        obj is the base circle of the potato
        """
        self.obj = obj
        Patate.objects.append(self)

    def get_candy(self, candy, frame_start, frame_stop):
        self.obj.keyframe_insert(data_path="location", frame=frame_start)
        self.obj.location = candy.location
        self.obj.keyframe_insert(data_path="location", frame=frame_stop)

    def look_toward_candy(self, candy):

        # vectuer regard
        look_v = self.get_face_local_normal()[:-1] # remove Z axis
        # vectuer bonbon
        toward_v = (
            candy.location[0] - self.obj.location[0],
            candy.location[1] - self.obj.location[1]
        )
        # Angle entre
        angle = np.arccos(
            np.dot(look_v, toward_v) / (np.linalg.norm(look_v) * np.linalg.norm(toward_v))
        )
        # rotation
        self.obj.keyframe_insert(data_path="rotation_euler", frame=0)
        self.obj.select_set(state=True)
        bpy.ops.transform.rotate(value=angle, orient_axis="Z")
        self.obj.keyframe_insert(data_path="rotation_euler", frame=10)



    def get_face_keypoints(self):
        grp = self.get_grp("Face_keypoints")
        vertices = []

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

    def get_face_local_normal(self):
        face_keypoints = self.get_face_keypoints()
        v1, v2, v3 = face_keypoints[0].co, face_keypoints[1].co, face_keypoints[2].co

        vec1 = np.subtract(v1, v2)
        vec2 = np.subtract(v1, v3)

        normal = mathutils.Vector(np.cross(vec1, vec2))

        return normal

    def get_face_global_normal(self):
        local_look_v = self.get_face_local_normal()
        world_mat = self.obj.matrix_world.to_3x3()
        global_look_v = world_mat @ local_look_v
        return global_look_v

    @property
    def body(self):
        return self.obj.children[0].children[1]

    @property
    def eyes(self):
        return self.obj.children[0].children[0]
    
    @classmethod
    def spawn_potato(cls, x, y, z=0, name='auto'):
        
        model = bpy.data.objects[cls.MODEL_NAME]
        object = spawn_from_model(model, x, y ,z)
               
        if name == 'auto':
            object.name = "Patate.{}".format(len(cls.objects))
        else:
            object.name = name
            
        patate = cls(object)
        return patate
    
    def eat_candy(self, candy):
        deselect_all()
        
        candy.select_set(state=True)
        bpy.ops.object.delete()
        

patate = Patate.spawn_potato(0,0)
candy = bpy.data.objects["Candy-base"]

# make potato move to candy
patate.obj.location = (10,0,0)
patate.look_toward_candy(candy)
patate.get_candy(candy, 0, 80)
