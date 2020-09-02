from map_i80 import I80, I80Car
from traffic_gym_v2 import PatchedCar
import numpy as np

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class ControlledI80Car(I80Car):

    # Import get_lane_set from PatchedCar
    get_lane_set = PatchedCar.get_lane_set

    def __init__(self, df, y_offset, look_ahead, screen_w, font=None, kernel=0, dt=1/10, is_circ_road=False):
        super().__init__(df, y_offset, look_ahead, screen_w, font, kernel, dt, is_circ_road=is_circ_road)
        self.is_controlled = False
        self.buffer_size = 0
        self.lanes = None
        self.arrived_to_dst = False  # arrived to destination
        self.frames = list()
        self.removal_budget = 0
        self.reps = 0

    # TODO: Fix this
    @property
    def current_lane(self):
        # If following the I-80 trajectories
        if not self.is_controlled or len(self._states_image) < self.buffer_size:
            return super().current_lane

        x_max = self.screen_w - self.look_ahead
        x = self._position[0]
        if not self.is_circ_road and x > x_max:
            self.off_screen = True
            self.arrived_to_dst = True
        #print(f"{self}.off_screen = {self.off_screen}")
    

        # Fetch the y location
        y = self._position[1]

        # If way too up
        if y < self.lanes[0]['min']:
            self.off_screen = True
            self.arrived_to_dst = False
            return 0

        # Maybe within a sensible range?
        for lane_idx, lane in enumerate(self.lanes):
            if lane['min'] <= y <= lane['max']:
                return lane_idx

        # Or maybe on the ramp
        bottom = self.lanes[-1]['max']
        if y <= bottom + 53 - x * 0.035:
            return 6

        # Actually, way too low
        self.off_screen = True
        self.arrived_to_dst = False
        return 6

    @property
    def is_autonomous(self):
        return self.is_controlled and len(self._states_image) > self.buffer_size

    def step(self, action, env=None):
        super().step(action)
        len_road =  self.screen_w - self.look_ahead
        if self.is_autonomous and self.is_circ_road and self._position[0] > len_road:
            self._position[0] -= len_road
            self.reps += 1
            self.removal_budget += self._width*self._length

    @property
    def valid(self): # TODO: Dont define the finish line in seperate locations (eg. current_lane)
        return self.reps > 0 or (\
            self.back[0] > self.look_ahead and (\
                self.front[0] < self.screen_w - 1.75 * self.look_ahead or self.is_circ_road
            ) 
        )



class ControlledI80(I80):

    # Environment's car class
    EnvCar = ControlledI80Car

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, safety_factor = 0.0, is_circ_road=False, **kwargs):
        super().reset(**kwargs)
        self.mode = "circular" if is_circ_road else "straight"

        MAX_SPEED = 130
        look_ahead = MAX_SPEED * 1000 / 3600 * self.SCALE
        look_sideways = 2 * self.LANE_W
        view_length = 2 * look_ahead
        view_width = 2 * look_sideways
        self.safety_length = view_length * safety_factor
        self.safety_width = view_width * safety_factor

        observation = None
        while observation is None:
            observation, reward, done, info = self.step()
        return observation

    def is_in_view(self, car):
        cont_car = self.controlled_car.get('locked')
        # Background cars are blind
        #if not cont_car.is_autonomous:
         #   return False

        if cont_car is None or cont_car == False or not cont_car.is_autonomous:
            return False

        def get_car_corners(v, length, width):
            center = v._position + 0.5* v._direction * v._length
            ortho_direction = np.array((v._direction[1], -v._direction[0]))

            parallel_offset = v._direction * 0.5*length 
            orth_offset = ortho_direction * 0.5*width

            res = []
            for sign1, sign2 in [(-1, -1), (1, -1),  (1, 1), (-1, 1)]:
                res.append(center + sign1*parallel_offset + sign2*orth_offset)
            return res

        def get_safety_corners(cont_car):
            return get_car_corners(cont_car, self.safety_length, self.safety_width)

        # ToDo: If runtime slows down alot, consider comparing distance between rectangles 
        # to rule out very far away points quickly. Note that usually, close points will be compared
        def is_overlap(rect1, rect2):
            rect1 = Polygon(rect1)
            rect2 = Polygon(rect2)
            return rect1.intersects(rect2)

        car_corners = get_car_corners(car, car._length, car._width)
        vision_corners = get_safety_corners(cont_car)

        if is_overlap(vision_corners, car_corners):
            return True
        if not self.is_circ_road():
            return False
        # Constant taken from step
        len_road =  cont_car.screen_w - cont_car.look_ahead
        # ToDo speed this up by checking one or two rectangles instead of three
        return is_overlap([c+np.array([len_road, 0]) for c in car_corners], vision_corners) or\
        is_overlap([c-np.array([len_road, 0]) for c in car_corners], vision_corners)
