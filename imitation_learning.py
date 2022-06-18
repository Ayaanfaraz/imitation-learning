#!/usr/bin/env python3
import glob
import os
import sys
import random
import time
import sys
from matplotlib import image
import numpy as np
import cv2
import math
import pickle

from collections import deque
# import pygame
try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
import tensorflow as tf
from threading import Thread
from multiprocessing import Process, log_to_stderr
import logging
from tqdm import tqdm
import queue

import matplotlib.pyplot as plt
import xception
import torch.nn as nn
import torch.optim as optim
import torch

import jerrys_helpers
import tiramisuModel.tiramisu as tiramisu
from torchvision import transforms
from mergedModel import MyEnsemble as fusionModel

"Starting script for any carla programming"

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

IM_WIDTH = 300
IM_HEIGHT = 300
TIMESTEPS_PER_EPISODE = 400
REPLAY_MEMORY_SIZE = 10_000

EPISODES = 130

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.9991 ## 0.9975 99975
MIN_EPSILON = 0.001

datas = []
images = []

#Load up Jerrys pretrained model
semantic_uncertainty_model = tiramisu.FCDenseNet67(n_classes=23).to(device='cuda:0')
semantic_uncertainty_model.float()
jerrys_helpers.load_weights(semantic_uncertainty_model,'models/weights67latest.th')

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.534603774547577, 0.570066750049591, 0.589080333709717],
    std = [0.186295211315155, 0.181921467185020, 0.196240469813347])
])

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()

        # weather = carla.WeatherParameters(
        # cloudiness=0.0,
        # precipitation=0.0,
        # sun_altitude_angle=90.0)

        # self.world.set_weather(weather)

        self.world.set_weather(carla.WeatherParameters.Default)
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        try:
            data = [self._retrieve_data(q, timeout) for q in self._queues]
            assert all(x.frame == self.frame for x in data)
            return data
        except:
            time.sleep(10)
            self.frame = self.world.tick()
            data = [self._retrieve_data(q, timeout) for q in self._queues]
            assert all(x.frame == self.frame for x in data)
            return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class CarEnv:
    STEER_AMT = 1.0   ## full turn for every single time
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    num_timesteps = 0

    def __init__(self):
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04')
        self.map = self.world.get_map()   ## added for map creating
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]  ## grab tesla model3 from library

    def reset(self):
        self.collision_hist = []
        self.obstacle_data=[]    
        self.actor_list = []
        self.num_timesteps = 1
        
        try:
            self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=3.0)
        except:
            time.sleep(10)

        self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=3.0)

        self.spawn_point = random.choice(self.waypoints).transform #Used to be waypoint[0]
        self.spawn_point.location.z += 2
        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)  ## changed for adding waypoints

        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")  ## fov, field of view

        self.ss_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.ss_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.ss_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.ss_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.rgb_sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_sensor)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0)) # initially passing some commands seems to help with time. Not sure why.
        self.vehicle.set_autopilot(True)
        #time.sleep(4)  # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        #We need to add trhis after we start the synchronous mode in the main loop
        
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.actor_list.append(self.colsensor)

        obstacle_detector = self.world.get_blueprint_library().find('sensor.other.obstacle')
        obstacle_detector.set_attribute("distance", f"17")
        self.obstacle_detector = self.world.spawn_actor(obstacle_detector, transform, attach_to=self.vehicle)
        self.obstacle_detector.listen(lambda event: self.obstacle_hist(event))
        self.actor_list.append(self.obstacle_detector)
        
    def collision_data(self, event):
        self.collision_hist.append(event)

    def obstacle_hist(self, event):
        self.obstacle_data.append(event)

    def process_img(self, image):
        #image.convert(cc.CityScapesPalette)
        i = np.array(image.raw_data, dtype=np.dtype("uint8"))
        i2 = i.reshape((self.im_height, self.im_width, 4))
        image = i2[:, :, :3]
       
        ## #Get semantic image ######
        # Normalize rgb input Image
        normalized_image = transform_norm(image)
        rgb_input = torch.unsqueeze(normalized_image, 0)
        rgb_input = rgb_input.to(torch.device("cuda:0"))
        # Get semantic segmented raw output
        semantic_uncertainty_model.eval().to(device='cuda:0')
        model_output = semantic_uncertainty_model(rgb_input) #Put single image rgb in tensor and pass in
        raw_semantic = jerrys_helpers.get_predictions(model_output) #Gets an unlabeled semantic image (red one)
        rgb_semantic = jerrys_helpers.color_semantic(raw_semantic[0]) #gets color converted semantic (like our convert cityscape)
        #Convert Jerry model float64 input to uint8
        rgb_semantic = cv2.normalize(src=rgb_semantic, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ##### Get Semantic Image #######


        #### Get Uncertainty Image ######
        semantic_uncertainty_model.train().to(device='cuda:0')
        mc_results = []
        output = semantic_uncertainty_model(rgb_input).detach().cpu().numpy()
        output = np.squeeze(output)
        # RESHAPE OUTPUT BEFORE PUTTING IT INTO mc_results
        # reshape into (480000, 23)
        # then softmax it
        output = jerrys_helpers.get_pixels(output)
        output = jerrys_helpers.softmax(output)
        mc_results.append(output)
        
        # boom we got num_samples passes of a single img thru the NN
        # now we use those samples to make uncertainty maps  
        mc_results = [mc_results]
        aleatoric = jerrys_helpers.calc_aleatoric(mc_results)[0]
        aleatoric = np.reshape(aleatoric, (IM_HEIGHT, IM_WIDTH))
        aleatoric = cv2.normalize(src=aleatoric, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        aleatoric = cv2.merge((aleatoric,aleatoric,aleatoric))
        # cv2.imshow("Semantic Segmentation", rgb_semantic)
        # cv2.imshow("Aleatoric Uncertainty", al)
        return rgb_semantic, aleatoric
    
    def draw_image(self, image):
        array = np.array(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        return array


    def step(self,sync_mode):
        control = self.vehicle.get_control()
        data = [control.steer, control.throttle, control.brake]
        
        print("steer ", data[0], " throttle ", data[1], " brake ", data[2])

        reward = 0
        done = False

        if len(self.collision_hist)!=0:
            self.vehicle.set_autopilot(False)
            done = True

        if self.num_timesteps >= TIMESTEPS_PER_EPISODE:  ## when to stop
            self.vehicle.set_autopilot(False)
            done = True

        try:
            snapshot, image_rgb = sync_mode.tick(timeout=20.0) #This is next state image
        except:
            print("error")
            return None, reward, done, True #return reward for old state and next state image

        semantic_segmentation, aleatoric_uncertainty = env.process_img(image_rgb)

        datas.append(data)
        images.append((semantic_segmentation,aleatoric_uncertainty))

        image_rgbs = env.draw_image(image_rgb)
        cv2.imshow("Ground Truth RGB",image_rgbs)
        cv2.waitKey(1)
        cv2.imshow("Semantic Segmentation", semantic_segmentation)
        cv2.waitKey(1)
        cv2.imshow("Aleatoric Uncertainty", aleatoric_uncertainty)
        cv2.waitKey(1)
        current_state = (semantic_segmentation, aleatoric_uncertainty)
        return current_state, reward, done, None

if __name__ == '__main__':
    FPS = 20
    # For stats
    ep_rewards = [-200]
    process_start = time.time()
    # For more repetitive results
    random.seed(1)
    #np.random.seed(2021)

    # Create models folder, this is where the model will go 
    if not os.path.isdir('models'):
        os.makedirs('models')

    startEpisode = 1
    env = CarEnv()

    rewards = []
    episode_list = []

    # Iterate over episodes
    for episode in tqdm(range(0, EPISODES), unit='episodes'):
        #try:
            pygame.init()
            env.collision_hist = []
            env.obstacle_data = []
            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 0

            # Reset environment and sensors etc
            env.reset()
            current_state = None
            # Reset done flag
            done = False
            episode_start = time.time()
            # Play for given number of seconds only
            #with synchronus mode()
            #state = rgb
            with CarlaSyncMode(env.world, env.rgb_sensor, fps=10) as sync_mode:
                
                snapshot, image_rgb = sync_mode.tick(timeout=20.0)
                proc_start = time.time()
                semantic_segmentation, aleatoric_uncertainty = env.process_img(image_rgb)
                
                proc_end = time.time()-proc_start
                current_state = (semantic_segmentation, aleatoric_uncertainty)
                while True:
                    ## For one action, apply it twice so the car can actually apply angle 
                    new_state, reward, done, err = env.step(sync_mode)
                    ##
                    if err is True:
                        print("ERROR!!!!")
                        break

                    # Transform new continous state to new discrete state and count reward
                    episode_reward += reward

                    current_state = new_state
                    step += 1
                    print("step: "+ str(step))
                    env.num_timesteps = step

                    if done:
                        break

            for actor in env.actor_list:
                actor.destroy()
            pygame.quit()
        # except:
            with open('_out/data.pkl','wb') as inf:
                pickle.dump(datas,inf)
            with open('_out/images.pkl','wb') as of:
                pickle.dump(images,of)
    #torch.cuda.empty_cache()