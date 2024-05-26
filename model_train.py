from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import random
import serial
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# python train.py --train -e 50 -m model_name -s 500 (for train the model)
# python train.py -m model_name -s 500 (for test the model)
# we need to import python modules from the $SUMO_HOME/tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

def duration_to_vector(duration):
    # Create a zero vector of size 26
    action_vector = np.zeros(26)

    # Set the index corresponding to the duration value to 1
    action_vector[duration - 15] = 1

    return action_vector
def get_vehicle_numbers(lanes):
    vehicle_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = 0
        for k in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(k) > 10:
                vehicle_per_lane[l] += 1
    return vehicle_per_lane

def get_state(lanes):
    vehicle_per_lane = dict()
    waiting_time_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = 0
        waiting_time_per_lane[l] = 0
        for k in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(k) > 10:
                vehicle_per_lane[l] += 1
                waiting_time_per_lane[l] += traci.vehicle.getWaitingTime(k)
    return list(vehicle_per_lane.values()) + list(waiting_time_per_lane.values())
def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time


def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)



class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.linear4 = nn.Linear(self.fc3_dims, self.n_actions)


        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        actions = self.linear4(x)
        return actions


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        fc3_dims,
        batch_size,
        n_actions,
        junctions,

        max_memory_size=500000,
        epsilon_dec=5e-6,
        epsilon_end=0.2,

    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100


        self.Q_eval = Model(
            self.lr, self.input_dims, self.fc1_dims, self.fc2_dims,self.fc3_dims, self.n_actions
        )
        self.memory = dict()
        for junction in junctions:
            self.memory[junction] = {
                "state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "new_state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "reward_memory":np.zeros(self.max_mem, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem, dtype=np.bool_),
                "mem_cntr": 0,
                "iter_cntr": 0,
            }


    def store_transition(self, state, state_, action,reward, done,junction):
        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = state_
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation):
        #print("epsilon:",self.epsilon)
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)

        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        """lane = action // 11  # 44 aksiyonu 4 şeride böler
        duration = action % 11 + 15  # 44 aksiyonu 11 süreye böler ve 15 ekler (15-25 arası süre)"""

        # Aksiyonu süre setinin boyutuna göre mod alır ve bu indekse karşılık gelen süreyi seçer

        return action
    
    def reset(self,junction_numbers):
        for junction_number in junction_numbers:
            self.memory[junction_number]['mem_cntr'] = 0

    def save(self,model_name):
        torch.save(self.Q_eval.state_dict(),f'models/{model_name}.bin')

    def learn(self, junction):
        self.Q_eval.optimizer.zero_grad()

        batch= np.arange(self.memory[junction]['mem_cntr'], dtype=np.int32)

        state_batch = torch.tensor(self.memory[junction]["state_memory"][batch]).to(
            self.Q_eval.device
        )
        new_state_batch = torch.tensor(
            self.memory[junction]["new_state_memory"][batch]
        ).to(self.Q_eval.device)
        reward_batch = torch.tensor(
            self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device)
        action_batch = self.memory[junction]["action_memory"][batch]

        q_eval = self.Q_eval.forward(state_batch)[batch, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        assert q_target.dim() == q_eval.dim(), "q_target and q_eval dimensions do not match"

        # Check if the tensors contain any NaN or Inf values
        assert torch.isfinite(q_target).all(), "q_target contains NaN or Inf values"
        assert torch.isfinite(q_eval).all(), "q_eval contains NaN or Inf values"

        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = (
            self.epsilon - self.epsilon_dec
            if self.epsilon > self.epsilon_end
            else self.epsilon_end

        )


def run(train=True,model_name="model",epochs=50,steps=500,ard=False):
    if ard:
        arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
        def write_read(x):
            arduino.write(bytes(x, 'utf-8'))
            time.sleep(0.05)
            data = arduino.readline()
            return data
    """execute the TraCI control loop"""
    epochs = epochs
    steps = steps
    best_time = np.inf
    total_time_list = list()
    traci.start(
        [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"]
    )
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))
    print(all_junctions)
    brain = Agent(
        gamma=0.99,
        epsilon=1,
        lr=0.06,
        input_dims=9,
        # input_dims = len(all_junctions) * 4,
        fc1_dims=256,
        fc2_dims=256,
        fc3_dims=256,
        batch_size=2048,
        n_actions=30,
        junctions=junction_numbers,

    )

    if not train:
        brain.Q_eval.load_state_dict(torch.load(f'models/{model_name}.bin',map_location=brain.Q_eval.device))

    print(brain.Q_eval.device)
    traci.close()
    for e in range(epochs):
        if train:
            traci.start(
            [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )
        else:
            traci.start(
            [checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfodemo.xml","--start"]
            )

        print(f"epoch: {e}")
        """select_lane = [
            ["yryr", "GrGr"],
            ["ryry", "rGrG"],
            ["yryr", "GrGr"],
            ["ryry", "rGrG"],
        ]"""

        select_lane = [
             ["yyyyrrrrrrrrrrrr", "GGGGrrrrrrrrrrrr"],
             ["rrrryyyyrrrrrrrr", "rrrrGGGGrrrrrrrr"],
             ["rrrrrrrryyyyrrrr", "rrrrrrrrGGGGrrrr"],
             ["rrrrrrrrrrrryyyy", "rrrrrrrrrrrrGGGG"],
         ]

        step = 0
        total_time = 0
        duration = 10
        traffic_lights_time = dict()
        prev_vehicles_per_lane = dict()
        prev_action = dict()
        all_lanes = list()
        waiting_time = 0
        reward = 0
        lane = 0
        for junction_number, junction in enumerate(all_junctions):
            prev_waiting_time = 0
            prev_action[junction_number] = 0
            traffic_lights_time[junction] = 0
            prev_vehicles_per_lane[junction_number] = [0] * 8
            # prev_vehicles_per_lane[junction_number] = [0] * (len(all_junctions) * 4) 
            all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

        while step <= steps:
            traci.simulationStep()
            for junction_number, junction in enumerate(all_junctions):
                controled_lanes = traci.trafficlight.getControlledLanes(junction)

                waiting_time = get_waiting_time(controled_lanes)


                #print("waiting_time:",waiting_time,"prev_waiting_time:",prev_waiting_time)
                total_time += waiting_time

                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(controled_lanes)
                    # vehicles_per_lane = get_vehicle_numbers(all_lanes)

                    reward = (prev_waiting_time - waiting_time)
                    #print("reward:",reward,"waiting_time:",waiting_time,"prev_time",prev_waiting_time)

                    # storing previous state and current state
                    state_ = list(vehicles_per_lane.values()) + [waiting_time]
                    deneme = list(prev_vehicles_per_lane[junction_number]) + [prev_waiting_time]
                    state = (prev_vehicles_per_lane[junction_number]+[prev_waiting_time])

                    #print("state:",state,"state_:",state_)
                    prev_vehicles_per_lane[junction_number] = state_[:-1]
                    prev_waiting_time = waiting_time
                    brain.store_transition(state, state_, duration-10,reward,(step==steps),junction_number)

                    #selecting new action based on current state
                    duration = brain.choose_action(state_) + 10
                    prev_action[junction_number] = lane%4
                    phaseDuration(junction, 3, select_lane[lane%4][0])
                    phaseDuration(junction, duration, select_lane[lane%4][1])

                    traffic_lights_time[junction] = duration
                    lane+=1
                    if train:
                        brain.learn(junction_number)
                else:
                    traffic_lights_time[junction] -= 1
            step += 1
        print("total_time",total_time)
        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                brain.save(model_name)

        traci.close()
        sys.stdout.flush()
        if not train:
            break
    if train:
        plt.plot(list(range(len(total_time_list))),total_time_list)
        plt.xlabel("epochs")
        plt.ylabel("total time")
        plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
        plt.show()

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="model",
        help="name of model",
    )
    optParser.add_option(
        "--train",
        action = 'store_true',
        default=False,
        help="training or testing",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=50,
        help="Number of epochs",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=500,
        help="Number of steps",
    )
    optParser.add_option(
       "--ard",
        action='store_true',
        default=False,
        help="Connect Arduino", 
    )
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    model_name = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    ard = options.ard
    run(train=train,model_name=model_name,epochs=epochs,steps=steps,ard=ard)
