import os

import argparse
import random
import torch
import torch.nn
import torch.nn.parallel
import numpy
import gym
from os import path
#import planning
import utils
from dataloader import DataLoader
from imageio import imwrite
import time
import eval_policy







def process_one_episode(opt,
                        env,
                        car_path,
                        plan_file,
                        index,
                        car_sizes):
    movie_dir = path.join(
        opt.save_dir, 'videos_simulator', plan_file, f'ep{index + 1}')

    timeslot, car_id = utils.parse_car_path(car_path)
    # if None => picked at random
    inputs = env.reset(time_slot=timeslot, vehicle_id=car_id)
    done, mu, std = False, None, None
    images, states, costs, actions, mu_list, std_list, grad_list = [], [], [], [], [], [], []
    cntr = 0
    # inputs, cost, done, info = env.step(numpy.zeros((2,)))
    dist_t0 = env.controlled_car['locked'].dist
    #input_state_t0 = inputs['state'].contiguous()[-1]
    cost_sequence, action_sequence, state_sequence = [], [], []
    has_collided = False
    off_screen = False

    # Hardcode the state
    cntr_car = env.controlled_car["locked"]
    cntr_car._speed = 400.0
    cntr_car._direction[0], cntr_car._direction[1] = 1.0, 0.0

    it_limit = 60 #Avoid excess disk useage due to control flow bug
    while not done and cntr < it_limit:
        print("_"*50)
        print(f"cntr = {cntr}")
        print(f"cntr_car = {cntr_car}")
        print(f"cntr_car.step_counter = {cntr_car.step_counter}")
        print(f"cntr_car.is_auto = {cntr_car.is_autonomous}")

        input_images = inputs['context'].contiguous()
        input_states = inputs['state'].contiguous()

        print(f"len(input_images) = {len(input_images)}")

        a = [0.0, 0.0] # No acceleration/steering
        
        action_sequence.append(a)
        state_sequence.append(input_states)
        cntr += 1
        cost_test = 0

        inputs, cost, done, info = env.step(a)
        if not opt.ignore_crash and info.collisions_per_frame > 0:
            has_collided = True
            # print(f'[collision after {cntr} frames, ending]')
            done = True
        off_screen = info.off_screen

        images.append(input_images[-1])
        states.append(input_states[-1])
        costs.append([cost['pixel_proximity_cost'], cost['lane_cost']])
        cost_sequence.append(cost)
        
        actions.append(a)
        mu_list.append(mu)
        std_list.append(std)

        print(f"cntr_car.off_screen = {cntr_car.off_screen}")
        print(f"cntr_car.arrived_to_dst = {cntr_car.arrived_to_dst}")
        print(f"len(info.frames) = {len(info.frames)}")
        print(f"len(env._state_images) = {len(env.controlled_car['locked']._states_image)}")
        print(f"len(images) = {len(images)}")


    print("_"*50)
    print(f"done = {done}, it_limit = {it_limit}, cntr = {cntr}")
    
    dist_tfinal = env.controlled_car['locked'].dist
    #input_state_tfinal = inputs['state'][-1]


    if mu is not None:
        mu_list = numpy.stack(mu_list)
        std_list = numpy.stack(std_list)
    else:
        mu_list, std_list = None, None

    images = torch.stack(images)
    states = torch.stack(states)
    costs = torch.tensor(costs)
    actions = torch.tensor(actions)

    if len(images) > 3:
        images_3_channels = (images[:, :3] + images[:, 3:]).clamp(max=255)
        utils.save_movie(path.join(movie_dir, 'ego'),
                         images_3_channels.float() / 255.0,
                         states,
                         costs,
                         actions=actions,
                         mu=mu_list,
                         std=std_list,
                         pytorch=True)

        if opt.save_sim_video:
            print(f"len(info.frames) = {len(info.frames)}")
            sim_path = path.join(movie_dir, 'sim')
            print(f'[saving simulator movie to {sim_path}]')
            if not path.exists(sim_path): 
                os.mkdir(sim_path)
            for n, img in enumerate(info.frames):
                imwrite(path.join(sim_path, f'im{n:05d}.png'), img)

    returned = eval_policy.SimulationResult()
    returned.time_travelled = len(images)
    returned.distance_travelled = dist_tfinal - dist_t0 #input_state_tfinal[0] - input_state_t0[0]
    returned.road_completed = 1 if cost['arrived_to_dst'] else 0
    returned.off_screen = off_screen
    returned.has_collided = has_collided
    returned.action_sequence = numpy.stack(action_sequence)
    returned.state_sequence = numpy.stack(state_sequence)
    returned.cost_sequence = numpy.stack(cost_sequence)

    return returned


def parse_args():
    opt = eval_policy.parse_args()
    opt.ignore_crash = True
    return opt


def main():
    opt = parse_args()
    device = utils.get_device()
	
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    data_path = 'traffic-data/state-action-cost/data_i80_v0'

    dataloader = DataLoader(None, opt, 'i80')

    splits = torch.load(path.join(data_path, 'splits.pth'))


    gym.envs.registration.register(
        id='I-80-v1',
        entry_point='map_i80_ctrl:ControlledI80',
        kwargs=dict(
            fps=10,
            nb_states=opt.ncond,
            display=False,
            delta_t=0.1,
            store_simulator_video=opt.save_sim_video,
            show_frame_count=False,
        )
    )

    print('Building the environment (loading data, if any)')
    env_names = {
        'i80': 'I-80-v1',
    }
    env = gym.make(env_names[opt.map])

    plan_file = eval_policy.build_plan_file_name(opt)
    print(f'[saving to {path.join(opt.save_dir, plan_file)}]')

    # different performance metrics
    time_travelled, distance_travelled, road_completed = [], [], []
    # values saved for later inspection
    action_sequences, state_sequences, cost_sequences =  [], [], []

    #writer = utils.create_tensorboard_writer(opt)

    n_test = len(splits['test_indx'])
    n_test = min(1, n_test) # Ignore others

    time_started = time.time()
    total_images = 0

    for j in range(n_test):
        car_path = dataloader.ids[splits['test_indx'][j]]
        timeslot, car_id = utils.parse_car_path(car_path)
        car_sizes = torch.tensor(
                    dataloader.car_sizes[sorted(list(dataloader.car_sizes.keys()))[
                        timeslot]][car_id]
                )[None, :]
        simulation_result = process_one_episode(
                    opt,
                    env,
                    car_path,
                    plan_file,
                    j,
                    car_sizes
                )

        time_travelled.append(simulation_result.time_travelled)
        distance_travelled.append(simulation_result.distance_travelled)
        road_completed.append(simulation_result.road_completed)
        action_sequences.append(torch.from_numpy(
            simulation_result.action_sequence))
        state_sequences.append(torch.from_numpy(
            simulation_result.state_sequence))
        cost_sequences.append(simulation_result.cost_sequence)
        total_images += time_travelled[-1]

        log_string = ' | '.join((
            f'ep: {j + 1:3d}/{n_test}',
            f'time: {time_travelled[-1]}',
            f'distance: {distance_travelled[-1]:.0f}',
            f'success: {road_completed[-1]:d}',
            f'mean time: {torch.Tensor(time_travelled).mean():.0f}',
            f'mean distance: {torch.Tensor(distance_travelled).mean():.0f}',
            f'mean success: {torch.Tensor(road_completed).mean():.3f}',
        ))
        print(log_string)
        utils.log(path.join(opt.save_dir, f'{plan_file}.log'), log_string)

        if False: #writer is not None:
            # writer.add_video(
            #     f'Video/success={simulation_result.road_completed:d}_{j}',
            #     simulation_result.images.unsqueeze(0),
            #     j
            # )
            writer.add_scalar('ByEpisode/Success',
                              simulation_result.road_completed, j)
            writer.add_scalar('ByEpisode/Collision',
                              simulation_result.has_collided, j)
            writer.add_scalar('ByEpisode/OffScreen',
                              simulation_result.off_screen, j)
            writer.add_scalar('ByEpisode/Distance',
                              simulation_result.distance_travelled, j)

    diff_time = time.time() - time_started
    print('avg time travelled per second is', total_images / diff_time)

    torch.save(action_sequences, path.join(
        opt.save_dir, f'{plan_file}.actions'))
    torch.save(state_sequences, path.join(opt.save_dir, f'{plan_file}.states'))
    torch.save(cost_sequences, path.join(opt.save_dir, f'{plan_file}.costs'))

    if False: #writer is not None:
        writer.close()



if __name__ == '__main__':
    main()