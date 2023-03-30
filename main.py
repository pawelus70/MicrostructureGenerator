import itertools
import random
from PIL import Image
import os
import subprocess
# import numpy
from abaqus import *
import numpy as np
import PySimpleGUI as sg
import threading
from generatemesh import generate_mesh
from structureAnalisis import structure_analsis


def make_win2():
    sg.theme('Dark2')
    # layout = [[sg.Text('Start plane: '), sg.Image('start.png'), sg.Text('End plane: '), sg.Image('end.png')]]
    layout = [[sg.Text('Start plane: '), sg.Image('start.png'), sg.Text('End plane: '), sg.Image('end.png')]]
    return sg.Window('Results', layout, finalize=True)


def make_win3():
    sg.theme('Dark2')
    # layout = [[sg.Text('Start plane: '), sg.Image('start.png'), sg.Text('End plane: '), sg.Image('end.png')]]
    layout = [[sg.Text('Start plane: '), sg.Image('start.png'), sg.Text('End plane: '), sg.Image('end.png'),
               sg.Text('Mesh plane: '), sg.Image('output_mesh.png')]]
    return sg.Window('Results', layout, finalize=True)

def make_win4():
    sg.theme('Dark2')
    # layout = [[sg.Text('Start plane: '), sg.Image('start.png'), sg.Text('End plane: '), sg.Image('end.png')]]
    layout = [[sg.Text('Label plane: '), sg.Image('labeled.png')]]
    return sg.Window('Results', layout, finalize=True)


def popup(message):
    sg.theme('Black')
    layout = [[sg.Text(message)]]
    return sg.Window('Message', layout, no_titlebar=True, keep_on_top=True, finalize=True)


def generate_ms(algorithm, random_seeds, absorbing, nbh_type_vn, empty, types_of_grain, nucleation_number, limit_of_steps):
    # Sets of colors (Can be randomized by rand, but we prefer to have max 10 types of grain)
    # FIX 10 color
    pixels = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
              (255, 0, 255), (125, 125, 255), (125, 255, 125), (255, 125, 125)]

    # Settings of grain, size and method (max 10 types)
    types_of_grain = int(types_of_grain)
    size_of_plane = 100
    #nucleation_number = int(np.power(size_of_plane, 2) * 0.004)
    nucleation_number = int(nucleation_number)
    absorbing = absorbing
    nbh_type_vn = nbh_type_vn

    # Settings if we want specific order
    random_seeds = random_seeds
    # numbers_in_x = 3
    # numbers_in_y = 5
    # step_in_x = int(np.floor((size_of_plane / numbers_in_x)))
    # step_in_y = int(np.floor((size_of_plane / numbers_in_y)))
    step_in_x = int(size_of_plane * 0.1)
    step_in_y = int(size_of_plane * 0.2)

    # Settings for MC method
    limit_of_step = int(limit_of_steps)
    kt = 0.2

    # List of method coordinates
    vn_cord = [[(0, -1), (1, 0), (0, 1), (-1, 0)]]
    hx_cord = [[(0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1)],
               [(-1, -1), (0, -1), (-1, 0), (1, 0), (0, 1), (1, 1)]]

    # VN cross-check
    def vn_hex_ca(current_step, next_step, check_arr, type_of_nh):
        # Fill all empty spaces until all are filled
        while np.count_nonzero(current_step) != current_step.size:
            for y in range(size_of_plane):
                for x in range(size_of_plane):
                    if current_step[y, x] == 0:
                        # There is the part where the fun begin

                        # Randomize if hex
                        current_method_type = type_of_nh[np.random.randint(0, len(type_of_nh))]

                        # Like For Each
                        for new_c in current_method_type:
                            # Adding new coordinates
                            temp_x = x + new_c[0]
                            temp_y = y + new_c[1]

                            # If absorbing you can't go out of bounds
                            if absorbing:
                                if 0 <= temp_x < size_of_plane and 0 <= temp_y < size_of_plane:
                                    if current_step[temp_y, temp_x] != 0:
                                        check_arr[current_step[temp_y, temp_x]] += 1
                            else:
                                # Modify coordinates if periodic and it's out of bounds
                                if temp_x < 0:
                                    temp_x = temp_x + size_of_plane
                                if temp_x >= size_of_plane:
                                    temp_x = 0
                                if temp_y < 0:
                                    temp_y = temp_y + size_of_plane
                                if temp_y >= size_of_plane:
                                    temp_y = 0

                                if current_step[temp_y, temp_x] != 0:
                                    check_arr[current_step[temp_y, temp_x]] += 1

                    # Fill empty slots with specific or random value
                    # Check if there are any neighbor
                    if np.any(check_arr):
                        # Check Where is the maximum
                        result = np.where(check_arr == np.amax(check_arr))
                        # If more than one, jest shuffle and take random
                        if result[0].size > 1:
                            np.random.shuffle(result[0])
                            next_step[y, x] = result[0][0]
                        else:
                            # Else if there is only one with maximum take this
                            next_step[y, x] = result[0][0]
                    # Clear check array for next loop
                    check_arr = np.zeros(types_of_grain + 1)

            # Go to next step
            current_step = np.copy(next_step)

        return current_step

    # Rand with permutation // Can be deleted
    def rand_prem_nc(current_step):
        # Prepare random allocation (only individual grain)
        arr = np.random.permutation(types_of_grain)

        # Create random nucleation ~test
        for x in range(nucleation_number):
            x_axis = np.random.randint(0, size_of_plane)
            y_axis = np.random.randint(0, size_of_plane)
            current_step[y_axis, x_axis] = (arr[x] + 1)

        return current_step

    # Rand without permutation
    def rand_prem(current_step):
        # Prepare random allocation (grain can repeat)
        processed_seeds = 0
        while processed_seeds != nucleation_number:
            x_axis = np.random.randint(0, size_of_plane)
            y_axis = np.random.randint(0, size_of_plane)
            if current_step[y_axis, x_axis] == 0:
                current_step[y_axis, x_axis] = np.random.randint(1, types_of_grain + 1)
                processed_seeds += 1

        return current_step

    # Rand with specific order
    def spec_order(current_step):
        # Allocation in specific order
        for y in range(1, size_of_plane, step_in_y):
            for x in range(1, size_of_plane, step_in_x):
                current_step[y, x] = np.random.randint(1, types_of_grain + 1)

        return current_step

    # Core od CA
    def cellular_automata():
        # Prepare plan for simulation
        current_step = np.zeros((size_of_plane, size_of_plane), np.int8)

        # Create random seeds
        if random_seeds:
            current_step = rand_prem(current_step)
        else:
            current_step = spec_order(current_step)

        # Which method to use VN or HEX
        if nbh_type_vn:
            type_of_nh = np.copy(vn_cord)
        else:
            type_of_nh = np.copy(hx_cord)

        # Copy to next step
        next_step = np.copy(current_step)
        # print("Start plane: \n", next_step)

        # Prepare array for VN / HEX
        check_arr = np.zeros(types_of_grain + 1, np.int8)

        to_img = vn_hex_ca(current_step, next_step, check_arr, type_of_nh)
        # Display result
        # print("CA: \n", to_img)

        return current_step, to_img

    # Core of MC
    def monte_carlo(ca_state):
        # Copy CA if possible
        current_step = np.copy(ca_state)
        # Create all indexes for plane
        indexes = list(itertools.product(np.arange(0, size_of_plane), np.arange(0, size_of_plane)))

        # Which method to use VN or HEX
        if nbh_type_vn:
            type_of_nh = np.copy(vn_cord)
        else:
            type_of_nh = np.copy(hx_cord)

        # Check plane if is filled
        if np.count_nonzero(current_step) == 0:
            if random_seeds:
                current_step = np.random.randint(1, types_of_grain + 1, (size_of_plane, size_of_plane))
            else:
                z = 0
                for y in range(size_of_plane):
                    for x in range(size_of_plane):
                        current_step[y, x] = (z % types_of_grain) + 1
                        z += 1

        # Copy to next step
        next_step = np.copy(current_step)
        # print("Start plane: \n", next_step)
        # print("Indexes: \n", indexes)

        # Checking and applying through all indexes
        for step in range(limit_of_step):
            np.random.shuffle(indexes)
            for index in indexes:

                # Prepare array for VN / HEX
                check_arr = np.zeros(types_of_grain + 1, np.int8)

                # Randomize if hex
                current_method_type = type_of_nh[np.random.randint(0, len(type_of_nh))]

                for new_c in current_method_type:
                    # Adding new coordinates
                    temp_x = index[0] + new_c[0]
                    temp_y = index[1] + new_c[1]

                    # If absorbing you can't go out of bounds
                    if absorbing:
                        if 0 <= temp_x < size_of_plane and 0 <= temp_y < size_of_plane:
                            if current_step[temp_y, temp_x] != 0:
                                check_arr[current_step[temp_y, temp_x]] += 1
                    else:
                        # Modify coordinates if periodic and it's out of bounds
                        if temp_x < 0:
                            temp_x = temp_x + size_of_plane
                        if temp_x >= size_of_plane:
                            temp_x = 0
                        if temp_y < 0:
                            temp_y = temp_y + size_of_plane
                        if temp_y >= size_of_plane:
                            temp_y = 0

                        if current_step[temp_y, temp_x] != 0:
                            check_arr[current_step[temp_y, temp_x]] += 1

                # Checking is there different neighbours
                if check_arr[current_step[index[1], index[0]]] == current_method_type.size:
                    continue
                # Checking energy
                # check_arr[0] = 0
                current_energy = current_method_type.size - check_arr[current_step[index[1], index[0]]]
                arr_of_candidates = np.where(check_arr != 0)[0]
                candidate = arr_of_candidates[np.random.randint(0, arr_of_candidates.size)]
                new_energy = current_method_type.size - check_arr[candidate]
                # Calculate difference
                energy_difference = new_energy - current_energy

                # New type based on energy
                if energy_difference > 0:
                    probability = np.exp(-energy_difference / kt)
                    if np.random.random() < probability:
                        next_step[index[1], index[0]] = candidate
                else:
                    next_step[index[1], index[0]] = candidate

            current_step = np.copy(next_step)
            # print("next plane: \n", next_step)
        return current_step

    # Generate image
    def generate_image(start_plane, end_plane):
        # Making image from results
        start_image = Image.new('RGB', (size_of_plane, size_of_plane))
        end_image = Image.new('RGB', (size_of_plane, size_of_plane))
        start_pixels = start_image.load()
        end_pixels = end_image.load()

        # Adding pixels to image
        for x in range(size_of_plane):
            for y in range(size_of_plane):
                start_pixels[x, y] = pixels[start_plane[y][x]]
                end_pixels[x, y] = pixels[end_plane[y][x]]

        # Saving image
        start_image = start_image.resize(((size_of_plane * 3), (size_of_plane * 3)), Image.NEAREST)
        start_image.save('start.png')
        end_image.save('genmesh.png')
        end_image = end_image.resize(((size_of_plane * 3), (size_of_plane * 3)), Image.NEAREST)
        end_image.save('end.png')

    # Main program
    if algorithm == "CA":
        input_plane, output = cellular_automata()
    elif algorithm == "MC":
        if empty:
            input_plane = np.zeros((size_of_plane, size_of_plane), np.int8)
        else:
            temp_plane, ca_plane = cellular_automata()
            input_plane = ca_plane

        output = monte_carlo(input_plane)
    else:
        print("SyntaX")
        return

    generate_image(input_plane, output)
    # generate_mesh()
    window.write_event_value('JOB DONE', None)


# (Algorithm[CA, MC], Random nucleation?, Absorbing?, VN type?, Empty?, (int) number of grain types [max: 10], number of steps for MC)
# generate_ms("CA", False, True, False, False, 5, 10)
def gen_mesh():
    generate_mesh()
    window.write_event_value('JOB DONE MESH', None)


def gen_struct():
    structure_analsis()
    window.write_event_value('JOB DONE STRUCT', None)


sg.theme('Dark2')
# All the stuff inside window.
layout = [[sg.Text('Parameters:')],
          [sg.Text('Method'), sg.Combo(['CA', 'MC'], default_value='CA', key='met')],
          [sg.Checkbox('Random seeds', default=True, key='rand')],
          [sg.Checkbox('Absorbing', default=True, key='abs')],
          [sg.Checkbox('Von Neuman', default=True, key='vn')],
          [sg.Text('Number of grain:')],
          [sg.InputText(size=15, default_text=2, key='nog')],
          [sg.Text('Quantity of grain:')],
          [sg.InputText(size=15, default_text=80, key='nn')],
          [sg.HorizontalSeparator()],
          [sg.Text('Settings for MC:')],
          [sg.Checkbox('Start with empty', default=True, key='emp')],
          [sg.Text('Steps:')],
          [sg.InputText(size=15, default_text=10, key='stp')],
          [sg.Button('Proceed'), sg.Button('Results'), sg.Button('GenMesh'), sg.Button('GenStruct')]]

# Create the Window
window = sg.Window('Microstructure maker', layout, size=(300, 450), element_justification='c')
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
        break
    elif event == 'Proceed':
        # generate_ms(algorithm, random_seeds, absorbing, nbh_type_vn, empty, types_of_grain, limit_of_steps)
        popup_w = popup('Please wait...')
        threading.Thread(target=generate_ms, args=(
        values['met'], values['rand'], values['abs'], values['vn'], values['emp'], values['nog'], values['nn'], values['stp']),
                         daemon=True).start()
        # generate_ms(values['met'], values['rand'], values['abs'], values['vn'], values['emp'], values['nog'],
        # values['stp'])
    elif event == 'JOB DONE':
        popup_w.close()
        window2 = make_win2()
    elif event == 'JOB DONE MESH':
        popup_w.close()
        window3 = make_win3()
    elif event == 'JOB DONE STRUCT':
        popup_w.close()
        window4 = make_win4()
    elif event == 'GenMesh':
        popup_w = popup('Please wait...')
        threading.Thread(target=gen_mesh, args=(), daemon=True).start()
    elif event == 'GenStruct':
        popup_w = popup('Please wait...')
        threading.Thread(target=gen_struct, args=(), daemon=True).start()
    elif event == 'Results':
        window3 = make_win3()
    print('You entered ', values['met'], values['rand'], values['abs'], values['vn'], values['nog'], values['nn'], values['emp'],
          values['stp'])

window.close()
