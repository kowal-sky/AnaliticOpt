import numpy as np
import matplotlib.pyplot as plt
from optimization.analitic_opt import Tank


class CaseNotFound(Exception):
    pass


def test_case(case_name, eta_w = 0.9):

    '''
    input: case name in the format "fluid size" with fluid in ["methane", "nitrogen", "methane"] and
    size in ["small", "medium", "large"]
    
    output: Tank instance ready for ploting and optimizing

    '''


    # TODO intento de generalizar el generador de casos, por ahora está hardcodeado para 
    # TODO coincidir con los casos de Carlos
    # fluid, size  = case_name.split(" ")

    # P = 100000
    # T_air=298.15
    # eta_w = 0.7

    # if size == 'large':
    #     V = 165000
    #     U_L=0.19
    #     U_V=U_L
    #     q_b = 200000/(np.pi*((76.4**2)/4))
    # elif size == 'medium':


    if case_name == 'methane large':
        fluid = 'methane'
        P = 100000
        V = 165000
        T_air=298.15
        U_L=0.19
        U_V=U_L
        eta_w = 0.7
        q_b = 200000/(np.pi*((76.4**2)/4))
        size_name = 'large'

    elif case_name == 'hydrogen medium':
        fluid = 'hydrogen'
        P = 100000
        V = 300
        T_air=298.15
        U_L=3.73e-3
        U_V=U_L
        eta_w = 0.9
        q_b = 100/(np.pi*((8**2)/4))
        size_name = 'medium'
    
    elif case_name == 'nitrogen medium':
        fluid = 'nitrogen'
        P = 100000
        V = 800
        T_air=298.15
        U_L=0.026
        U_V=U_L
        eta_w = 0.9
        q_b = 100/(np.pi*((8**2)/4))
        size_name = 'medium'

    
    else:
        raise CaseNotFound('Case name not found, the current options are:\n\t\t"methane large"' \
        '\n\t\t"hydrogen medium"\n\t\t"nitrogen medium"')

    tank = Tank(fluid=fluid, P=P, V = V , T_air=T_air, U_L=U_L, U_V=U_V, eta_w=eta_w, q_b = q_b, size_name = size_name)

    return tank

