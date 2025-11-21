import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from scipy.optimize import minimize
import math


class Tank:

    St = 1.02
    
    def __init__(self, fluid, P, V, T_air, U_L, U_V = None, eta_w = 0.8, q_b = None, size_name = None):


        # * fluid properties

        self.fluid = fluid

        self.P = P

        # T_L = T_sat
        self.T_L = CP.PropsSI('T', 'P', P, 'Q', 1, fluid) #     K

        self.rho_L = CP.PropsSI('D', 'P', P, 'Q',0, fluid) #      
        self.rho_V_avg = CP.PropsSI('D', 'P', P, 'Q',1, fluid) #  

        self.h_L = CP.PropsSI('H', 'P', P, 'Q', 0, fluid)
        self.h_V = CP.PropsSI('H', 'P', P, 'Q', 1, fluid)

        # es cp_V_avg, pero en el modelo no se usa cp_L
        self.cp_avg = CP.PropsSI('C', 'P', P, 'Q', 1, fluid) 
        
        self.k_V_avg = CP.PropsSI('L', 'P', P, 'Q', 1, fluid)


        # * tank properties

        self.V_T = V
        self.T_air = T_air
        self.U_L = U_L

        if U_V:
            self.U_V = U_V
        else:
            self.U_V = U_L

        self.eta_w = eta_w

        self.q_b = q_b

        self.size_name = size_name

    # def fill_tank(self, cryogen):


    #     # self.T_L = cryogen.T_L #     K

    #     # self.rho_L = cryogen.rho_L #      
    #     # self.rho_V_avg = cryogen.rho_V_avg #  

    #     # self.h_L = cryogen.h_L
    #     # self.h_V = cryogen.h_V

    #     # # es cp_V_avg, pero en el modelo no se usa cp_L
    #     # self.cp_avg = cryogen.cp_avg 

    #     # self.k_V_avg = cryogen.k_V_avg

    #     # self.q_b = self.U_L * (self.T_air - self.T_L)
    #     # self.q_b = 100/(np.pi*((8**2)/4))
    #     # self.q_b = 200000/(np.pi*((76.4**2)/4))


    # * mathematical functions ***************************************************************************************

    def d_i(self, a):
        return np.power(4 * self.V_T / (a * np.pi), 1/3)

    def d_o(self, a):
        return self.St * self.d_i(a)
    
    def A_i(self, a): #* ***************************************************** crece app como a^(2/3)
        return np.pi / 4 * np.square(self.d_i(a))
    
    def Q_bot(self, a):
        return self.q_b * self.A_i(a)
    
    def Q_Ltot0(self, LF0, a):
        return ((4* self.d_o(a) / np.square(self.d_i(a))) * (self.T_air - self.T_L) * 
                ((self.U_L - self.eta_w * self.U_V) * LF0  + self.eta_w * self.U_V)) + self.Q_bot(a)

    def v_z0(self, LF0, a): #* *****************************************************************************crece app linealmente con a
        return (4*self.Q_Ltot0(LF0, a)) / (self.rho_V_avg * np.pi * 
                np.square(self.d_i(a)) * (self.h_V - self.h_L))
    
    def H(self, LF0, a): #* ***************************************************************************** crece app linealmente con a
        return self.rho_V_avg * self.cp_avg * self.v_z0(LF0, a)
    
    def S(self, a):
        return (4 * self.U_V * self.d_o(a)) / np.square(self.d_i(a)) * (1 - self.eta_w)

    def xi_p(self, LF0, a): #* ************************************************ con H >>  4 * k * self.S(a)) crece app linealmente con a
        H = self.H(LF0, a)
        k = self.k_V_avg
        return ((H + np.sqrt(np.square(H) + 4 * k * self.S(a))) / 
                (2 * k))
    
    def xi_m(self, LF0, a):
        H = self.H(LF0, a)
        k = self.k_V_avg
        return ((H - np.sqrt(np.square(H) + 4 * k * self.S(a))) / 
                (2 * k))

    def l_v(self, LF0, a): #* **************************** crece como a^(-2/3)
        return self.V_T * (1 - LF0) / self.A_i(a)
    
    def aa_p(self, LF0, a): #* ***************************** crece como    a * exp(a^(1/3))
        xi_p = self.xi_p(LF0, a)
        return xi_p * np.exp(self.l_v(LF0, a) * xi_p)
        # return xi_p * np.power(xi_p), np.exp(self.l_v(LF0, a))

    def aa_m(self, LF0, a):
        xi_m = self.xi_m(LF0, a)
        return xi_m * np.exp(self.l_v(LF0, a) * xi_m)
        # return xi_m * np.power(xi_m), np.exp(self.l_v(LF0, a))
    
    def c1(self, LF0, a):
        aa_m = self.aa_m(LF0, a)
        aa_p = self.aa_p(LF0, a)


        # numeric fix
        if aa_p > 1e10:
            return (self.T_L - self.T_air) / (1 - (aa_m/aa_p))
        else:
            return aa_p * (self.T_L - self.T_air) / (aa_p - aa_m)
        
    def c2(self, LF0, a):
        aa_m = self.aa_m(LF0, a)    
        aa_p = self.aa_p(LF0, a)

        return -aa_m * (self.T_L - self.T_air) / (aa_p - aa_m)


    def delta_T_V0_old(self, LF0, a): ## version original

        l_v = self.l_v(LF0, a)
        xi_p = self.xi_p(LF0, a)
        xi_m = self.xi_m(LF0, a)

        return (
            -1 / l_v *
            (
                self.c1(LF0, a) / xi_m * (np.exp(xi_m * l_v ) - 1) + 
                self.c2(LF0, a) / xi_p * (np.exp(xi_p * l_v ) - 1)
            )
         
          )
    
    # def delta_T_V0(self, LF0, a): # versión con intento de seno hiperbólico
    #     l_v = self.l_v(LF0, a)
    #     xi_p = self.xi_p(LF0, a)
    #     xi_m = self.xi_m(LF0, a)

    #     arg_p = xi_p * l_v
    #     arg_m = xi_m * l_v

    #     # if np.abs(arg_p) > 700:
    #     #     exp_p = 2 * np.exp(arg_p / 2) * np.sinh(arg_p / 2) / xi_p
    #     # else:
    #     #     exp_p = (np.exp(arg_p ) - 1) / xi_p
    #     exp_p = (np.exp(arg_p ) - 1) / xi_p


    #     return (
    #         -1 / l_v *
    #         (
    #             self.c1(LF0, a) / xi_m * (np.exp(xi_m * l_v ) - 1) + 
    #             self.c2(LF0, a) * exp_p
    #         )
         
    #       )

    def delta_T_V0(self, LF0, a):
        
        l_v = self.l_v(LF0, a)
        H = self.H(LF0, a)
        k_v_avg = self.k_V_avg
        # en esta expresion u = \sqrt(H^2 - 4k_v_avg * S)
        S = self.S(a)
        u = np.sqrt(np.square(H) + 4 * k_v_avg * S)

        # variable para exponencial positiva dependiente de u
        exppu = np.exp(u * l_v / (2 * k_v_avg))
        # variable para exponencial negativa dependiente de u
        expnu = np.exp(-u * l_v / (2 * k_v_avg))

        # variable para exponencial positiva dependiente de H
        exppH = np.exp(H * l_v / (2 * k_v_avg))

        if H * l_v / (2 * k_v_avg) > 700:
            return(
                2 * (self.T_L - self.T_air) / (l_v * S) * 
                (H * u * (np.exp(-l_v * S / (H+u)) - 0.5) - (np.square(H) + 2  * k_v_avg * S) / 2) / 
                (H + u)

            )



        return (2 * (self.T_L - self.T_air) / (l_v * S) * 
                (H * u * (exppH - expnu / 2 - exppu / 2) + (np.square(H) + 2 * k_v_avg * S) / 2 * (expnu - exppu)) / 
                (-H * (expnu - exppu) + u * (expnu + exppu))

        )
    
    # def delta_T_V0(self, LF0, a): #! está MALAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaaaaaaaaaaaaaaaaaaaaaaaa
    #     H = self.H(LF0,a)
    #     k_V_avg = self.k_V_avg
    #     return (

    #         (self.T_air - self.T_L) / self.l_v(LF0, a) * 
    #         (np.exp(H / k_V_avg) / (self.aa_p(LF0, a) - self.aa_m(LF0, a)) * (np.sqrt(np.square(H) + 4 * k_V_avg * self.S(a)) / k_V_avg) - 1)

    #     )

    # def delta_T_V0(self, LF0, a): ## 2nd numerical fix #! AAAAAAAAAAAAAAAAAAAAA erroneo

    #     l_v = self.l_v(LF0, a)
    #     xi_p = self.xi_p(LF0, a)
    #     xi_m = self.xi_m(LF0, a)

    #     # primero chequear aproximaciones numéricas en caso de ser necesarias
    #     if xi_p * l_v > 700:
    #         return -(self.T_air - self.T_L) / (xi_p * l_v)
    #     elif xi_m * l_v > 700:
    #         return -(self.T_air - self.T_L) / (xi_m * l_v)

    #     # fórmula exacta si no hay ningun error numérico
    #     else:
    #         exp_p = np.exp(xi_p * l_v)
    #         exp_m = np.exp(xi_m * l_v)
    #         return (self.T_air - self.T_L) / l_v * (exp_m - exp_p) / (xi_p * exp_p - xi_m * exp_m)

    def delta_T_V0_1(self, LF0, a):
        return self.c1(LF0, a) / self.xi_m(LF0, a) * (np.exp(self.xi_m(LF0, a) * self.l_v(LF0, a) ) - 1)

    # # # nuevamente separar en 2 partes para ver que la segunda explota

    def dtv0_1_a(self, LF0, a):
        return self.c1(LF0, a) / self.xi_m(LF0, a)
    
    #! De aquí en adelante los derivados de deltaT_V0 no tienen sentido, no usar

    def dtv0_1_b(self, LF0, a):
        return (np.exp(self.xi_p(LF0, a) / self.l_v(LF0, a) ) - 1)
    
    def dtv0_1_b_argumento(self, LF0, a):
        return self.xi_p(LF0, a) / self.l_v(LF0, a) 


    def delta_T_V0_2(self, LF0, a):
        return self.c2(LF0, a) / self.xi_m(LF0, a) * (np.exp(self.xi_m(LF0, a) / self.l_v(LF0, a) ) - 1)

    # def C_w(self, LF0, a):
    #     return(
    #         -1/(self.rho_L * (self.h_V - self.h_L)) * (4*self.d_o(a) / np.square(self.d_i(a))) * 
    #         (self.U_L*self.T_air - self.eta_w * self.U_V * self.delta_T_V0(LF0, a))
    #     )

    # def C_w(self, LF0, a):
    #     return(
    #         -1/(self.rho_L * (self.h_V - self.h_L)) * (4*self.d_o(a) / np.square(self.d_i(a))) * 
    #         (self.U_L*(self.T_air - self.T_L) - self.eta_w * self.U_V * self.delta_T_V0(LF0, a))
    #     )

    def C_w(self, LF0, a): # reescritura luego de volver a resolver el modelo
        return(

            -4*self.d_o(a)/(self.rho_L*(self.h_V - self.h_L)*np.square(self.d_i(a))) *
            (
                self.U_L * (self.T_air - self.T_L) - self.eta_w * self.U_V * self.delta_T_V0(LF0, a)
            )

        )
    
    def C_w_1(self, LF0, a):

        return(
            self.U_L * (self.T_air - self.T_L)

        )

    def C_w_2(self, LF0, a):
        return self.eta_w * self.U_V * self.delta_T_V0(LF0, a)

    # def D_w(self, LF0, a):
    #     return(
    #         -1/(self.rho_L * (self.h_V - self.h_L)) * (self.Q_bot(a) + 
    #             (4*self.d_o(a) / np.square(self.d_i(a))) * 
    #         (self.eta_w * self.U_V * self.delta_T_V0(LF0, a) - self.U_L * self.T_L) * self.V_T)
    #     )

    # post cambio U_L*V_T
    # def D_w(self, LF0, a):
    #     return(
    #         -1/(self.rho_L * (self.h_V - self.h_L)) * (self.Q_bot(a) + 
    #             (4*self.d_o(a) / np.square(self.d_i(a))) * 
    #         (self.eta_w * self.U_V * self.delta_T_V0(LF0, a)) * self.V_T)
    #     )
    
    def D_w(self, LF0, a):
        return(

            -1/(self.rho_L*(self.h_V - self.h_L)) * 
            (
                self.Q_bot(a) + 4 * self.eta_w * self.U_V * self.d_o(a) / np.square(self.d_i(a)) * self.delta_T_V0(LF0, a) * self.V_T
            )


        )

    def V_L(self, LF0, a, tf):
        return (
            self.D_w(LF0, a) / self.C_w(LF0, a) * (np.exp(self.C_w(LF0, a)*tf)-1) + 
            LF0 * self.V_T * np.exp(self.C_w(LF0, a) * tf)

        )

    def BOR(self, LF0, a, tf):

        return (

            (1 - np.exp(self.C_w(LF0, a) * tf)) *   
            (1 + self.D_w(LF0, a) / (LF0 * self.C_w(LF0,a) * self.V_T))

            # (1 - np.exp(self.C_w(LF0, a)) ** tf) * 
            # (1 + self.D_w(LF0, a) / (LF0 * self.C_w(LF0,a) * self.V_T))

        ) * 86400 / tf
    
    
    def f(self, arg):
        if arg > 700:
            # return 1 + arg / 2 + arg ** 2 / 6 + arg ** 3 / 24
            sum = 1
            for i in range(1, 6):
                sum += arg ** i / math.factorial(i-1)
            return sum
        
        else:
            return (np.exp(arg) - 1) / arg

    def new_delta_T_V0(self, LF0, a):
        
        arg_p = self.l_v(LF0, a) * self.xi_p(LF0, a)
        arg_m = self.l_v(LF0, a) * self.xi_m(LF0, a)

        return -self.c1(LF0, a) * self.f(arg_p) - self.c2(LF0, a) * self.f(arg_m)
        
    def new_C_w(self, LF0, a): # reescritura luego de volver a resolver el modelo
        return(

            -4*self.d_o(a)/(self.rho_L*(self.h_V - self.h_L)*np.square(self.d_i(a))) *
            (
                self.U_L * (self.T_air - self.T_L) - self.eta_w * self.U_V * self.new_delta_T_V0(LF0, a)
            )

            # -4*self.d_o(a)/(self.rho_L*(self.h_V - self.h_L)*np.square(self.d_i(a))) *
            # (
            #     - self.eta_w * self.U_V * self.new_delta_T_V0(LF0, a)
            # )

        )
    
    def new_D_w(self, LF0, a):
        return(

            -1/(self.rho_L*(self.h_V - self.h_L)) * 
            (
                self.Q_bot(a) + 4 * self.eta_w * self.U_V * self.d_o(a) / np.square(self.d_i(a)) * self.new_delta_T_V0(LF0, a) * self.V_T
            )


        )


    def new_BOR(self, LF0, a, tf):

        l_v = self.l_v(LF0, a)
        xi_p = self.xi_p(LF0, a)

        if l_v * xi_p > 700:

            return (

            (1 - np.exp(-self.new_C_w(LF0, a) * tf)) *   
   
            (1 - l_v * xi_p * np.exp(-l_v * xi_p) / (LF0 * (self.T_L - self.T_air)))

            # (1 - np.exp(self.C_w(LF0, a)) ** tf) * 
            # (1 + self.D_w(LF0, a) / (LF0 * self.C_w(LF0,a) * self.V_T))

            ) * 86400 / tf

        return self.BOR(LF0, a, tf)



    # def f2(self, LF0, a, tf):
    #     return (

    #         # 1 - (self.V_L(LF0, a, tf) / (self.V_T * LF0))
    #         # 1 - (self.V_L(LF0, a, tf) / (self.V_L(LF0, a, 0)))
    #         (self.V_T * LF0 - self.V_L(LF0, a, tf) / (self.V_T * LF0))

    #     ) * 86400 / tf

    # def f3(self, LF0, a, tf):
    #     return (

    #         1 - (self.V_L(LF0, a, tf) / (self.V_T))

    #     ) * 86400 / tf

    # * plotting methods ***************************************************************************************



    # def plot_BOR_fixLF0(self, a_range, LF0_list, tf):
    #     fig, ax = plt.subplots()

    #     # A = np.linspace(a_range[0], a_range[1], 100)
    #     A = a_range

    #     Y = [[self.f(LF0, a, 720 * 3600) for a in A] for LF0 in LF0_list]

    #     for i in range(len(LF0_list)):
    #         plt.plot(A, Y[i], label=f'LF0={LF0_list[i]}')

    #     plt.xlabel('aspect ratio')
    #     plt.ylabel('BOR (%vol/day)')

    #     #TODO Boil of Gas (BOG)  es masa evaporada / tiempo

    #     plt.legend()
    #     plt.show()

        # titulo en ejes

    # def plot_func_fixLF0(self, func, a_range, LF0_list, tf = None, fig_ax = None, show = True):

    #     Tank.dummy_func = func

    #     if fig_ax is None:
    #         fig, ax = plt.subplots()
    #     else:
    #         fig, ax = fig_ax

    #     # A = np.linspace(a_range[0], a_range[1], 100)
    #     A = a_range

    #     if tf is None:
    #         Y = [[self.dummy_func(LF0, a) for a in A] for LF0 in LF0_list]
    #     else:
    #         Y = [[self.dummy_func(LF0, a, tf) for a in A] for LF0 in LF0_list]

    #     extra = ''
    #     if not show: extra = ' 1st'


    #     for i in range(len(LF0_list)):
    #         plt.plot(A, Y[i], label=f'LF0={LF0_list[i]}' + extra)

    #     plt.xlabel('aspect ratio')
    #     plt.ylabel(func.__name__)

    #     #TODO Boil of Gas (BOG)  es masa evaporada / tiempo

    #     extra_tf = ''
    #     if tf is not None: extra_tf += f' | t={tf / 3600} h'

    #     plt.title(f'{func.__name__} for {self.fluid} in tank of volume {self.V_T} m^3' + extra_tf + f', P = {self.P} Pa,\n U_L = U_V = {self.U_L} Wm^-2K^-1, eta_w = {self.eta_w}, q_b = {self.q_b}')

    #     #  U_L, U_V = None, eta_w = 0.8

    #     plt.legend()
    #     if show: plt.show()


    def plot_BOR_fixLF0(self, a_range, LF0_list, tf, show = True, opts = False, tol = 1e-8):
        self.plot_func_fixLF0(Tank.BOR, a_range, LF0_list, tf, show = show, opts = opts, tol = tol)





    def plot_func_fixLF0(self, func, a_range, LF0_list, tf = None, fig_ax = None, show = True, opts = False, tol = 1e-8):

        Tank.dummy_func = func

        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

        # A = np.linspace(a_range[0], a_range[1], 100)
        A = a_range

        if opts: x_opts = []; y_opts = []; a0 = 0.5

        if tf is None:
            Y = [[self.dummy_func(LF0, a) for a in A] for LF0 in LF0_list]

            if opts:
                for LF0 in LF0_list:

                    f = lambda x: self.dummy_func(LF0, x)

                    res = minimize(f, a0, bounds=[(a_range[0], a_range[-1])], tol = tol)

                    x_opts.append(res.x[0])
                    y_opts.append(f(res.x[0]))

        else:
            Y = [[self.dummy_func(LF0, a, tf) for a in A] for LF0 in LF0_list]

            if opts:
                for LF0 in LF0_list:

                    f = lambda x: self.dummy_func(LF0, x, tf)

                    res = minimize(f, a0, bounds=[(a_range[0], a_range[-1])], tol = tol)

                    x_opts.append(res.x[0])
                    y_opts.append(f(res.x[0]))

        extra = ''
        if not show: extra = ' 1st'


        for i in range(len(LF0_list)):
            plt.plot(A, Y[i], label=f'LF0={LF0_list[i]}' + extra)


        if opts:
            plt.scatter(x_opts, y_opts, color='red', zorder=5, marker = 'o')
            plt.plot(x_opts, y_opts, 'r--', label='optimal values')

        plt.xlabel('aspect ratio')
        plt.ylabel(func.__name__)

        #TODO Boil of Gas (BOG)  es masa evaporada / tiempo

        extra_tf = ''
        if tf is not None: extra_tf += f' | t={tf / 3600} h'

        plt.title(f'{func.__name__} for {self.fluid} in tank of volume {self.V_T} m^3' + extra_tf + f', P = {self.P} Pa,\n U_L = U_V = {self.U_L} Wm^-2K^-1, eta_w = {self.eta_w}, q_b = {self.q_b}')

        #  U_L, U_V = None, eta_w = 0.8

        plt.legend()
        if show: plt.show()

        # titulo en ejes



