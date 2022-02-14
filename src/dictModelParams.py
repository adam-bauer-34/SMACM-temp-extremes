"""
Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinios.edu

A set of dictionaries with model parameters calculated using ERA5 data. These
will be imported to HeatwaveFreq_main.py.
"""

SGP_dict = {'alpha_s': 4, 'alpha_r': 8, 'nu': 0.03, 'mu': 40, 'm_0': 0.1,
            'F_mean': 226.96, 'F_std': 44.53, 'Td_mean': 291.55, 'Td_std':
            2.75, 'omega': 3.5, 'p_0': 0.375, 'p_scale': 0.15}
Witchita_dict = {'alpha_s': 4, 'alpha_r': 7, 'nu': 0.02, 'mu': 40, 'm_0': 0.2,
                 'F_mean': 226.44, 'F_std': 47.42, 'Td_mean': 291.29, 'Td_std':
                 2.92, 'omega': 3.5, 'p_0': 0.375, 'p_scale': 0.15}
Dallas_dict = {'alpha_s': 4, 'alpha_r': 7, 'nu': 0.018, 'mu': 38, 'm_0': 0.3,
               'F_mean': 226.84, 'F_std': 42.69, 'Td_mean': 293.42, 'Td_std':
               2.21, 'omega': 3.5, 'p_0': 0.375, 'p_scale': 0.15}
NY_dict = {'alpha_s': 2, 'alpha_r': 7, 'nu': 0.013, 'mu': 38, 'm_0': 0.72,
           'F_mean': 216.44, 'F_std': 67.10, 'Td_mean': 290.82, 'Td_std': 3.51,
           'omega': 3.5, 'p_0': 0.375, 'p_scale': 0.15}
SEA_dict = {'alpha_s': 6, 'alpha_r': 7, 'nu': 0.011, 'mu': 35, 'm_0': 0.68,
           'F_mean': 216.16, 'F_std': 66.79, 'Td_mean': 285.30, 'Td_std': 2.13,
           'omega': 3.5, 'p_0': 0.375, 'p_scale': 0.15}
ATL_dict = {'alpha_s': 6.5, 'alpha_r': 7, 'nu': 0.03, 'mu': 35, 'm_0': 0.2,
           'F_mean': 204.61, 'F_std': 49.87, 'Td_mean': 292.72, 'Td_std': 2.67,
           'omega': 3.5, 'p_0': 0.375, 'p_scale': 0.15}

loc_param_dict = {'SGP': SGP_dict, 'WIT': Witchita_dict, 'DAL':
                  Dallas_dict, 'NY': NY_dict, 'SEA': SEA_dict, 'ATL': ATL_dict}
