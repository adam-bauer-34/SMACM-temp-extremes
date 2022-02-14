def getTFluxFull(T, m, param_list):
    # define model parameters 
    C = param_list[0] 
    F = param_list[1] 
    alpha = param_list[2]
    v_L = param_list[3]
    L = param_list[4]
    gamma = param_list[5]
    T_D = param_list[6]
    mu = param_list[7]
    m0 = param_list[8]

    feedback = alpha + L * v_L * gamma * (m + m0) 
    tempdiff = T - T_D 
    flux = C**(-1) * ( F - feedback * tempdiff )

    return flux 

def getMFluxFull(T, m, param_list, precip):
    # define model parameters 
    C = param_list[0] 
    F = param_list[1] 
    alpha = param_list[2]
    v_L = param_list[3]
    L = param_list[4]
    gamma = param_list[5]
    T_D = param_list[6]
    mu = param_list[7]

    tempdiff = T - T_D 
    flux = precip - v_L * gamma * m * tempdiff * mu**(-1)

    return flux

def getMFlux1D(m, param_list, precip):
    # define model parameters 
    C = param_list[0] 
    F = param_list[1] 
    alpha = param_list[2]
    v_L = param_list[3]
    L = param_list[4]
    gamma = param_list[5]
    T_D = param_list[6]
    mu = param_list[7]
    m0 = param_list[8]

    denom = mu * (alpha + v_L * L * gamma * (m + m0))
    flux = precip - F * v_L * gamma * m * denom**(-1)

    return flux 