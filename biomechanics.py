from cmath import pi


def WinterTable(M,H):
# input is body mass in kg and body height in m
#
# The function is based on the anthropometric table as publishe in
# 'Biomechanics and Motor Control of Human Movement', second Ed. (1990) by David A. Winter
#
       
        m_F = 2* 0.0145 * M
        m_L = 2* (0.0465+0.1) * M # relative mass of shank+thigh
        m_T = 0.678 * M

        l_F = 0.039 * H # vertical length...
        l_L = 0.53 * H - l_F
        l_T = (0.87-0.53) * H # heigth of glenohumeral joint above hip

        h_L = 0.553 * l_L
        h_T = 0.626 * (0.818-0.53) * H # heigth trunk com above hip
        h_com = (h_L*m_L + (l_L+h_T)*m_T) / (m_L+m_T) #com height above ankle

        # Moment of inertia for rotation about segment COM
        J_Fc = m_F * (0.475*l_F)**2
        J_Lc = m_L * (0.326*l_L)**2
        J_Tc = m_T * (0.496*l_T)**2

        # Moment of inertia for rotation about joints
        J_L = J_Lc + m_L*h_L**2
        J_T = J_Tc + m_T*h_T**2
        J_B = J_Lc + J_Tc + m_L*h_L**2 + m_T*(l_L + h_T)**2
        
        anthro = {
                'H': H,
                'M': M,
                'J': J_B,
                'mgh': (M-m_F)*9.81*h_com,
                'sm': {
                        'description': 'segment mass',
                        'foot': m_F,
                        'legs': m_L,
                        'trunk': m_T
                },
                'sl': {
                        'description': 'segment length',
                        'foot': l_F,
                                'leg': l_L,
                        'trunk': l_T
                },
                'smh': {
                        'description': 'segment com height above ankle joints',
                        'legs': h_L,
                        'trunk': h_T,
                        'body': h_com
                },
                'Jcom': {
                        'description': 'moment of inertia for rotation around segment center of mass',
                        'foot': J_Fc,
                        'leg': J_Lc,
                        'trunk': J_Tc
                },
                'Jj': {
                        'description': 'moment of inertia for rotation around proximal joint',
                        'legs': J_L,
                        'trunk': J_T,
                        'body': J_B
                },
                'eyesAboveGround': 0.936 * H,
                'shoulderAboveAnkle': (0.818-0.039) * H,
        }
        return anthro



def get_com(df,resp_name):
        import numpy as np
        import pandas as pd

        h_hmd = np.mean(df['head_ty'])
        h_sm  = np.mean(df['sho_ty'])
        h_hm  = np.mean(df['hip_ty'])

        H = h_hmd
        M = 1 #normalized since absolute mass does not play a role.
        wt = WinterTable(M,H) # needs to be in (m) for correct moment of inertia calculations
    
        if resp_name=='com_tz' or resp_name=='com_tap':
                swdir = 'tz'
                angle = False
        elif resp_name=='com_rz' or resp_name=='com_rap':
                swdir = 'tz'
                angle = True
        elif resp_name=='com_tx' or resp_name=='com_tml':
                swdir = 'tx'
                angle = False
        elif resp_name=='com_rx' or resp_name=='com_rml':
                swdir = 'tx'
                angle = True
        else: 
                print('get_com input not recognized; should be com_tap/_rap/_tml/_rml')
                return

        sho = df['sho_'+swdir].to_numpy()
        hip = df['hip_'+swdir].to_numpy()

        hip = hip - np.mean(hip)
        sho = sho - np.mean(sho)
        
        hT = wt['smh']['trunk']
        hL = wt['smh']['legs']
        mT = wt['sm']['trunk']
        mL = wt['sm']['legs']

        comL = hip * hL / h_hm
    
#     % movement of trunk above hip is taken relative to hip marker (i.e. hip
#     % marker is used as approximate centre of rotation for the hip joint).
        comT = hip + ((sho - hip) * hT / (h_sm - h_hm))
        
        comB = ( comL * mL + comT * mT ) / (mL+mT)
    
        if angle:
                out = np.arcsin(comB / (wt['smh']['body'])) *180/pi
        else:
                out = comB

        df[resp_name] = out

        return df




