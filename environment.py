import numpy as np
import constants as cs

class Environment:

    def setUnits(self):
        self.wmin_bt = np.array(self.wmin_bt) * cs.CM2FS
        self.wmax_bt = np.array(self.wmax_bt) * cs.CM2FS
        self.dw_bt   = np.array(self.dw_bt  ) * cs.CM2FS

    def __init__(self, id = 1):

        self.init_flag = 0
        self.id = id

        self.T        = [] # K
        self.T_offset = [] # K
        # self.N_bt_mtp = []

        self.thermalization = []
        self.scat_interval  = [] # fs
        self.scat_rate      = []   # fs-1
        
        self.wmin_bt = [] # cm-1
        self.wmax_bt = []  # cm-1
        self.dw_bt   = []   # cm-1

    def initialize(self):
        self.setUnits()
        self.w_bt = np.arange(self.wmin_bt, self.wmax_bt+1e-5, self.dw_bt) # cm-1
        self.N_coh = len(self.w_bt) 
        self.init_flag = 1