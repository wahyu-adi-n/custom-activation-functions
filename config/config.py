import utils.custom_afs as custom_afs
import torch.nn as nn
import torch

afs_dict = {
                'ReLU' : nn.ReLU(),
                'NegReLU' : custom_afs.NegReLU(),
                'Dslope_0.25' : custom_afs.DSlopeReLU(0.25),
                # 'Dslope_0.75' : custom_afs.DSlopeReLU(0.75),
                # 'Dslope_1.25' : custom_afs.DSlopeReLU(1.25),
                'DiffYRelu_0.5' : custom_afs.Diff_Y_ReLU(0.5),
                # 'DiffYRelu_1.0' : custom_afs.Diff_Y_ReLU(1.0),
                'PosHill_0.25' : custom_afs.Pos_Hill_V1(0.25),
                # 'PosHill_0.5' : custom_afs.Pos_Hill_V1(0.5),
                # 'PosHill_0.75' : custom_afs.Pos_Hill_V1(0.75),
                'SmallNeg_0.1' : custom_afs.Small_Neg(0.1),
                # 'SmallNeg_0.5' : custom_afs.Small_Neg(0.5),
                'Pos_Hill_V2' : custom_afs.Pos_Hill_V2((-1,0), (0,1), (1,0)),
                # 'Pos_Hill_V2_s' : custom_afs.Pos_Hill_V2((-0.5,0),(0,1.25),(0.5,0)),
                # 'Pos_Hill_V2_xs' : custom_afs.Pos_Hill_V2((-0.25,0),(0,0.75),(0.25,0)),
                'Pos_Hill_V3' : custom_afs.Pos_Hill_V3((-1,0), (0,1), (1,0)),
                # 'Pos_Hill_V3_s' : custom_afs.Pos_Hill_V3((-0.5,0), (0,1.25), (0.5,0)),
                # 'Pos_Hill_V3_xs' : custom_afs.Pos_Hill_V3((-0.25,0), (0,0.75), (0.25,0)),
                'Double_Hill' : custom_afs.Double_Hill((-2,0), (-1,1), (0,0), (1,1), (2,0)),
                # 'Double_Hill_s' : custom_afs.Double_Hill((-1,0), (-0.5,1), (0,0), (0.5,1), (1,0)),
                # 'Double_Hill_xs' : custom_afs.Double_Hill((-0.75,0), (-0.4,0.9), (0,0), (0.4,0.9), (0.75,0)),
                'Val_Hill' : custom_afs.Val_Hill((-2,0), (-1,-1), (1,1), (2,0)),
                # 'Val_Hill_s' : custom_afs.Val_Hill((-1,0), (-0.5,-0.9), (0.5,0.9), (1,0)),
                # 'Val_Hill_xs' : custom_afs.Val_Hill((-0.75,0), (-0.25,-0.75), (0.25,0.75), (0.75,0))
            }

device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 3
classes = ['normal', 'pneumonia']