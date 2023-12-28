import torch
from torch import nn
# import torch.nn.functional as F
# from torch.autograd import Function

from . import _C

def np_multiply(h_A, h_B):
    return _C.np_multiply(h_A, h_B)

def tensor_multiply(h_A, h_B):
    return _C.tensor_multiply(h_A, h_B)
# class DiffSP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         dsp = _C.CUDSP()

#         class DSPFunction(Function):
#             @staticmethod
#             def forward(ctx, points, triangles, scale, threshold):
#                 verts, faces, verts_occ, verts_map = dsp.forward(points, triangles, scale, threshold)
#                 ctx.points = points
#                 ctx.triangles = triangles
#                 return verts, faces, verts_occ, verts_map

#         self.func = DSPFunction

#     def forward(self, points, triangles, threshold=0.001, iter=1000):
#         verts = points
#         faces = triangles
#         scale = max(max(verts[:,0].max()-verts[:,0].min(), verts[:,1].max()-verts[:,1].min()), verts[:,2].max()-verts[:,2].min())
#         for it in range(iter):
#             num_faces = faces.shape[0]
#             verts, faces, verts_occ, verts_map = self.func.apply(verts, faces, scale, threshold)
#             # applied is often used to each element of a collection
#             verts = verts[verts_occ.view(-1).bool()]
#             faces = faces[faces[:, 0] >= 0]
#             faces[:,0] = verts_map[faces[:,0].long()].view(-1) # make it continous in memory
#             faces[:,1] = verts_map[faces[:,1].long()].view(-1)
#             faces[:,2] = verts_map[faces[:,2].long()].view(-1)
#             if faces.shape[0] == num_faces:
#                 print("Converged at iteration {}".format(it))
#                 break

        
#         return verts, faces