'''
   Polya Conjecture Gradient Descent Solver
   See c++ version in the MFEM library for more detail 
   How to run:
      mpirun -np 2 python <arguments>
  
'''

import sys
from os.path import expanduser, join, dirname
import numpy as np

from mfem.common.arg_parser import ArgParser
import mfem.par as mfem
from mpi4py import MPI


num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank


parser = ArgParser(description='Ex11 ')
parser.add_argument('-m', '--mesh',
                    default='star.mesh',
                    action='store', type=str,
                    help='Mesh file to use.')
parser.add_argument('-rs', '--refine-serial',
                    default=2,
                    action='store', type=int,
                    help="Number of times to refine the mesh uniformly in serial.")

parser.add_argument('-rp', '--refine-parallel',
                    default=1,
                    action='store', type=int,
                    help="Number of times to refine the mesh uniformly in parallel.")
parser.add_argument('-o', '--order',
                    action='store', default=1, type=int,
                    help=("Finite element order (polynomial degree) or -1 for isoparametric space."))
parser.add_argument("-n", "--num-eigs",
                    action='store', default=1, type=int,
                    help="Number of desired eigenmodes.")
parser.add_argument("-sp", "--strumpack",
                    action='store_true', default=False,
                    help="Use the STRUMPACK Solver.")
parser.add_argument('-vis', '--visualization',
                    action='store_true', default=True,
                    help='Enable GLVis visualization')
args = parser.parse_args()


ser_ref_levels = args.refine_serial
par_ref_levels = args.refine_parallel
order = args.order
nev = args.num_eigs
visualization = args.visualization
use_strumpack = args.strumpack
if (myid == 0):
    parser.print_options(args)

device = mfem.Device('cpu')
if myid == 0:
    device.Print()

# 1. Initialize visualization socket
if (visualization):
    mode_sock = mfem.socketstream("localhost", 19916)
    mode_sock.precision(8)

# TODO Add to CLI Parameters
# Various Temp Global Variables
maxIterations = 100
numVert = 6
numElem = 5
numBdrElem = 5

vertices = np.array([
            [0.0,0.0],        
            [2.0,0.0],
            [0.31,0.95],
            [-0.81,0.59],
            [-0.81,-0.59],
            [0.31,-0.95]
            ])

# Main Loop
for numIter in range(maxIterations):
    # Generate Mesh From initial Vertices
    mesh = mfem.Mesh(2, numVert, numElem, numBdrElem)
    mesh.AddVertex(vertices[0,0], vertices[0,1])
    mesh.AddVertex(vertices[1,0], vertices[1,1])
    mesh.AddVertex(vertices[2,0], vertices[2,1])
    mesh.AddVertex(vertices[3,0], vertices[3,1])
    mesh.AddVertex(vertices[4,0], vertices[4,1])
    mesh.AddVertex(vertices[5,0], vertices[5,1])

    # Add Elements
    mesh.AddTriangle(0,1,2)
    mesh.AddTriangle(0,2,3)
    mesh.AddTriangle(0,3,4)
    mesh.AddTriangle(0,4,5)
    mesh.AddTriangle(0,5,1)

    # Add Boundary
    mesh.AddBdrSegment(1,2)
    mesh.AddBdrSegment(2,3)
    mesh.AddBdrSegment(3,4)
    mesh.AddBdrSegment(4,5)
    mesh.AddBdrSegment(5,1)

    # Finalize the Mesh
    mesh.SetAttributes()

