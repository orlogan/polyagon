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

# Initialize visualization socket
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

    # 0. Update Mesh
    # NOTE: Very simple update example just for testing
    stepSize = 0.01
    vertices[1,0] = vertices[1,0] - stepSize


    # 1. Generate Mesh From initial Vertices
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
    dim = mesh.Dimension()

    # 2. Refine the serial mesh on all processors to 
    # increase the resolution. Uses `ref_levels` var
    for i in range(ser_ref_levels):
        mesh.UniformRefinement()

    # 3. Define a parallel mesh by partitioning the
    # serial mesh defined above
    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    del mesh
    for i in range(par_ref_levels):
        pmesh.UniformRefinement()

    # 4. Define a parallel fem space on the parallel
    # mesh. 
    # Note: We use cont. Lagrange fem of the order
    # specified in the CLI args.
    # If order < 1, we use an isoparametric space
    #
    # NOTE: The Original .edp file uses P1 instead
    # of H1 as its fem method.
    if order > 0:
        fec = mfem.H1_FECollection(order, dim)
    elif pmesh.GetNodes():
        fec = pmesh.GetNodes().OwnFEC()
    else:
        fec = mfem.H1_FECollection(1, dim)

    fespace = mfem.ParFiniteElementSpace(pmesh, fec)
    fe_size = fespace.GlobalTrueVSize()

    if (myid == 0):
        print('Number of unknowns: ' + str(fe_size))
  

    # 5. Create the parallel bilinear forms a(,) and 
    # m(,) on the fem space.
    # 
    # a => The Laplacian Operator
    # m => A simple mass matrix needed to solve the 
    # generalized eigenvalue problem below
    #
    # Note: The Dirichlet Boundary Conditions are 
    # implemented using elemination with special
    # values on the diagonal to shift the 
    # --Dirichlet Eigenvalues-- out of the 
    # computational range
    #
    # TODO: Compare this formulation with the similar
    # one used in the non-python files
    one = mfem.ConstantCoefficient(1.0)

    ess_bdr = mfem.intArray()
    if pmesh.bdr_attributes.Size() != 0:
        ess_bdr.SetSize(pmesh.bdr_attributes.Max())
        ess_bdr.Assign(1)

    a = mfem.ParBilinearForm(fespace)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    # TODO: Figure out what a MassIntegrator is

    if pmesh.bdr_attributes.Size() == 0:
        # Add a mass term if the mesh has no boundary, e.g. periodic mesh or
        # closed surface.
        a.AddDomainIntegrator(mfem.MassIntegrator(one))

    a.Assemble()
    a.EliminateEssentialBCDiag(ess_bdr, 1.0)
    a.Finalize()


    m = mfem.ParBilinearForm(fespace)
    # TODO: What is a DomainIntegrator
    m.AddDomainIntegrator(mfem.MassIntegrator(one))
    m.Assemble()

    # TODO: What does This mean?
    # Shift the eigenvalue corresponding to eliminated dofs to a large value
    m.EliminateEssentialBCDiag(ess_bdr,  3.0e-300)
    m.Finalize()

    A = a.ParallelAssemble()
    M = m.ParallelAssemble()

    if use_strumpack:
        import mfem.par.strumpack as strmpk
        Arow = strmpk.STRUMPACKRowLocMatrix(A)

    # 6. Define and Configure the LOBPCG eigensolver
    # and the BoomerAMG preconditioner, the latter
    # of which will be used by LOBPCG when solving A.
    # 
    # Set the matrices which define the generalized 
    # eigenproblem 
    #
    # A x = lambda M x
    #
    # NOTE: PyMFEM doesn't support SuperLU
    if use_strumpack:
        args = ["--sp_hss_min_sep_size", "128", "--sp_enable_hss"]
        strumpack = strmpk.STRUMPACKSolver(args, MPI.COMM_WORLD)
        strumpack.SetPrintFactorStatistics(True)
        strumpack.SetPrintSolveStatistics(False)
        strumpack.SetKrylovSolver(strmpk.KrylovSolver_DIRECT)
        strumpack.SetReorderingStrategy(strmpk.ReorderingStrategy_METIS)
        strumpack.SetMC64Job(strmpk.MC64Job_NONE)
        # strumpack.SetSymmetricPattern(True)
        strumpack.SetOperator(Arow)
        strumpack.SetFromCommandLine()
        precond = strumpack
    else:
        amg = mfem.HypreBoomerAMG(A)
        amg.SetPrintLevel(0)
        precond = amg

    # NOTE: This seems to be a good outline for 
    # the last few TODOs above
    lobpcg = mfem.HypreLOBPCG(MPI.COMM_WORLD)
    lobpcg.SetNumModes(nev)
    lobpcg.SetPreconditioner(precond)
    lobpcg.SetMaxIter(200)
    lobpcg.SetTol(1e-8)
    lobpcg.SetPrecondUsageMode(1)
    lobpcg.SetPrintLevel(1)
    lobpcg.SetMassMatrix(M)
    lobpcg.SetOperator(A)

    # 7. Compute the eigenmodes and extract the array
    # of the eigenvalues.
    # Define a parallel grid function to represent 
    # each of the eigenmodes returned by the solver.
    eigenvalues = mfem.doubleArray()
    lobpcg.Solve()
    lobpcg.GetEigenvalues(eigenvalues)
    x = mfem.ParGridFunction(fespace)

    #
    #   MAIN TODO: Write Gradient Descent using
    #   the derivative of the eigenvalues wrt verts
    #
    # NOTE: The general outline is in the edp file, but I need to 
    # Cross check with the derivative definitions

    


    # 8. Save the refined mesh and modes in parallel
    # viewed later using GLVis:
    # "glvis -np <np> -m mesh -g mode"
    smyid = '{:0>6d}'.format(myid)
    mesh_name = "solutions/mesh."+smyid
    pmesh.Print(mesh_name, 8)

    for i in range(nev):
        x.Assign(lobpcg.GetEigenvector(i))
        sol_name = "solutions/mode_"+str(i).zfill(2)+"."+smyid
        x.Save(sol_name, 8)
    
    # 9. Send the solution via socket to a local GLVis
    # server
    if (visualization):
        for i in range(nev):
            if (myid == 0):
                print("Eigenmode " + str(i+1) + '/' + str(nev) +
                      ", Lambda = " + str(eigenvalues[i]))

            # convert eigenvector from HypreParVector to ParGridFunction
            x.Assign(lobpcg.GetEigenvector(i))

            mode_sock.send_text("parallel " + str(num_procs) + " " + str(myid))
            mode_sock.send_solution(pmesh,   x)
            mode_sock.send_text("window_title 'Eigenmode " + str(i+1) + '/' +
                                str(nev) + ", Lambda = " + str(eigenvalues[i]) + "'")

# Close GLVis socket after main loop finished running
if (visualization):
    mode_sock.close()
