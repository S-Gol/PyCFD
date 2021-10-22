from numba.np.ufunc import parallel
import numpy as np
from numba import njit

class FluidSim:
    """
    Class containing the simulation space for a fluid. Manages fluid properties + simulation. 
    """
    def __init__(self, nx, ny, xSize, ySize, rho = 1, nu = 0.1, dt = 0.001, nit = 50, boundaryDP = None, boundary0Vel = None, walls=None):

        #Boundary conditions
        if boundaryDP is None:
            self.dp = np.ones([nx,ny])
        else:
            self.dp = (~boundaryDP).astype(int)
        if boundary0Vel is None: 
            self.velBoundary = np.ones([nx, ny])
        else: 
            self.velBoundary = (~boundary0Vel).astype(int)

        if walls is not None:
            self.velBoundary *= (~walls).astype(int)
            self.dp *= (~walls).astype(int)


        #Simulation variables 
        self.nx = nx
        self.ny = ny
        self.c = 1
        self.dx = xSize/(nx - 1)
        self.dy = ySize/(ny - 1)
        self.t = 0.0
        self.nit = nit
        self.indices = np.indices([nx, ny])
        self.xIndices = self.indices[0,:,:].flatten()
        self.yIndices = self.indices[1,:,:].flatten()

        #Simulation space dimensions
        self.x = np.linspace(0, 2, nx)
        self.y = np.linspace(0, 2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        #Fluid properties
        self.rho = rho
        self.nu = nu
        self.dt = dt

        #Simulation field arrays 
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx)) 
        self.b = np.zeros((ny, nx))
    @njit(parallel=True)
    def build_up_b(b, rho, dt, u, v, dx, dy):
        """
        Used internally to calculate intermediate b-value for poisson pressure equation. DO NOT USE.  
        """
        
        b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
        return b
    @njit(parallel=True)
    def pressure_poisson(p, dx, dy, b, nit):
        """
        Used internally to find the pressure at a point from the b-matrix intermediate
        values. DO NOT USE. 
        """
        pn = np.empty_like(p)
        pn = p.copy()
        
        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                            (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                            (2 * (dx**2 + dy**2)) -
                            dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                            b[1:-1,1:-1])


        return p
    
    def timestep(self):
        """Iterate through time, updating the simulation"""
        self.t += self.dt
        u = self.u
        v = self.v 
        dt = self.dt
        dx = self.dx
        dy = self.dy 
        p = self.p 
        rho = self.rho
        nu = self.nu
        
        b = np.zeros((self.ny, self.nx))
        

        b = FluidSim.build_up_b(b, rho, dt, u, v, dx, dy)
        p = FluidSim.pressure_poisson(p, dx, dy, b, self.nit)
        
        u, v = FluidSim.velocity_calcs(self.u, self.v, p, self.dt, self.dx, self.dy, self.nu, self.rho, self.dp)


        u = u * self.velBoundary
        v = v * self.velBoundary

        self.u = u
        self.v = v
        self.p = p

    def advectField(self, c, u=None, v=None, scaledIndices = None):
        """
        Advect property c across the velocity field. 
        """
        advected = c.copy()
        if u is None:
            u=self.u
        if v is None:
            v=self.v
        if scaledIndices is None:
            offsets = self.indices.copy()
        else:
            offsets = scaledIndices.copy()
        xIndices = offsets[0,:,0]
        yIndices = offsets[1,0,:]

        offsetX = u*10 / self.dx 
        offsetY = v*10 / self.dy 

        offsets[0,:,:] -= offsetX.astype(np.int32).transpose()
        offsets[1,:,:] -= offsetY.astype(np.int32).transpose()
        
        offsets[0,:,:] = np.clip(offsets[0,:,:], 0, advected.shape[0]-1)
        offsets[1,:,:] = np.clip(offsets[1,:,:], 0, advected.shape[1]-1)

        advected[yIndices,xIndices] = c[offsets[1,xIndices,yIndices], offsets[0,xIndices,yIndices]]
        return advected
    
    @njit(parallel=True)
    def velocity_calcs(u, v, p, dt, dx, dy, nu, rho, dp):
        """
        Used internally for JIT, DO NOT USE. 
        """
        un = u.copy()
        vn = v.copy()
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                un[1:-1, 1:-1] * dt / dx *
                (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                vn[1:-1, 1:-1] * dt / dy *
                (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])*(dp[1:-1, 1:-1]) +
                nu * (dt / dx**2 *
                (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                dt / dy**2 *
                (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])*(dp[1:-1, 1:-1]) +
                        nu * (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
        return u, v
