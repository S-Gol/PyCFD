from numba.np.ufunc import parallel
import numpy as np
from numba import njit

class FluidSim:
    """
    Class containing the simulation space for a fluid. Manages fluid properties + simulation. 
    """
    def __init__(self, nx, ny, xSize, ySize, rho = 1, nu = 0.1, dt = 0.001, nit = 50):

        #Simulation variables 
        self.nx = nx
        self.ny = ny
        self.c = 1
        self.dx = 2/(nx - 1)
        self.dy = 2/(ny - 1)
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
        Used to calculate intermediate b-value for poisson pressure equation. 
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
        Iterative method used to find the pressure at a point from the b-matrix intermediate
        values. 
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

            #TODO apply proper boundary conditions from input
            p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
            p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
            p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
            p[-1, :] = 0        # p = 0 at y = 2
            
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
        
        u, v = FluidSim.velocity_calcs(self.u, self.v, p, self.dt, self.dx, self.dy, self.nu, self.rho )

        #TODO add better boundary conditions here
        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 1    # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
        
        
        self.u = u
        self.v = v
        self.p = p

    def advectField(self, c,u,v):
        advected = c.copy()

        offsetX = u  / self.dx
        offsetY = v  / self.dy
        offsets = self.indices.copy()

        offsets[0,:,:] -= offsetX.astype(np.int32).transpose()
        offsets[1,:,:] -= offsetY.astype(np.int32).transpose()
        
        offsets[0,:,:] = np.clip(offsets[0,:,:], 0, self.nx-1)
        offsets[1,:,:] = np.clip(offsets[1,:,:], 0, self.ny-1)

        advected[self.yIndices,self.xIndices] = c[offsets[1,self.xIndices,self.yIndices], offsets[0,self.xIndices,self.yIndices]]

        return advected
    @njit(parallel=True)
    def velocity_calcs(u, v, p, dt, dx, dy, nu, rho):
        un = u.copy()
        vn = v.copy()
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                un[1:-1, 1:-1] * dt / dx *
                (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                vn[1:-1, 1:-1] * dt / dy *
                (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                nu * (dt / dx**2 *
                (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                dt / dy**2 *
                (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
        return u, v
