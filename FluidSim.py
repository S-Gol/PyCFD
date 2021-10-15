import numpy
import numba

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

        #Simulation space dimensions
        self.x = numpy.linspace(0, 2, nx)
        self.y = numpy.linspace(0, 2, ny)
        self.X, self.Y = numpy.meshgrid(self.x, self.y)

        #Fluid properties
        self.rho = rho
        self.nu = nu
        self.dt = 0.001

        #Simulation field arrays 
        self.u = numpy.zeros((ny, nx))
        self.v = numpy.zeros((ny, nx))
        self.p = numpy.zeros((ny, nx)) 
        self.b = numpy.zeros((ny, nx))

    def build_up_b(self, b, rho, dt, u, v, dx, dy):
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

    def pressure_poisson(self, p, dx, dy, b):
        """
        Iterative method used to find the pressure at a point from the b-matrix intermediate
        values. 
        """
        pn = numpy.empty_like(p)
        pn = p.copy()
        
        for q in range(self.nit):
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
        self.u, self.v, self.p = self.cavity_flow()

    def cavity_flow(self):
        u = self.u
        v = self.v 
        dt = self.dt
        dx = self.dx
        dy = self.dy 
        p = self.p 
        rho = self.rho
        nu = self.nu
        
        
        un = numpy.empty_like(u)
        vn = numpy.empty_like(v)
        b = numpy.zeros((self.ny, self.nx))
        
        un = u.copy()
        vn = v.copy()
        
        b = self.build_up_b(b, rho, dt, u, v, dx, dy)
        p = self.pressure_poisson(p, dx, dy, b)
        
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

        #TODO add better boundary conditions here
        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 1    # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
        
        
        return u, v, p