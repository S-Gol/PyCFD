import numpy as np
import cv2
from scipy import ndimage
import FluidSim



class FluidsRenderer:
    def __init__(self, sim, upscaleMult = 5, arrowSpacing=5, 
    dyePoints = [[0,0,0]], forcedVel=[], windowName = "PyCFD",
    kernelSize = 3):
        self.sim = sim
        self.upscaleMult = upscaleMult
        self.arrowSpacing = arrowSpacing
        self.dyePoints = dyePoints

        self.fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        self.forcedVel = forcedVel

        self.imageSize = [sim.nx*upscaleMult, sim.ny * upscaleMult]
        self.downscaledIndices = (sim.indices * upscaleMult)[:,::arrowSpacing,::arrowSpacing]
        self.upscaledIndices = (np.indices(self.imageSize)/self.upscaleMult).astype(int)
        print(self.upscaledIndices)
        self.rowsUpsc = self.downscaledIndices[1,:,:].flatten()
        self.colsUpsc = self.downscaledIndices[0,:,:].flatten()

        self.rows = (self.rowsUpsc / upscaleMult).astype(np.int32)
        self.cols = (self.colsUpsc / upscaleMult).astype(np.int32)
        
        self.arrowOrigins = [[self.rowsUpsc[i], self.colsUpsc[i]] for i in range(len(self.rows))]
        self.window_name=windowName

        cv2.namedWindow(self.window_name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.imageSize[0],self.imageSize[1])
        self.smoothKernel =  np.ones([kernelSize, kernelSize]) / (kernelSize*kernelSize)
        self.dye = np.zeros(self.imageSize)


    def startContinousRender(self):
        self.video = cv2.VideoWriter('Output.avi', self.fourcc,30, [self.imageSize[0], self.imageSize[1]], 0)
        while True:
            #Simulation steps
            for vel in self.forcedVel:
                x,y,u,v = vel
                self.sim.u[y,x] = u
                self.sim.v[y,x] = v
            for d in self.dyePoints:
                x,y,r = d
                self.dye[x*self.upscaleMult-5:x*self.upscaleMult+5, y*self.upscaleMult-5:y*self.upscaleMult+5]+=r
            self.sim.timestep()

            uUpscaled = cv2.resize(u, self.imageSize)
            vUpscaled = cv2.resize(v, self.imageSize)

            self.dye = sim.advectField(self.dye, u=uUpscaled, v=vUpscaled, scaledIndices=self.upscaledIndices)
            #self.dye = ndimage.convolve(self.dye, self.smoothKernel)

            scaledImage = (self.dye).astype(np.uint8)

            u = (sim.u[self.rows, self.cols].flatten()*self.arrowSpacing*5).astype(np.int32)
            v = (sim.v[self.rows, self.cols].flatten()*self.arrowSpacing*5).astype(np.int32)
            endX = self.colsUpsc+u
            endY = self.rowsUpsc+v

            for i in range(len(self.rows)):
                cv2.arrowedLine(scaledImage, [self.colsUpsc[i], self.rowsUpsc[i]], [endX[i], endY[i]], 255)

            
            self.video.write(scaledImage)
            cv2.imshow(self.window_name, scaledImage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()


xSize = 50
ySize = 50

walls = np.zeros([xSize, ySize], dtype=bool)
walls[0:2,:]=True
walls[-2:,:]=True

r2=1000
cx = 100
cy = 100
xs = []
ys = []
for x in range(xSize):
    for y in range(ySize):
        r = (x-cx)**2+(y-cy)**2
        if r < r2:
            xs.append(x)
            ys.append(y)


velocities = [[5,y,1,0] for y in range(0,ySize)]
walls[xs, ys] = True
sim = FluidSim.FluidSim(xSize,ySize,2,2, dt=0.00001, walls=walls, nu=0.1)

dyePoints = [[20,20,5]]

renderer = FluidsRenderer(sim, forcedVel=velocities, dyePoints=dyePoints)
renderer.startContinousRender()
