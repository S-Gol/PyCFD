import numpy as np
import cv2
from scipy import ndimage
import FluidSim
from lic import *


class FluidsRenderer:
    def __init__(self, sim, upscaleMult = 5, arrowSpacing=5, 
    forcedVel=[], windowName = "PyCFD",
    kernelSize = 3):
        self.sim = sim
        self.upscaleMult = upscaleMult
        self.arrowSpacing = arrowSpacing

        self.fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        self.forcedVel = forcedVel

        self.imageSize = [sim.nx*upscaleMult, sim.ny * upscaleMult]
        self.downscaledIndices = (sim.indices * upscaleMult)[:,::arrowSpacing,::arrowSpacing]
        self.upscaledIndices = (np.indices(self.imageSize)/self.upscaleMult).astype(int)
        self.rowsUpsc = self.downscaledIndices[1,:,:].flatten()
        self.colsUpsc = self.downscaledIndices[0,:,:].flatten()

        self.rows = (self.rowsUpsc / upscaleMult).astype(np.int32)
        self.cols = (self.colsUpsc / upscaleMult).astype(np.int32)
        
        self.arrowOrigins = [[self.rowsUpsc[i], self.colsUpsc[i]] for i in range(len(self.rows))]
        self.window_name=windowName

        cv2.namedWindow(self.window_name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.imageSize[0],self.imageSize[1])
        self.smoothKernel =  np.ones([kernelSize, kernelSize]) / (kernelSize*kernelSize)
        self.baseImage = np.zeros(self.imageSize)
        self.LICNoise = np.random.rand(*(self.imageSize))*cv2.resize(self.sim.velBoundary, self.imageSize)
        self.LICBase = np.zeros([self.imageSize[0], self.imageSize[1],2])
    def startContinousRender(self):
        self.video = cv2.VideoWriter('Output.avi', self.fourcc,30, [self.imageSize[0], self.imageSize[1]], 0)
        nt=0
        while True:
            nt+=1
            #Simulation steps
            for vel in self.forcedVel:
                x,y,u,v = vel
                self.sim.u[y,x] = u
                self.sim.v[y,x] = v
            self.sim.timestep()

            uUpscaled = cv2.resize(self.sim.u, self.imageSize)
            vUpscaled = cv2.resize(self.sim.v, self.imageSize)


            endX = self.colsUpsc+(self.sim.u[self.rows, self.cols].flatten()*self.arrowSpacing*5).astype(np.int32)
            endY = self.rowsUpsc+(self.sim.v[self.rows, self.cols].flatten()*self.arrowSpacing*5).astype(np.int32)

            scaledImage = self.baseImage.copy()
            velStack = self.LICBase.copy()

            velStack[:,:,0]=uUpscaled
            velStack[:,:,1]=vUpscaled

            lic_image = lic_flow(velStack, t=nt/10.0, len_pix=5, noise=self.LICNoise)*255

            cimage=cv2.cvtColor(lic_image.astype(np.uint8),cv2.COLOR_GRAY2BGR)

            for i in range(len(self.rows)):
                cv2.arrowedLine(cimage, [self.colsUpsc[i], self.rowsUpsc[i]], [endX[i], endY[i]], [0,255,0])

            
            self.video.write((lic_image.astype(np.uint8)))
            cv2.imshow(self.window_name, cimage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()


