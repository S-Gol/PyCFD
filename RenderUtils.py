import numpy as np
import cv2
from scipy import ndimage
import FluidSim
from lic import *
from sympy.physics.vector import curl
from sympy.physics.vector import ReferenceFrame


class FluidsRenderer:
    def __init__(self, sim, upscaleMult = 5, arrowSpacing=5, 
    forcedVel=[], windowName = "PyCFD", arrows=True, colorMode = "speed", callback=None, arrowLength = 20
    ):
        """
        Easy rendering utility for fluid sim.
        colorMode: speed, LIC


        """
        
        if colorMode != "speed" and colorMode != "LIC" and colorMode!= "pressure":
            print("Invalid color mode")
            raise ValueError
        self.callback = callback
        self.sim = sim
        self.upscaleMult = upscaleMult
        self.arrowSpacing = arrowSpacing
        self.useArrows = arrows
        self.colorMode = colorMode
        self.arrowLength = arrowLength
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.forcedVel = forcedVel

        self.imageSize = (int(sim.nx*upscaleMult), int(sim.ny * upscaleMult))
        self.downscaledIndices = (sim.indices * upscaleMult)[:,::arrowSpacing,::arrowSpacing].astype(int)
        self.upscaledIndices = (np.indices(self.imageSize)/self.upscaleMult).astype(int)
        self.rowsUpsc = self.downscaledIndices[1,:,:].flatten().astype(np.int32)
        self.colsUpsc = self.downscaledIndices[0,:,:].flatten().astype(np.int32)

        self.rows = (self.rowsUpsc / upscaleMult).astype(np.int32)
        self.cols = (self.colsUpsc / upscaleMult).astype(np.int32)
        
        self.arrowOrigins = [[self.rowsUpsc[i], self.colsUpsc[i]] for i in range(len(self.rows))]
        self.window_name=windowName

        cv2.namedWindow(self.window_name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.imageSize[0],self.imageSize[1])
        self.baseImage = np.zeros(self.imageSize)
        self.LICNoise = np.random.rand(*(self.imageSize)).transpose()*cv2.resize(self.sim.velBoundary, self.imageSize)
        self.LICBase = np.zeros([self.imageSize[1], self.imageSize[0],2])
        self.R = ReferenceFrame('R')

    def startContinousRender(self):
        self.video = cv2.VideoWriter('Output.mp4', self.fourcc,60, [self.imageSize[0], self.imageSize[1]])
        nt=0
        while True:
            nt+=1
            #Simulation steps
            for vel in self.forcedVel:
                x,y,u,v = vel
                self.sim.u[y,x] = u
                self.sim.v[y,x] = v

            if self.callback is not None:
                self.callback()
            self.sim.timestep()

            uUpscaled = cv2.resize(self.sim.u, self.imageSize)
            vUpscaled = cv2.resize(self.sim.v, self.imageSize)
            pUpscaled = cv2.resize(self.sim.p, self.imageSize)
            if self.colorMode=="speed" or self.useArrows == True:
                speeds = np.sqrt(self.sim.u**2+self.sim.v**2)
                speedMax = np.max(speeds)

            if self.colorMode == "LIC":
                velStack = self.LICBase.copy()

                velStack[:,:,0]=uUpscaled
                velStack[:,:,1]=vUpscaled

                lic_image = lic_flow(velStack, t=nt/10.0, len_pix=10, noise=self.LICNoise)*255
                mainImage=cv2.cvtColor(lic_image.astype(np.uint8),cv2.COLOR_GRAY2BGR)
            if self.colorMode == "speed":

                mainImage = cv2.cvtColor((speeds*255/speedMax).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                mainImage = cv2.applyColorMap(mainImage, cv2.COLORMAP_JET)
                mainImage = cv2.resize(mainImage, self.imageSize)

            if self.colorMode == "pressure":
                mainImage=cv2.cvtColor(pUpscaled.astype(np.uint8),cv2.COLOR_GRAY2BGR)*255/1000

                
                pressureImage = cv2.resize(self.sim.p*255/np.max(self.sim.p), self.imageSize)
                mainImage=cv2.cvtColor(pressureImage.astype(np.uint8),cv2.COLOR_GRAY2BGR)
                mainImage=cv2.applyColorMap(mainImage, cv2.COLORMAP_JET)
            
            if self.useArrows:

                endX = (self.colsUpsc+(self.sim.u[self.rows, self.cols].flatten()*self.arrowSpacing*self.sim.dt/self.sim.dx)/speedMax*self.arrowLength).astype(np.int32)
                endY = (self.rowsUpsc+(self.sim.v[self.rows, self.cols].flatten()*self.arrowSpacing*self.sim.dt/self.sim.dx)/speedMax*self.arrowLength).astype(np.int32)
                for i in range(len(self.rows)):
                    cv2.arrowedLine(mainImage, [self.colsUpsc[i], self.rowsUpsc[i]], [endX[i], endY[i]], [int(0),int(0),int(0)])

            
            self.video.write((mainImage.astype(np.uint8)))
            cv2.imshow(self.window_name, mainImage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()


