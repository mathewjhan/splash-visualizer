import numpy as np
import pyaudio
import audioop
import aubio
import music21
import struct
import random
import math
from pydub import AudioSegment
from pydub.utils import make_chunks
from tkinter import *
from tkinter.filedialog import askopenfilename

#################################################
# !!!!!!!!!!! PRESS M ON HOMEPAGE !!!!!!!!!!!!! #
# 15112 Term Project                            #
# Splash! - an audio visualizer                 #
# by Mathew Han (mathewh)                       #
#################################################

# Template class for reference
# Main mode windows inherit from screen object
class Screen(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def processEvent(self, event): pass

    def onTimerFired(self): pass

    def draw(self, canvas): pass

# Temporary text object
# Disappears after 60 frames
class TemporaryText(object):
    def __init__(self, x, y, text):
        self.x = x
        self.y = y
        self.text = text
        self.timer = 99999
        self.font = ("Arial", 20)

    def resetTimer(self):
        self.timer = 0

    def draw(self, canvas):
        if self.timer < 60:
            canvas.create_text(self.x, self.y, text=self.text, font=self.font, fill="red")
            self.timer+=1

# Custom button class for clickable buttons
class Button(object):
    def __init__(self, x1, y1, x2, y2, text):
        self.topLeft = (x1, y1)
        self.botRight = (x2, y2)
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.width = x2-x1
        self.height = y2-y1
        self.text = text
        self.isDown = False
        self.radius = 25
        self.font = ("Sans Seriff", 15)
        self.oColor = "#ca2e55"
        self.iColor = "#ffe0b5"
        self.fontColor = "#ca2e55"

        # Points listen from
        # https://stackoverflow.com/questions/44099594/
        # how-to-make-a-tkinter-canvas-rectangle-with-rounded-corners

    def getPoints(self, x1, y1, x2, y2, radius):
        points = [x1+radius, y1,
                  x1+radius, y1,
                  x2-radius, y1,
                  x2-radius, y1,
                  x2, y1,
                  x2, y1+radius,
                  x2, y1+radius,
                  x2, y2-radius,
                  x2, y2-radius,
                  x2, y2,
                  x2-radius, y2,
                  x2-radius, y2,
                  x1+radius, y2,
                  x1+radius, y2,
                  x1, y2,
                  x1, y2-radius,
                  x1, y2-radius,
                  x1, y1+radius,
                  x1, y1+radius,
                  x1, y1]
        return points

    def onClick(self):
        self.oColor, self.iColor, self.fontColor = self.iColor, self.oColor, self.iColor
        self.isDown = True

    def onRelease(self):
        self.oColor, self.iColor, self.fontColor = self.iColor, self.oColor, self.iColor
        self.isDown = False

    def isClicked(self, eventX, eventY):
        return (self.x1 <= eventX <= self.x2 and
                self.y1 <= eventY <= self.y2)

    def onTimerFired(self):
        pass

    def draw(self, canvas):
        cx = (self.x1+self.x2)/2
        cy = (self.y1+self.y2)/2
        sx1, sy1 = self.x1+self.width/20, self.y1+self.height/10
        sx2, sy2 = self.x2-self.width/20, self.y2-self.height/10
        oPoints = self.getPoints(self.x1, self.y1, self.x2, self.y2, self.radius)
        iPoints = self.getPoints(sx1, sy1, sx2, sy2, self.radius)
        canvas.create_polygon(oPoints, fill=self.oColor, smooth=True)
        canvas.create_polygon(iPoints, fill=self.iColor, smooth=True)
        canvas.create_text(cx, cy, fill=self.fontColor, text=self.text, font=self.font)

# Help screen
class Help(Screen):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.font = ("Georgia", 30)

    def drawHelp(self, canvas):
        text = """
        WELCOME TO SPLASH!

        Global:
            Backspace - Go back to homepage

        Visualizer/Sing Mode:
            Space - Play/Pause music
            Left - Rewind 15 seconds
            Right - Forward 15 seconds

        Sing Mode:
            Have fun singing to your favorite songs!
            H - Hide the tuner!
                """
        canvas.create_text(self.width/2, self.height/2, text=text, font=self.font)

    def draw(self, canvas):
        self.drawHelp(canvas)

# Homepage screen
class Homepage(Screen):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.font = ("Comic Sans MS", 100)
        self.bWidth = 200
        self.bHeight = 100
        self.counter = 0
        self.balloons = []
        self.offset = height/7
        self.buttons = []
        self.lb = Button(width/2-self.bWidth/2, 3*self.offset-self.bHeight/2, 
            width/2+self.bWidth/2, 3*self.offset+self.bHeight/2, "Load")
        self.startVis = Button(width/2-self.bWidth/2, 4*self.offset-self.bHeight/2,
         width/2+self.bWidth/2, 4*self.offset+self.bHeight/2, "Visualize")
        self.startGame = Button(width/2-self.bWidth/2, 5*self.offset-self.bHeight/2, 
            width/2+self.bWidth/2, 5*self.offset+self.bHeight/2, "Sing")
        self.help = Button(width/2-self.bWidth/2, 6*self.offset-self.bHeight/2, 
            width/2+self.bWidth/2, 6*self.offset+self.bHeight/2, "Help")
        self.buttons.extend([self.lb, self.startVis, self.startGame, self.help])

    def drawTitle(self, canvas):
        canvas.create_text(self.width/2, self.offset, text="SPLASH!", font=self.font)

    def processEvent(self, event):
        for button in self.buttons:
            if str(event.type) == "ButtonPress" and button.isClicked(event.x, event.y):
                button.onClick()
            elif str(event.type) == "ButtonRelease" and button.isDown:
                button.onRelease()

    # Adds pretty colors :O
    def createBalloons(self):
        for i in range(random.randint(1,10)):
            angle = 3*math.pi/2
            speed = random.randint(10, 20)
            radius = random.randint(10,40)
            x = random.randint(radius, self.width-40)
            y = self.height-radius-1
            self.balloons.append(Balloon(x, y, angle, speed, radius))

    def drawBalloons(self, canvas):
        for balloon in self.balloons:
            balloon.draw(canvas)

    def updateBalloons(self):
        for balloon in self.balloons:
            balloon.move()
            balloon.accelerate()

    def removeBalloons(self, cWidth, cHeight):
        for balloon in self.balloons:
            if(balloon.cy >= cHeight or balloon.cy < 0 or
               balloon.cx >= cWidth or balloon.cx < 0): 
                self.balloons.remove(balloon)

    def onTimerFired(self):
        self.counter += 1
        self.updateBalloons()
        self.removeBalloons(self.width, self.height)
        if self.counter % 10 == 0:
            self.createBalloons()

    def draw(self, canvas):
        self.drawBalloons(canvas)
        self.drawTitle(canvas)
        self.lb.draw(canvas)
        self.startVis.draw(canvas)
        self.startGame.draw(canvas)
        self.help.draw(canvas)

# Visualizer screen
class Visualizer(Screen):
    def __init__(self, paths, width, height, mode):
        super().__init__(width, height)
        self.paudio = pyaudio.PyAudio()
        self.RATE = 44100
        self.CHANNELS = 2
        self.CHUNK = 2*1<<10
        self.FORMAT = pyaudio.paInt16
        self.CHUNKLENGTH = 50

        self.mode = mode
        self.isPaused = False
        self.curData = None
        self.musicData = None
        self.isStarted = False
        self.hideTuner = False

        self.fps = int(1000/self.CHUNKLENGTH)
        self.paths = paths
        self.circleDiameter = height/2
        self.curRange = None

        self.colors = ["#70e4ef", "#e2ef70", "#ef709d", "#f038ff", "#3772ff"]
        self.token = 0
        self.counter = 0
        self.maxRMS = -1
        self.pastRMS = []
        self.prevRMS = None
        self.curRMS = None

        self.monoChannels = []
        self.audioFiles = []
        self.audioData = []
        self.balloons = []

        self.cx = width/2
        self.cy = height/2

        self.curPitch = ""

        for path in paths:
            self.load(path)

        self.musicStream = self.paudio.open(format=self.FORMAT,
                                       channels=self.CHANNELS,
                                       rate=self.RATE,
                                       output=True)
        self.inputStream = self.paudio.open(format=self.FORMAT,
                                       channels=1,
                                       rate=self.RATE,
                                       input=True)

        self.pDetection = aubio.pitch("default", 8192,
            8192//2, 44100)
        self.pDetection.set_unit("Hz")
        self.pDetection.set_silence(-40)

        self.m21Pitch = music21.pitch.Pitch()

    # Load audio files
    def load(self, path):
        audio = AudioSegment.from_file(path)
        chunks = make_chunks(audio, self.CHUNKLENGTH) # 1764 
        for chunk in chunks:
            self.maxRMS = max(self.maxRMS, audioop.rms(chunk.raw_data, audio.channels))

        monos = audio.split_to_mono()
        self.monoChannels.append(monos)

        self.audioFiles.append(audio)
        self.audioData.append(chunks)

    def playPause(self):
        self.isPaused = not self.isPaused

    def readInput(self):
        rawData = self.inputStream.read(self.CHUNK)
        self.curData = rawData

    # Converts raw audio data to frequency to its note
    def updatePitch(self):
        rawData = self.curData
        fixedData = (((self.convertDecimal(rawData)-128)*256)*1./32768).astype(aubio.float_type)
        pitch = self.pDetection(fixedData)[0]*2
        if pitch != 0:
            self.m21Pitch.frequency = pitch
            self.curPitch = str(self.m21Pitch.name)
        else: self.curPitch =  ""

    # Used for dividing the bar graph into subintervals
    def getSubChunks(self, lower, upper, n):
        step = int((upper-lower)/n)
        if step <= 0: step = 1
        return [i for i in range(lower, upper)[::step]]

    # Get intervals for bar graphs
    def getIntervals(self, chunkLength, reference):
        deltas = [int((reference[i]-reference[i-1])/reference[-1]*chunkLength) 
        for i in range(1, len(reference))]
        intervals = [sum(deltas[:i]) for i in range(1, len(deltas)+1)]
        return [0] + intervals

    # Returns audio spectrum for raw audio data as a 
    # numpy array with relative values (0, 1)
    def getRatios(self, chunkAbs):
        reference = [0,20,50,100,200,500,1000,2000,5*10**3,10**4,2*10**4]
        sections = 10
        averages = []
        upperBound = 1
        intervals = self.getIntervals(len(chunkAbs), reference)
        for i in range(1, len(intervals)):
            lower = intervals[i-1]
            upper = intervals[i]+1
            subChunks = self.getSubChunks(lower, upper, sections)
            for j in range(1, len(subChunks)):
                left = subChunks[j-1]
                right = subChunks[j]+1
                avg = sum(chunkAbs[left:right])/(right-left)
                upperBound = max(upperBound, avg)
                averages.append(avg)

        averages = np.array(averages)/upperBound
        self.curRange = np.argmax(averages)
        return averages

    # Takes in audio segment object and returns
    # the complete string of raw data
    def getSamples(self, audio):
        return audio._data

    # Legacy code, used for determining BPM
    def getFPB(self):
        SECONDSPERMIN = 60
        return int((SECONDSPERMIN*self.fps)/self.bpm)

    # Legacy code, spent a long time trying to get this to work
    # Eventually worked, but bpm detection is general is very difficult
    # Calculated bpms were always around +-5 of the actual one
    # I decided to use another approach
    # Implementation:
    # https://github.com/aubio/aubio/blob/master/python/demos/demo_bpm_extract.py
    def getBPM(self):
        beats = []
        rawData = self.getSamples(self.monoChannels[0][1])
        length = len(rawData)
        n = int(length/(1<<10))
        i = 0
        max_nb_bit = float(2**(16-1))  
        for i in range(n):
            intData = self.convertDecimal(rawData[i*1024:(i+1)*1024:2])
            chunk = (intData/max_nb_bit).astype('float32')
            if(len(chunk) == 512):
                isBeat = self.tempo(chunk)
                if(isBeat): 
                    beat = self.tempo.get_last_s()
                    beats.append(beat)
        bpms = 60./np.diff(beats)
        return np.median(bpms)

    # Converts a raw data chunk to integers (0, 255)
    def getIntChunk(self, chunk):
        chunk = chunk
        return self.convertDecimal(chunk)[1::2]

    # Updates the visualizer rms
    def updateRMS(self):
        chunk = self.curData
        rms = audioop.rms(chunk, 2)
        self.maxRMS = max(self.maxRMS, rms)
        self.curRMS = rms

    # Gets the actual heights of the bars as a numpy
    # array
    def getBars(self):
        chunkInt = self.getIntChunk(self.curData)
        chunkFFT = np.fft.fft(chunkInt)
        chunkAbs = np.abs(chunkFFT)[1:]

        ratios = self.getRatios(chunkAbs)
        bars = ratios*100
        return bars

    def isEmptyList(self):
        return self.audioData == []

    # Returns True if the delta of volume is within a
    # certain range, essentially tells visualizer to 
    # create the circles
    def isBigLouder(self):
        rms = self.curRMS 
        if len(self.pastRMS) < 30:
            self.pastRMS.append(rms)
        else:
            self.pastRMS.pop(0)
            self.pastRMS.append(rms)
        if (self.prevRMS != None and np.std(self.pastRMS) < 1500 and
         rms != 0 and self.prevRMS*1.2*self.maxRMS**0.5/rms <= rms):
            return True

        elif self.prevRMS != None and self.prevRMS*1.6 <= rms:
            self.prevRMS = rms
            return True
        else:
            self.prevRMS = rms
            return False

    # Converts raw audio data into int values (0, 255)
    def convertDecimal(self, chunk):
        return np.array(struct.unpack(str(len(chunk))+'B', chunk))

    def playAudio(self):
        chunk = self.musicData
        self.musicStream.write(chunk)
        self.token+=1

    # Combines the audio data from multiple files
    # If only one file, then returns its audio data
    # Combining the data is not used anymore in this
    # visualizer, but the function is still used for conversions
    def combineData(self, chunkNum):
        numFiles = len(self.audioFiles)
        rawData = 0
        for chunks in self.audioData:
            if chunkNum >= len(chunks):
                self.audioData.remove(chunks)

        for i in range(len(self.audioData)):
            # Gets raw data of ms*176 BYTES
            temp = self.audioData[i][chunkNum].raw_data
            rawData += (np.fromstring(temp, np.int16)*(1/numFiles))
        rawData = np.array(rawData, dtype=np.int16).tostring()
        self.musicData = rawData
        if self.mode == "vis": self.curData = rawData

    # Adds some padding at the end of the bars to make it
    # look nicer :O
    def padBars(self, start, end):
        if math.isnan(start) or math.isnan(end):
            start, end = 1, 1
        else: start, end = int(start), int(end)
        step = int(abs(end-start)/3)+1
        if start > end: order = (end, start)
        else: order = (start, end) 
        return [i for i in range(*order)[::step]]

    # Balloons sooo pretty 
    def createBalloons(self, cWidth, cHeight):
        rms = self.curRMS
        n = int((rms/self.maxRMS)*15)
        cx = self.cx
        cy = self.cy
        if len(self.balloons) < 175:
            for i in range(n*14):
                angle = random.randint(0, 360) * math.pi/180
                color = random.choice(self.colors)
                cx = cx+math.cos(angle)
                cy = cy+math.sin(angle)
                speed = random.randint(30+2*n, 30+3*n)*(self.curRMS/self.maxRMS + 0.75)
                radius = random.randint(n, 2*n)
                self.balloons.append(Balloon(cx, cy, angle, speed, radius))

    def updateBalloons(self):
        for balloon in self.balloons:
            balloon.move()
            balloon.accelerate()

    def removeBalloons(self, cWidth, cHeight):
        for balloon in self.balloons:
            if(balloon.cy >= cHeight or balloon.cy < 0 or
               balloon.cx >= cWidth or balloon.cx < 0): 
                self.balloons.remove(balloon)

    def removeAllBalloons(self):
        self.balloons = []

    def drawBalloons(self, canvas):
        for balloon in self.balloons:
            balloon.draw(canvas)

    def drawPitch(self, canvas):
        canvas.create_text(self.width/2, 7*self.height/8, text=self.curPitch, 
            fill="Black", font=("Arial", 40))

    def drawBeatCircle(self, canvas, cWidth, cHeight, diameter):
        rms = self.curRMS
        factor = rms/self.maxRMS+0.5
        if self.curRange < 3:
            miniRad = diameter/4*(factor+0.25)
        else:
            miniRad = diameter/4*factor
        radius = diameter/2*factor
        cx = self.cx
        cy = self.cy
        canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius, width=3, 
            fill="", outline="#f686a8")
        canvas.create_oval(cx-miniRad, cy-miniRad, cx+miniRad, cy+miniRad, 
            width=1, fill="#fe5d9f", outline="")
        
    def drawCircle(self, canvas, cWidth, cHeight, diameter):
        rInner = diameter/2
        rms = self.curRMS
        factor = rms/self.maxRMS+0.6
        bars = list(self.getBars()*factor)
        bars = 2*(self.padBars(bars[0], bars[-1])+bars)

        angle = 2*math.pi/len(bars)
        cx = self.cx
        cy = self.cy
        for i in range(len(bars)):
            height = bars[i]
            color = random.choice(self.colors)
            compX0 = cx+math.cos((i)*angle)*rInner
            compY0 = cy+math.sin((i)*angle)*rInner
            compX1 = cx+math.cos((i)*angle)*(height+rInner)*factor
            compY1 = cy+math.sin((i)*angle)*(height+rInner)*factor
            p0 = (compX0, compY0)
            p1 = (compX1, compY1)
            canvas.create_line(p0, p1, fill=color, width=3)

    def drawWave(self, canvas, cWidth, cHeight):
        rms = self.curRMS
        factor = factor = rms/self.maxRMS+1
        chunkInt = self.getIntChunk(self.curData)
        chunkFFT = np.fft.fft(chunkInt)
        chunkAbs = np.abs(chunkFFT)[1:int(len(chunkFFT)/4)]
        ratios = [4 for i in range(10)] + list(-self.getRatios(chunkAbs)*100) + [1 for i in range(10)]
        offset = cWidth/len(ratios)/2
        for i in range(1, len(ratios)):
            x1 = (i-1)*offset
            y1 = cHeight/2+ratios[i-1]/2
            x2 = (i-1)*offset
            y2 = cHeight/2-ratios[i-1]/2
            canvas.create_line(x1, y1, x2, y2, fill="pink", width=4)
            canvas.create_line(cWidth-x1,y1,cWidth-x2,y2, fill="pink", width=4)

    def onTimerFired(self):
        if self.isPaused: return
        if not self.isEmptyList():
            if self.mode == "sing":
                self.readInput()
                if self.counter%10==0: self.updatePitch()
            self.combineData(self.token)
            self.updateRMS()
            self.isStarted = True
            self.playAudio()
            if self.isBigLouder():
                self.createBalloons(self.width, self.height)
            self.updateBalloons()
            self.removeBalloons(self.width, self.height)
            self.counter+=1
        else: counter = 0

    def processEvent(self, event):
        if event.keysym == "space":
            self.playPause()
        if self.isPaused: return
        if event.keysym == "Left":
            length = int(15000/self.CHUNKLENGTH)
            if self.token - length >= 0:
                self.token -= length
            else:
                self.token = 0
                self.removeAllBalloons()
        if event.keysym == "Right":
            length = int(15000/self.CHUNKLENGTH)
            if self.token + length < len(self.audioData[0]):
                self.token += length
            else:
                self.token = len(self.audioData[0]) - 1
                self.removeAllBalloons()
        if event.keysym == "h":
            self.hideTuner = not self.hideTuner

    def draw(self, canvas):
        self.drawWave(canvas, self.width, self.height)
        if self.curData != None and self.mode == "sing" or not self.isEmptyList():
            self.drawCircle(canvas, self.width, self.height, self.circleDiameter)
        self.drawBalloons(canvas)
        self.drawBeatCircle(canvas, self.width, self.height, self.circleDiameter/1.5)
        if self.curData != None and self.mode == "sing" and not self.hideTuner:
            self.drawPitch(canvas)

# Balloon objects (the colorful circles drawn onto the
# screen)
class Balloon(object):
    def __init__(self, x, y, angle, speed, radius):
        self.rad = radius
        self.colors = ["#70e4ef", "#e2ef70", "#ef709d", "#f038ff", "#3772ff"]
        self.color = random.choice(self.colors)
        self.cx = x
        self.cy = y
        self.angle = angle
        self.speed = speed

    def draw(self, canvas):
        cx, cy, rad = self.cx, self.cy, self.rad
        canvas.create_oval(cx-rad, cy-rad, cx+rad, cy+rad, fill=self.color, outline="")

    def move(self):
        if(self.speed >= 4):
            self.cx += self.speed*math.cos(self.angle)
            self.cy += self.speed*math.sin(self.angle)

    def accelerate(self):
        if self.speed >= 40:
            self.speed-=1

# The main window that runs (AKA root)
# 15112 framework implemented as a class
# Framework from:
# https://www.cs.cmu.edu/~112n18/notes/notes-animations-part2.html
class ApplicationWindow(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.counter = 0
        self.timerDelay = 1
        self.screen = "home"
        self.bgColor = "#f5fbef"

        self.paths = []
        self.home = Homepage(self.width, self.height)
        self.help = Help(self.width, self.height)
        self.vis = None
        self.plsLoad = TemporaryText(self.width/2, 2*self.height/7, "PLS LOAD SOMETHING")
        
        self.root = Tk()
        self.root.title("Splash!")
        self.canvas = Canvas(self.root, width=self.width, height=self.height, 
            background=self.bgColor)
        self.canvas.pack()  

        self.root.bind("<ButtonPress-1>", lambda event:
                                self.mousePressedWrapper(event, self.canvas))
        self.root.bind("<ButtonRelease-1>", lambda event:
                                self.mouseReleasedWrapper(event, self.canvas))
        self.root.bind("<KeyPress>", lambda event:
                                self.keyPressedWrapper(event, self.canvas))
        self.timerFiredWrapper(self.canvas)

        self.root.mainloop()

    def startVisualizer(self, mode):
        self.vis = Visualizer(self.paths, self.width, self.height, mode)

    def endVisualizer(self):
        self.paths.pop()

    def getPath(self):
        Tk().withdraw()
        path = askopenfilename()
        return path

    def mousePressedWrapper(self, event, canvas):
        self.mousePressed(event)
        self.redrawAllWrapper(canvas)

    def mouseReleasedWrapper(self, event, canvas):
        self.mouseReleased(event)
        self.redrawAllWrapper(canvas)

    def keyPressedWrapper(self, event, canvas):
        self.keyPressed(event)
        self.redrawAllWrapper(canvas)

    def timerFiredWrapper(self, canvas):
        self.redrawAllWrapper(canvas)
        self.timerFired()
        canvas.after(self.timerDelay, self.timerFiredWrapper, self.canvas)

    def redrawAllWrapper(self, canvas):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, self.width, self.height,
                                fill=self.bgColor, width=0)
        self.redrawAll(canvas)
        canvas.update()

    def mousePressed(self, event):
        if self.screen == "home": 
            self.home.processEvent(event)

    def mouseReleased(self, event):
        if self.screen == "home": 
            self.home.processEvent(event)
            if self.home.lb.isClicked(event.x, event.y):
                if self.paths != []:
                    self.paths.pop()
                path = self.getPath()
                if path != "":
                    self.paths.append(path)
            if self.home.startVis.isClicked(event.x, event.y): 
                if self.paths == []: self.plsLoad.resetTimer()
                else:
                    self.screen = "vis"
                    self.startVisualizer("vis")      
            if self.home.startGame.isClicked(event.x, event.y):
                if self.paths == []: self.plsLoad.resetTimer()
                else:
                    self.screen = "vis"
                    self.startVisualizer("sing")
            if self.home.help.isClicked(event.x, event.y):
                    self.screen = "help"

    def keyPressed(self, event):
        if event.keysym == "BackSpace":
            self.screen = "home"
            self.endVisualizer()
        if self.screen == "vis" and self.vis.isStarted:
            self.vis.processEvent(event)

    def timerFired(self):
        if self.screen == "vis": 
            self.vis.onTimerFired()
            if self.vis.isEmptyList():
                self.screen = "home"
                self.paths.pop()
        if self.screen == "home": self.home.onTimerFired()

    def redrawAll(self, canvas):
        if self.screen == "home": 
            self.home.draw(canvas)
            self.plsLoad.draw(canvas)
        if self.screen == "vis" and self.vis.isStarted: self.vis.draw(canvas)
        if self.screen == "help": self.help.draw(canvas)

def main():
    ApplicationWindow(1500, 800)

if __name__ == '__main__':
    main()