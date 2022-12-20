#############################################################################
"""
Demo of demostrating fault results
Author: Xinming Wu, USTC
Version: 2020.03.26
"""
from datUtils import * 
#############################################################################

setupForSubset("xin")
pngDir = getPngDir()
s1,s2,s3 = getSamplings()
n1,n2,n3=s1.count,s2.count,s3.count
epfile = 'ep'
gxfile = 'gr'
c2file = 'c2'
c3file = 'c3'
plotOnly = True
plotOnly = False

def main(args):
  #goPlanarity()
  #goCoherence()
  goC3()
def goPlanarity():
  gx = readImage(gxfile)
  if not plotOnly:
    lof = LocalOrientFilter(4,1,1)
    u1 = zerofloat(n1,n2,n3)
    u2 = zerofloat(n1,n2,n3)
    u3 = zerofloat(n1,n2,n3)
    ep = zerofloat(n1,n2,n3)
    lof.applyForNormalPlanar(gx,u1,u2,u3,ep)
    writeImage(epfile,ep)
  else:
    ep = readImage(epfile)
  ep = pow(ep,8)
  plot3(s1,s2,s3,gx)
  plot3(s1,s2,s3,gx,sub(1,ep),cmin=0.25,cmax=1.0,cmap=jetRamp(1.0),
      clab="1-planarity",png="ep")
def goC2():
  sig1,sig2,sig3 = 16,1,1 
  gx = readImage(gxfile)
  if not plotOnly:
    lof = LocalOrientFilter(8,2)
    ets = lof.applyForTensors(gx)
    ets.setEigenvalues(0.0001,1.0,1.0)
    lsf = LocalSmoothingFilter()
    ref1 = RecursiveExponentialFilter(sig1)
    ref1.setEdges(RecursiveExponentialFilter.Edges.OUTPUT_ZERO_SLOPE)
    gn = zerofloat(n1,n2,n3)
    gd = zerofloat(n1,n2,n3)
    # compute the numerator of coherence
    lsf.apply(ets,sig2,gx,gn)
    gn = mul(gn,gn)
    ref1.apply1(gn,gn)
    # compute the denominator of coherence
    lsf.apply(ets,sig2,mul(gx,gx),gd)
    ref1.apply1(gd,gd)
    c2 = div(gn,gd)
    writeImage(c2file,c2)
  else:
    c2 = readImage(c2file)
  c2 = pow(c2,8)
  plot3(s1,s2,s3,gx,sub(1,c2),cmin=0.25,cmax=1.0,cmap=jetRamp(1.0),
        clab="1-coherence",png="c2")
# covariance-matrix-based semblance
# (Gersztenkorn and Marfurt, 1999)
def goC3():
  gx = readImage(gxfile)
  if not plotOnly:
    p2 = zerofloat(n1,n2,n3)
    p3 = zerofloat(n1,n2,n3)
    ep = zerofloat(n1,n2,n3)
    lsf = LocalSlopeFinder(8,2,2,5)
    lsf.findSlopes(gx,p2,p3,ep)
    ep = zerofloat(n1,n2,n3)
    copy(70,n2,n3,0,0,0,ep,0,0,0,p2)
    copy(70,n2,n3,0,0,0,ep,0,0,0,p3)
    cv = Covariance()
    em,es = cv.covarianceEigen(10,p2,p3,gx)
    c3 = div(em,es)
    writeImage(c3file,c3)
  else:
    c3 = readImage(c3file)
  c3 = pow(c3,8)
  plot3(gx,sub(1,c3),cmin=0.25,cmax=1.0,cmap=jetRamp(1.0),
        clab="1-coherence",png="c3")

def gain(x):
  g = mul(x,x) 
  ref = RecursiveExponentialFilter(20.0)
  ref.apply1(g,g)
  div(x,sqrt(g),x)
  return x

def checkNaN(gx):
  n3 = len(gx)
  n2 = len(gx[0])
  n1 = len(gx[0][0])
  for i3 in range(n3):
    for i2 in range(n2):
      for i1 in range(n1):
        if(gx[i3][i2][i1]!=gx[i3][i2][i1]):
          gx[i3][i2][i1] = 0
  return gx

def smooth(sig,u):
  v = copy(u)
  rgf = RecursiveGaussianFilterP(sig)
  rgf.apply0(u,v)
  return v

def smooth2(sig1,sig2,u):
  v = copy(u)
  rgf1 = RecursiveGaussianFilterP(sig1)
  rgf2 = RecursiveGaussianFilterP(sig2)
  rgf1.apply0X(u,v)
  rgf2.applyX0(v,v)
  return v

def normalize(e):
  emin = min(e)
  emax = max(e)
  return mul(sub(e,emin),1.0/(emax-emin))

def slice12(k3,f):
  n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
  s = zerofloat(n1,n2)
  SimpleFloat3(f).get12(n1,n2,0,0,k3,s)
  return s

def slice13(k2,f):
  n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
  s = zerofloat(n1,n3)
  SimpleFloat3(f).get13(n1,n3,0,k2,0,s)
  return s

def slice23(k1,f):
  n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
  s = zerofloat(n2,n3)
  SimpleFloat3(f).get23(n2,n3,k1,0,0,s)
  return s

#############################################################################
# graphics

def jetFill(alpha):
  return ColorMap.setAlpha(ColorMap.JET,alpha)
def jetFillExceptMin(alpha):
  a = fillfloat(alpha,256)
  a[0] = 0.0
  return ColorMap.setAlpha(ColorMap.JET,a)
def jetRamp(alpha):
  return ColorMap.setAlpha(ColorMap.JET,rampfloat(0.0,alpha/256,256))
def bwrFill(alpha):
  return ColorMap.setAlpha(ColorMap.BLUE_WHITE_RED,alpha)
def bwrNotch(alpha):
  a = zerofloat(256)
  for i in range(len(a)):
    if i<128:
      a[i] = alpha*(128.0-i)/128.0
    else:
      a[i] = alpha*(i-127.0)/128.0
  return ColorMap.setAlpha(ColorMap.BLUE_WHITE_RED,a)
def hueFill(alpha):
  return ColorMap.getHue(0.0,1.0,alpha)
def hueFillExceptMin(alpha):
  a = fillfloat(alpha,256)
  a[0] = 0.0
  return ColorMap.setAlpha(ColorMap.getHue(0.0,1.0),a)

def addColorBar(frame,clab=None,cint=None):
  cbar = ColorBar(clab)
  if cint:
    cbar.setInterval(cint)
  cbar.setFont(Font("Arial",Font.PLAIN,32)) # size by experimenting
  cbar.setWidthMinimum
  cbar.setBackground(Color.WHITE)
  frame.add(cbar,BorderLayout.EAST)
  return cbar

def convertDips(ft):
  return FaultScanner.convertDips(0.2,ft) # 5:1 vertical exaggeration

def plot2(s1,s2,x,u=None,g=None,x1=None,c=None,
        cmap=ColorMap.GRAY,clab="Amplitude",
        cmin=-2,cmax=2,title=None,png=None):
  sp = SimplePlot(SimplePlot.Origin.UPPER_LEFT)
  if title:
    sp.setTitle(title)
  n1,n2=s1.count,s2.count
  sp.addColorBar(clab)
  #sp.setSize(955,400)
  sp.setSize(755,500)
  sp.setHLabel("Inline (sample)")
  sp.setVLabel("Depth (sample)")
  sp.plotPanel.setColorBarWidthMinimum(60)
  sp.setVLimits(0,n1-1)
  sp.setHLimits(0,n2-1)
  sp.setFontSize(16)
  pv = sp.addPixels(s1,s2,x)
  pv.setColorModel(cmap)
  pv.setInterpolation(PixelsView.Interpolation.LINEAR)
  if cmin<cmax:
    pv.setClips(cmin,cmax)
  if u:
    cv = sp.addContours(s1,s2,u)
    cv.setContours(80)
    cv.setLineColor(Color.YELLOW)
  if g:
    pv = sp.addPixels(s1,s2,g)
    pv.setInterpolation(PixelsView.Interpolation.NEAREST)
    pv.setColorModel(ColorMap.getJet(0.8))
    pv.setClips(0.1,s1.count)
  if x1:
    x1k = zerofloat(n2)
    x2  = zerofloat(n2)
    x1s  = zerofloat(n1)
    for i1 in range(n1):
      x1s[i1] = i1
    cp = ColorMap(0,n1,ColorMap.JET)
    rgb = cp.getRgbFloats(x1s)
    ref = RecursiveExponentialFilter(1)
    for k in range(20,n1-20,15):
      for i2 in range(n2):
        x2[i2] = i2
        x1k[i2] = x1[i2][k]
      ref.apply(x1k,x1k)
      pv = sp.addPoints(x1k,x2)
      pv.setLineWidth(2.5)
      r,g,b=rgb[k*3],rgb[k*3+1],rgb[k*3+2]
      pv.setLineColor(Color(r,g,b))
  if pngDir and png:
    sp.paintToPng(700,3.333,pngDir+png+".png")

def plot3(s1,s2,s3,f,g=None,cmin=-2,cmax=2,zs=1.0,sc=1.4,
        ks=[175,330,377],ae=[45,35],vt=[-0.1,-0.06,0.0],
        cmap=None,clab=None,cint=None,surf=None,png=None):
  n3 = len(f)
  n2 = len(f[0])
  n1 = len(f[0][0])
  d1,d2,d3 = s1.delta,s2.delta,s3.delta
  f1,f2,f3 = s1.first,s2.first,s3.first
  l1,l2,l3 = s1.last,s2.last,s3.last
  sf = SimpleFrame(AxesOrientation.XRIGHT_YOUT_ZDOWN)
  cbar = None
  if g==None:
    ipg = sf.addImagePanels(s1,s2,s3,f)
    if cmap!=None:
      ipg.setColorModel(cmap)
    if cmin!=None and cmax!=None:
      ipg.setClips(cmin,cmax)
    else:
      ipg.setClips(-2.0,2.0)
    if clab:
      cbar = addColorBar(sf,clab,cint)
      ipg.addColorMapListener(cbar)
  else:
    ipg = ImagePanelGroup2(s1,s2,s3,f,g)
    ipg.setClips1(-2,2)
    if cmin!=None and cmax!=None:
      ipg.setClips2(cmin,cmax)
    if cmap==None:
      cmap = jetFill(0.8)
    ipg.setColorModel2(cmap)
    if clab:
      cbar = addColorBar(sf,clab,cint)
      ipg.addColorMap2Listener(cbar)
    sf.world.addChild(ipg)
  if cbar:
    cbar.setWidthMinimum(120)
  if surf:
    tg = TriangleGroup(True,surf)
    sf.world.addChild(tg)
  ipg.setSlices(ks[0],ks[1],ks[2])
  if cbar:
    sf.setSize(987,720)
  else:
    sf.setSize(850,720)
  vc = sf.getViewCanvas()
  vc.setBackground(Color.WHITE)
  radius = 0.5*sqrt(n1*n1+n2*n2+n3*n3)
  ov = sf.getOrbitView()
  zscale = zs*max(n2*d2,n3*d3)/(n1*d1)
  ov.setAxesScale(1.0,1.0,1.0)
  ov.setScale(sc)
  ov.setWorldSphere(BoundingSphere(BoundingBox(f3,f2,f1,l3,l2,l1)))
  ov.setTranslate(Vector3(vt[0],vt[1],vt[2]))
  ov.setAzimuthAndElevation(ae[0],ae[1])
  sf.setVisible(True)
  if png and pngDir:
    sf.paintToFile(pngDir+png+".png")
    if cbar:
      cbar.paintToPng(720,1,pngDir+png+"cbar.png")
#############################################################################
# Run the function main on the Swing thread
import sys
class _RunMain(Runnable):
  def __init__(self,main):
    self.main = main
  def run(self):
    self.main(sys.argv)
def run(main):
  SwingUtilities.invokeLater(_RunMain(main)) 
run(main)
