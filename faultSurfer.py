"""
Demonstrate 3D seismic image processing for faults and horizons
Author: Xinming Wu, USTC
Version: 2020.04.17
"""

from datUtils import *
setupForSubset("xin")
s1,s2,s3=getSamplings()
n1,n2,n3=s1.count,s2.count,s3.count

# Names and descriptions of image files used below.
gxfile  = "gx" # seismic image
fpfile  = "fpi" # cnn-based fault probability
fvfile = "fv"  # voting score map;
vpfile = "vp"  # voting strike;
vtfile = "vt"  # voting dip;
fskbase = "skin"  # voting dip;


# These parameters control the scan over fault strikes and dips.
# See the class FaultScanner for more information.
minPhi,maxPhi = 0,360
minTheta,maxTheta = 60,85
sigmaPhi,sigmaTheta = 10,25

# These parameters control the construction of fault skins.
# See the class FaultSkinner for more information.
lowerLikelihood = 0.1
upperLikelihood = 0.3
minSkinSize = 100

# These parameters control the computation of fault dip slips.
# See the class FaultSlipper for more information.
minThrow = 0.01
maxThrow = 45.0

# Directory for saved png images. If None, png images will not be saved;
# otherwise, must create the specified directory before running this script.
pngDir = getPngDir()

# Processing begins here. When experimenting with one part of this demo, we
# can comment out earlier parts that have already written results to files.
plotOnly = False
plotOnly = True
def main(args):
  #goFaultOrientScan()
  goFaultSurfaces()

# approximately estimates fault orientations
def goFaultOrientScan():
  sigmaPhi,sigmaTheta=8,12
  print "scan for approximate fault orientations..."
  gx = readImage(gxfile) # seismic image
  fp = readImage(fpfile) # cnn-based fault probability
  fos = FaultOrientScanner3(sigmaPhi,sigmaTheta)
  if not plotOnly:
    fv,vp,vt = fos.scan(minPhi,maxPhi,minTheta,maxTheta,sub(1,fp))
    writeImage(fvfile,fv)
    writeImage(vpfile,vp)
    writeImage(vtfile,vt)
  else:
    fv = readImage(fvfile)
  plot3(gx,fp,cmin=0.25,cmax=1.0,cmap=jetRamp(1.0),
      clab="CNN-based fault probability",png="fp")
  plot3(gx,fv,cmin=0.25,cmax=1.0,cmap=jetRamp(1.0),
      clab="Enhanced fault probability",png="fv")

# construct fault surfaces
def goFaultSurfaces():
  print "construct fault surfaces..."
  gx = readImage(gxfile)
  fp = readImage(fpfile)
  m1 = 1900
  if not plotOnly:
    osv = OptimalSurfaceVoter(10,20,30)
    fv = readImage(fvfile)
    vp = readImage(vpfile)
    vt = readImage(vtfile)
    fv = copy(m1,n2,n3,0,0,0,fv)
    vp = copy(m1,n2,n3,0,0,0,vp)
    vt = copy(m1,n2,n3,0,0,0,vt)
    u1 = zerofloat(m1,n2,n3)
    u2 = zerofloat(m1,n2,n3)
    u3 = zerofloat(m1,n2,n3)
    ep = zerofloat(m1,n2,n3)
    lof = LocalOrientFilter(4,2,2)
    lof.applyForNormalPlanar(fv,u1,u2,u3,ep)
    ft,pt,tt = osv.thin([fv,vp,vt])
    fsk = FaultSkinner()
    fsk.setGrowing(10,0.3)
    seeds = fsk.findSeeds(10,0.8,ep,ft,pt,tt)
    print(len(seeds))
    skins = fsk.findSkins(0.65,1000,seeds,fv,vp,vt)
    print(len(skins))
    sks = []
    for skin in skins:
      skin.smooth(5)
      sks.append(skin)
    removeAllSkinFiles(fskbase)
    writeSkins(fskbase,sks)
  else:
    #fv = readImage(fvfile)
    #ft = readImage("fvt")
    sks = readSkins(fskbase)
  k = 0
  for skin in sks:
    skin.updateStrike()
  plot3(gx,fp,cmin=0.25,cmax=1.,cmap=jetRamp(1.0),clab="Fault probability",png="fp")
  plot3(gx,skinx=sks,png="skins")
  #plot3(gx,ft,cmin=0.25,cmax=1.,cmap=jetRamp(1.0),clab="Fault probability",png="fp")


#post-processing to enhance faults 
#while at the same time estimating fault strikes and dips
def goFaultEnhance():
  fv = readImage(fvfile)
  plot3(fv,fv,cmin=0.01,cmax=1,cmap=jetRamp(1.0))
  if not plotOnly:
    fe = FaultEnhancer(15,15)
    fl,fp,ft = fe.scan(minPhi,maxPhi,minTheta,maxTheta,sub(1,fv))
    writeImage(flfile,fl)
    writeImage(fpfile,fp)
    writeImage(ftfile,ft)
  else:
    fl = readImage(flfile)
    fp = readImage(fpfile)
    ft = readImage(ftfile)
  plot3(fv,fv,cmin=0.01,cmax=1,cmap=jetRamp(1.0))
  plot3(fv,fl,cmin=0.01,cmax=1,cmap=jetRamp(1.0))
  plot3(fv,fp,cmin=0,cmax=360,cmap=hueFill(1.0),
        clab="Fault strike (degrees)",cint=45,png="fp")

#computed thinned fault attributes
def goThin():
  print "goThin ..."
  gx = readImage(gxfile)
  if not plotOnly:
    fl = readImage(flfile)
    fp = readImage(fpfile)
    ft = readImage(ftfile)
    flt,fpt,ftt = FaultScanner.thin([fl,fp,ft])
    writeImage(fltfile,flt)
    writeImage(fptfile,fpt)
    writeImage(fttfile,ftt)
  else:
    flt = readImage(fltfile)
    fpt = readImage(fptfile)
    ftt = readImage(fttfile)
  plot3(gx,clab="Amplitude")
  plot3(gx,flt,cmin=0.35,cmax=0.8,cmap=jetFillExceptMin(1.0),
        clab="Fault likelihood",png="flt")
  plot3(gx,fpt,cmin=0,cmax=360,cmap=hueFillExceptMin(1.0),
        clab="Fault strike (degrees)",cint=45,png="fpt")
  plot3(gx,ftt,cmin=55,cmax=80,cmap=jetFillExceptMin(1.0),
        clab="Fault dip (degrees)",png="ftt")

def goSmooth():
  print "goSmooth ..."
  flstop = 0.1
  fsigma = 8.0
  if not plotOnly:
    gx = readImage(gxfile)
    flt = readImage(fltfile)
    p2 = readImage(p2file)
    p3 = readImage(p3file)
    gsx = FaultScanner.smooth(flstop,fsigma,p2,p3,flt,gx)
    writeImage(gsxfile,gsx)
  else:
    gsx = readImage(gsxfile)
  #plot3(gx,clab="Amplitude")
  plot3(gsx,clab="Amplitude",png="gsx")

def goSkin():
  print "goSkin ..."
  sx = readImage(sxfile)
  if not plotOnly:
    #gsx = readImage(gsxfile)
    fl = readImage(flfile)
    fp = readImage(fpfile)
    ft = readImage(ftfile)
    fs = FaultSkinner()
    fs.setGrowLikelihoods(lowerLikelihood,upperLikelihood)
    fs.setMinSkinSize(minSkinSize)
    cells = fs.findCells([fl,fp,ft])
    skins = fs.findSkins(cells)
    for skin in skins:
      skin.smoothCellNormals(4)
    print "total number of cells =",len(cells)
    print "total number of skins =",len(skins)
    print "number of cells in skins =",FaultSkin.countCells(skins)
    removeAllSkinFiles(fskbase)
    writeSkins(fskbase,skins)
  else:
    skins = readSkins(fskbase)
  ps = goEventLocations()
  #plot3(gx,cells=cells,png="cells")
  plot3(sx,skins=skins,xyz=ps,png="skins")

def goSlip():
  print "goSlip ..."
  gx = readImage(gxfile)
  gsx = readImage(gsxfile)
  p2 = readImage(p2file)
  p3 = readImage(p3file)
  skins = readSkins(fskbase)
  fsl = FaultSlipper(gsx,p2,p3)
  fsl.setOffset(2.0) # the default is 2.0 samples
  fsl.setZeroSlope(False) # True only if we want to show the error
  fsl.computeDipSlips(skins,minThrow,maxThrow)
  print "  dip slips computed, now reskinning ..."
  print "  number of skins before =",len(skins),
  fsk = FaultSkinner() # as in goSkin
  fsk.setGrowLikelihoods(lowerLikelihood,upperLikelihood)
  fsk.setMinSkinSize(minSkinSize)
  fsk.setMinMaxThrow(minThrow,maxThrow)
  skins = fsk.reskin(skins)
  print ", after =",len(skins)
  removeAllSkinFiles(fssbase)
  writeSkins(fssbase,skins)
  smark = -999.999
  s1,s2,s3 = fsl.getDipSlips(skins,smark)
  writeImage(fs1file,s1)
  writeImage(fs2file,s2)
  writeImage(fs3file,s3)
  plot3(gx,skins=skins,smax=30.0,png="skinss1")
  plot3(gx,s1,cmin=-0.01,cmax=10.0,cmap=jetFillExceptMin(1.0),
        clab="Fault throw (samples)",png="gxs1")
  s1,s2,s3 = fsl.interpolateDipSlips([s1,s2,s3],smark)
  plot3(gx,s1,cmin=0.0,cmax=10.0,cmap=jetFill(0.3),
        clab="Vertical shift (samples)",png="gxs1i")
  plot3(gx,s2,cmin=-2.0,cmax=2.0,cmap=jetFill(0.3),
        clab="Inline shift (samples)",png="gxs2i")
  plot3(gx,s3,cmin=-1.0,cmax=1.0,cmap=jetFill(0.3),
        clab="Crossline shift (samples)",png="gxs3i")
  gw = fsl.unfault([s1,s2,s3],gx)
  plot3(gx)
  plot3(gw,clab="Amplitude",png="gw")

#undo the faulting in a 3D seismic image
#this method is supposed to work better for crossing faults
#this method UnfaultS is included in unfault
def goUnfault():
  print "goUnfault ..."
  gx = readImage(gxfile)
  if not plotOnly:
    fw = zerofloat(n1,n2,n3)
    lof = LocalOrientFilter(8.0,4.0,4.0)
    et = lof.applyForTensors(gx)
    et.setEigenvalues(0.002,1.0,1.0)

    wp = fillfloat(1.0,n1,n2,n3)
    mk = zerofloat(n1,n2,n3)
    skins = readSkins(fssbase)
    fsc = FaultSlipConstraints(skins)
    sp = fsc.screenPoints(wp)
    uf = UnfaultS(8.0,4.0)
    uf.setIters(200)
    uf.setTensors(et)
    np =  20*len(sp[0][0])
    scale = (n1*n2*n3/np)
    mul(sp[3][0],scale,sp[3][0])
    [t1,t2,t3] = uf.findShifts(sp,wp)
    [t1,t2,t3] = uf.convertShifts(40,[t1,t2,t3])
    gw = zerofloat(n1,n2,n3)
    uf.applyShifts([t1,t2,t3],gx,gw)
    writeImage(gwfile,gw)
    writeImage(ft1file,t1)
    writeImage(ft2file,t2)
    writeImage(ft3file,t3)
  else :
    gw = readImage(gwfile)
    #t1 = readImage(ft1file)
    #t2 = readImage(ft2file)
    #t3 = readImage(ft3file)
    skins = readSkins(fssbase)
  plot3(gx,png="gx")
  plot3(gw,clab="Amplitude",png="gw")
  plot3(gx,skins=skins,smax=20.0,png="skinss1")
  '''
  plot3(gx,t1,cmin=-10,cmax=10,cmap=jetFill(0.3),
        clab="Vertical shift (samples)",png="gxs1")
  plot3(gx,t2,cmin=-2.0,cmax=2.0,cmap=jetFill(0.3),
        clab="Inline shift (samples)",png="gxs2")
  plot3(gx,t3,cmin=-1.0,cmax=1.0,cmap=jetFill(0.3),
        clab="Crossline shift (samples)",png="gxs3")
  '''

#flatten the unfaulted seismic volume to get all seismic horizons
#this flattening method is included in flat
def goFlatten():
  print "go flattening..."
  gx = readImage(gxfile)
  gw = readImage(gwfile)
  if not plotOnly:
    p2 = zerofloat(n1,n2,n3)
    p3 = zerofloat(n1,n2,n3)
    ep = zerofloat(n1,n2,n3)
    sigma1,sigma2,sigma3,pmax=6,3,3,5
    lsf = LocalSlopeFinder(sigma1,sigma2,sigma3,pmax)
    lsf.findSlopes(gw,p2,p3,ep)
    ep = pow(ep,4)
    fl = Flattener3()
    fl.setWeight1(0.1)
    fl.setIterations(0.01,200)
    fm = fl.getMappingsFromSlopes(s1,s2,s3,p2,p3,ep)
    gf = fm.flatten(gw) # flattened seismic volume
    x1 = fm.x1          # horizon volume by flattening 
    writeImage(gffile,gf)
    writeImage(x1file,x1)
  else:
    gf = readImage(gffile)
  # plot3(gx)
  plot3(gw)
  plot3(gf,png="gf")

def goCorrection(sig1,sig2,k,smax,strain,w1):
  if not plotOnly:
    if(k==1):
      fx = readImage(gffile)
      x1 = readImage(x1file)
    else:
      fx = readImage("gc" +str(k))
      x1 = readImage("x1c"+str(k))
    p2 = zerofloat(n1,n2,n3)
    p3 = zerofloat(n1,n2,n3)
    ep = zerofloat(n1,n2,n3)
    lsf = LocalSlopeFinder(sig1,sig2)
    lsf.findSlopes(fx,p2,p3,ep);
    wp = zerofloat(n2,n3)
    for i3 in range(n3):
      for i2 in range(n2):
        wp[i3][i2] = sum(ep[i3][i2])/n1
    fs = zerofloat(n1,n2,n3)
    rgf = RecursiveExponentialFilter(1)
    rgf.apply1(fx,fs)
    gc = GlobalCorrelationFinder(-smax,smax)
    gc.setStrainMax(strain)
    ks = gc.getTraceIndexes(5,5,120,0.2,wp)
    ts = gc.findCorrelations(ks,fs)
    ep = pow(ep,8.0)
    fl = Flattener3Dw()
    fl.setWeight1(w1)
    fl.setIterations(0.01,200)
    fm = fl.getMappingsFromSlopesAndCorrelations(s1,s2,s3,0.001,p2,p3,ep,ks,ts)
    gx = fm.flatten(fx)
    xc1 = fm.x1
    fl.updateHorizonVolume(x1,xc1)
    writeImage("gc" +str(k+1),gx)
    writeImage("x1c"+str(k+1),xc1)
  else:
    #g2 = readImage("gc2")
    #g3 = readImage("gc3")
    g4 = readImage("gc6")
  #plot3(fx)
  plot3(g4,png='gc6')

def goHorizons():
  gx = readImage(gxfile)
  fl = Flattener3Dw()
  sk = readSkins(fssbase)
  if not plotOnly:
    x1 = readImage("x1c6")
    t1 = readImage(ft1file)
    t2 = readImage(ft2file)
    t3 = readImage(ft3file)
    ut = fl.rgtFromHorizonVolume(s1,x1)
    ux = zerofloat(n1,n2,n3)
    uf = UnfaultS(4,4)
    uf.applyShiftsX([t1,t2,t3],ut,ux)
    writeImage("uxx",ux)
  else:
    ux = readImage("uxx")
  xx = fl.horizonVolumeFromRgt(s1,ux)
  k1s = rampfloat(150,10,50)
  #k1s = [40,60,80,96,123,132,140,160,185]
  hzs = []
  for h1 in k1s:
    h = slice23(int(h1),xx)
    hzs.append(h)
  hu = HorizonUpdater3()
  hzu = copy(hzs)
  '''
  if not plotOnly:
    lof = LocalOrientFilterP(3,1,1)
    lsf = LocalSlopeFinder(3,1,1)
    u1 = zerofloat(n1,n2,n3)
    p2 = zerofloat(n1,n2,n3)
    p3 = zerofloat(n1,n2,n3)
    epg = zerofloat(n1,n2,n3)
    epu = zerofloat(n1,n2,n3)
    lof.applyForNormalPlanar(ux,u1,p2,p3,epu)
    lsf.findSlopes(gx,p2,p3,epg)
    epg = pow(epg,6)
    epu = pow(epu,6)
    hzu = copy(hzs) 
    epu = mul(epu,0.05)
    hu.updateHorizons(p2,p3,epg,epu,hzu)
    writeImage("hzu",hzu)
  else:
    hzu = readImage3D(n2,n3,len(k1s),"hzu")
  '''
  hx = hu.horizonImage(n1,n2,n3,0,k1s,hzu)

  x1s  = rampfloat(0,1,n1)
  cp = ColorMap(min(k1s),max(k1s),ColorMap.JET)
  rgb = cp.getRgbFloats(x1s)
  nh = len(k1s)
  tgs,tgc = [],[]
  hu = HorizonUpdater3()
  #tgc = hu.cutAwaryView(s2,s3,150,n1-1,380,n2-1,0,100,hzu)
  #tgc[nh-1] = TriangleGroup(True,s3,s2,hzu[nh-1])
  for ih in range(nh):
    k1 = int(k1s[ih])
    tg = TriangleGroup(True,s3,s2,hzu[ih])
    r,g,b=rgb[k1*3],rgb[k1*3+1],rgb[k1*3+2]
    #tgc[ih].setColor(Color(r,g,b))
    tg.setColor(Color(r,g,b))
    tgs.append(tg)
    #plot3(gx,htgs=[tg],png="surf"+str(ih))
  '''
    tgs.append(tg)
  plot3(gx,htgs=tgs,png="surfs")
  #plot3(gx,k1=153,k2=388,k3=89,#htgs=tgc,ae=[145,30],tx=[0,-0.2,-0.65],png="surfc")
  '''
  plot3(gx,ux,cmin=-5,cmax=n1*0.85,cmap=jetFill(0.8),png="ux")
  plot3(gx,g=hx,cmap=jetFillExceptMin(1.0),cmin=min(k1s),cmax=max(k1s),png="hx")
  plot3(gx,htgs=tgs,png="surf")

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

def plotPoints(xyz):
  #bb = BoundingBox(f3,f2,f1,l3,l2,l1)
  states = StateSet.forTwoSidedShinySurface(Color.CYAN);
  pg = EightEllipsoidsInCube()
  pg.setStates(states)
  world = World()
  world.addChild(pg)
  sf = SimpleFrame(world)
def plot3(f,g=None,cmin=None,cmax=None,cmap=None,clab=None,cint=None,
          xyz=None,htgs=None,cells=None,skins=None,skinx=None,smax=0.0,
          links=False,curve=False,trace=False,png=None):
  d1,d2,d3 = s1.delta,s2.delta,s3.delta
  f1,f2,f3 = s1.first,s2.first,s3.first
  l1,l2,l3 = s1.last,s2.last,s3.last
  n3 = len(f)
  n2 = len(f[0])
  n1 = len(f[0][0])
  sf = SimpleFrame(AxesOrientation.XRIGHT_YIN_ZDOWN)
  cbar = None
  if g==None:
    ipg = sf.addImagePanels(s1,s2,s3,f)
    if cmap!=None:
      ipg.setColorModel(cmap)
    if cmin!=None and cmax!=None:
      ipg.setClips(cmin,cmax)
    else:
      ipg.setClips(-3.0,3.0)
    if clab:
      cbar = addColorBar(sf,clab,cint)
      ipg.addColorMapListener(cbar)
  else:
    ipg = ImagePanelGroup2(s1,s2,s3,f,g)
    ipg.setClips1(-3.0,3.0)
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
  if xyz:
    pg = PointGroup(2.0,xyz)
    #bb = BoundingBox(f3,f2,f1,l3,l2,l1)
    #pg = SeismicEllipsoid.EightEllipsoidsInCube()
    #states = StateSet.forTwoSidedShinySurface(Color.CYAN);
    #pg.setStates(states);
    #World world = new World();
    #sf.world.addChild(pg);
    #SimpleFrame sf = new SimpleFrame(world);

    ss = StateSet()
    cs = ColorState()
    cs.setColor(Color.CYAN)
    ss.add(cs)
    pg.setStates(ss)
    #ss = StateSet()
    #ps = PointState()
    #ps.setSize(5.0)
    #ss.add(ps)
    #pg.setStates(ss)
    sf.world.addChild(pg)
  if htgs:
    for htg in htgs:
      #htg.setStates(ss)
      sf.world.addChild(htg)
  if cells:
    ss = StateSet()
    lms = LightModelState()
    lms.setTwoSide(True)
    ss.add(lms)
    ms = MaterialState()
    ms.setSpecular(Color.GRAY)
    ms.setShininess(100.0)
    ms.setColorMaterial(GL_AMBIENT_AND_DIFFUSE)
    ms.setEmissiveBack(Color(0.0,0.0,0.5))
    ss.add(ms)
    cmap = ColorMap(0.0,1.0,ColorMap.JET)
    xyz,uvw,rgb = FaultCell.getXyzUvwRgbForLikelihood(0.5,cmap,cells,False)
    qg = QuadGroup(xyz,uvw,rgb)
    qg.setStates(ss)
    sf.world.addChild(qg)
  if skinx:
    sg = Group()
    ss = StateSet()
    lms = LightModelState()
    #lms.setTwoSide(False)
    lms.setTwoSide(True)
    ss.add(lms)
    ms = MaterialState()
    ms.setSpecular(Color.GRAY)
    ms.setShininess(100.0)
    ms.setColorMaterial(GL_AMBIENT_AND_DIFFUSE)
    if not smax:
      ms.setEmissiveBack(Color(0.0,0.0,0.0))
    ss.add(ms)
    sg.setStates(ss)
    for skin in skinx:
      #cmap = ColorMap(0.0,1.0,ColorMap.JET)
      cmap = ColorMap(0.0,180,hueFill(1.0))
      #tg = skin.getTriMesh(cmap)
      #tg = skin.getTriMeshStrike(cmap)
      qg = skin.getQuadMeshStrike(cmap)
      sg.addChild(qg)
    sf.world.addChild(sg)

  if skins:
    sg = Group()
    ss = StateSet()
    lms = LightModelState()
    lms.setTwoSide(True)
    ss.add(lms)
    ms = MaterialState()
    ms.setSpecular(Color.GRAY)
    ms.setShininess(100.0)
    ms.setColorMaterial(GL_AMBIENT_AND_DIFFUSE)
    if not smax:
      ms.setEmissiveBack(Color(0.0,0.0,0.5))
    ss.add(ms)
    sg.setStates(ss)
    size = 2.0
    if links:
      size = 0.5 
    for skin in skins:
      if smax>0.0: # show fault throws
        cmap = ColorMap(0.0,smax,ColorMap.JET)
        xyz,uvw,rgb = skin.getCellXyzUvwRgbForThrow(size,cmap,False)
      else: # show fault likelihood
        cmap = ColorMap(0.0,180,ColorMap.JET)
        xyz,uvw,rgb = skin.getCellXyzUvwRgbForStrike(size,cmap,True)
      qg = QuadGroup(xyz,uvw,rgb)
      qg.setStates(None)
      sg.addChild(qg)
      if curve or trace:
        cell = skin.getCellNearestCentroid()
        if curve:
          xyz = cell.getFaultCurveXyz()
          pg = PointGroup(0.5,xyz)
          sg.addChild(pg)
        if trace:
          xyz = cell.getFaultTraceXyz()
          pg = PointGroup(0.5,xyz)
          sg.addChild(pg)
      if links:
        xyz = skin.getCellLinksXyz()
        lg = LineGroup(xyz)
        sg.addChild(lg)
    sf.world.addChild(sg)
  #ipg.setSlices(590,200,49)
  ipg.setSlices(50,n2-1,0)
  #ipg.setSlices(115,25,167)
  if cbar:
    sf.setSize(987,800)
  else:
    sf.setSize(850,800)
  vc = sf.getViewCanvas()
  vc.setBackground(Color.WHITE)
  radius = 0.5*sqrt(n1*n1+n2*n2+n3*n3)
  ov = sf.getOrbitView()
  zscale = 1.0*max(n2*d2,n3*d3)/(n1*d1)
  ov.setAxesScale(1.0,1.0,zscale)
  ov.setScale(1.6)
  ov.setWorldSphere(BoundingSphere(BoundingBox(f3,f2,f1,l3,l2,l1)))
  ov.setTranslate(Vector3(0.0,-1.0,-0.08))
  ov.setAzimuthAndElevation(105,86)
  sf.setVisible(True)
  if png and pngDir:
    sf.paintToFile(pngDir+png+".png")
    if cbar:
      cbar.paintToPng(137,1,pngDir+png+"cbar.png")

#############################################################################
run(main)
