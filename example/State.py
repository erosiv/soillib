import numpy as np
import soillib as soil
import soiltrace as trace
import skimage.transform as skt
import state

class View:

  draw_gizmo = True
  draw_grid = True
  draw_axis = False
  draw_cube = False

  mode = 0
  overlay = 2

  invert = 4
  permute = 2

  gamma = 2.0

  def __init__(self):

    self.Controller = None
    self.tracer = None
    self.samples = 0
    
  def clear(self):
    self.tracer.clear([0, 0, 0, 0])
    self.samples = 0

  def setup(self, size):
    self.Controller = state.CameraController(size)
    self.tracer = trace.tracer(self.Controller.camera())
    self.clear()

  def do_overlay(self, field, Scene):

    if field is None: return

    if self.overlay == 1:
      self.tracer.overlay_logscale(Scene.scene, field, Scene.logscale, [1, 1, 1])
    elif self.overlay == 2:
      self.tracer.overlay_sigmoid(Scene.scene, field, Scene.getSigScale(), [0, 0, 0], [1, 1, 1])
  
  def draw(self, texture, Scene):

    if Scene.updated:
      self.clear()
      Scene.updated = False

    if self.mode == 0:
      self.tracer.render_normal(Scene.scene, self.invert, self.permute)
    elif self.mode == 1:
      self.tracer.render_pos(Scene.scene)
    elif self.mode == 2:
      self.tracer.render(Scene.scene)
    elif self.mode == 3:
      self.tracer.render_depth(Scene.scene)
    elif self.mode == 4:
      self.tracer.render_matte(Scene.scene)

    # overlay rendering is done here! this should be extracted for more flexibility ...
    if Scene.data != None and Scene.show_discharge and self.mode != 2 and self.mode != 4:
      #self.do_overlay(Scene.data.debris, Scene)
      self.do_overlay(Scene.data.discharge, Scene)

    if self.mode == 2:
      self.samples += 1
      self.tracer.blit(texture, self.gamma)
    else:
      self.tracer.blit(texture, 1.0)
    
  def setDrawAxis(self, draw_axis):
    self.draw_axis = draw_axis

  def setDrawGrid(self, draw_grid):
    self.draw_grid = draw_grid

  def setDrawGizmo(self, draw_gizmo):
    self.draw_gizmo = draw_gizmo

  def setDrawCube(self, draw_cube):
    self.draw_cube = draw_cube

class Scene:

  def __init__(self):

    self.index = [0, 0]
    self.scene = None
    
    self.map = None
    self.data = None

    self.scale = None
    self.res = None
    self.hext = None
    self.height = None

    self.point_pos = None
    self.point_nrm = None

    self.updated = False

    self.tiff = None

    self.sun_angle_x = 67.5 / 360.0 * 2.0 * np.pi
    self.sun_angle_y = 315.0 / 360.0 * 2.0 * np.pi

    self.logscale = 32
    self.sigscale = 3
    self.sedscale = 5000.0

    self.show_height = True
    self.color_bedrock = [130/255, 124/255, 115/255]
    self.show_sediment = True
    self.color_sediment = [140/255, 120/255, 100/255]
    self.show_discharge = True
    self.color_discharge = [30/255, 70/255, 110/255]

  def getSigScale(self):
    return 1E6*1E6/self.index.elem()*self.sigscale

  def setup(self, res, scale, hext, map, data):

    if self.scene != None:
      self.scene.free_memory()

    self.map = map
    self.data = data
  
    self.scale = scale
    self.res = res
    self.hext = hext

    if not self.map is None:
      self.height[:] = 0.0
      if self.show_height: soil.add(self.height, self.map.height)
      if self.show_sediment: soil.add(self.height, self.map.sediment)

    self.scene = trace.scene(self.height, [res[1], res[0]], scale, hext)
    self.scene.sun.direction = self.direction()

    if not self.map is None:
      if self.show_height: trace.shade_base(self.scene.albedo, self.color_bedrock)
      if self.show_sediment: trace.shade_layer(self.scene.albedo, self.map.sediment, self.color_sediment, self.sedscale)
    
    if not self.data is None:
      if self.show_discharge: trace.shade_discharge(self.scene.albedo, self.data.discharge, self.getSigScale(), self.color_discharge)

    self.updated = True

  def blank(self):

    self.index = soil.index([64, 64])
    self.height = soil.buffer(soil.float32, 64*64, soil.gpu)
    soil.set(self.height, 0.0)

    res = np.array([self.index[1], self.index[0]])
    scale = np.array([1, 1, 1])
    hext = np.array([0, 0])

    self.setup(res, scale, hext, None, None)

  def setColor(self, albedo):

    soil.set(self.scene.albedo, albedo)
    self.updated = True

  def direction(self):
    theta = 0.0
    psi = self.sun_angle_y
    phi = self.sun_angle_x

    return [
      np.cos(theta)*np.cos(psi),
      -np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi),
      np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi)
    ]

  def setAngleX(self, angle):
    self.sun_angle_x = 2.0 * np.pi * float(angle) / 360.0
    self.scene.sun.direction = self.direction()
    self.updated = True

  def setAngleY(self, angle):
    self.sun_angle_y = 2.0 * np.pi * float(angle) / 360.0
    self.scene.sun.direction = self.direction()
    self.updated = True

  def setScatter(self, scatter):
    self.scene.scatter = bool(scatter)
    self.updated = True

  def setSigmaT(self, sigma_t):
    self.scene.sigma_t = sigma_t
    self.updated = True

  def setSigmaS(self, sigma_s):
    self.scene.sigma_s = sigma_s
    self.updated = True

  def setOverlayLogScale(self, logscale):
    self.logscale = logscale

  def setOverlaySigScale(self, sigscale):
    self.sigscale = sigscale
    if self.show_height: trace.shade_base(self.scene.albedo, self.color_bedrock)
    if self.show_sediment: trace.shade_layer(self.scene.albedo, self.map.sediment, self.color_sediment, self.sedscale)
    if self.show_discharge: trace.shade_discharge(self.scene.albedo, self.data.discharge, self.getSigScale(), self.color_discharge)

    self.updated = True

  '''
  def rbf_resample(self, height, N = 8192*2):

    index = self.index
    scale = self.scale

    pointraw = soil.pointcloud_sample(height, index, N)
    normals = soil.pointcloud_normal(height, pointraw, index, scale)

    print("FITTING RBF")
    rbf = soil.rbf()

    rbf.shape = 0.25 * 256/index[0]
    rbf.lrate_w = 0.001
#    rbf.lrate_s = 0.00001
#    rbf.lrate_c = 0.0

    rbf.init(index, 4096)
    rbf.fit(pointraw, 1024)
    print("DONE")

    soil.pointcloud_scale(pointraw, index, scale)
    self.point_pos = pointraw.cpu().numpy(soil.index([N]))
    self.point_nrm = normals.cpu().numpy(soil.index([N]))

    self.updated = True
    return rbf.sample(index)
  '''

class State:

  def __init__(self):

    self.title = "soilmachine - gui alpha"
    self.version = "soilmachine v0.0"

    self.ms = 0.0
    self.fps = 0.0

    # Rendering Parameters

    self.View = View()
    self.Scene = Scene()
    self.Erosion = state.Erosion()

  def setupSize(self, size):
    self.View.setup(size)

  def saveTIFF(self, filename):

    if filename == "":
      return

    index = soil.index([self.Scene.index[1], self.Scene.index[0]])
    buffer = soil.buffer(soil.float32, index.elem())
    
    buffer.gpu()
    self.Scene.height.gpu()
    soil.set(buffer, self.Scene.height)
    buffer.cpu()

    tiff = soil.geotiff(buffer, index)
    tiff.meta.scale = [self.Scene.scale[0], self.Scene.scale[2], self.Scene.scale[1]]
    tiff.write(filename)