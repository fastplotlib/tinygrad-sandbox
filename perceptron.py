"""
perceptron_demo.py — train a small perceptron in tinygrad on pygfx's own wgpu
device, and display each layer's weights live as a gfx.Image, updating the
textures from GPU buffers (no .numpy() per frame).

RUN THE SMOKE TEST FIRST (python gpu_smoke_test.py). If that fails, this will too.

Notes / caveats (read before debugging):
  * pygfx 0.16 / wgpu 0.31 assumed. The two integration seams are:
      - get_shared().device  -> the wgpu-py GPUDevice pygfx renders with
      - gfx.Texture._wgpu_object -> underlying wgpu texture (lazily created)
    If a pygfx update renames these, that's where to look.
  * WebGPU requires copy_buffer_to_texture bytes_per_row % 256 == 0. We size the
    visualization texture to a fixed WIDTH whose RGBA8 row (WIDTH*4) is a
    multiple of 256 (WIDTH=64 -> 256), and pack each weight matrix into that.
  * Float32-filterable may be unsupported on some backends; we display via
    rgba8unorm (normalized weights -> 0..255), which is universally safe.
"""
import numpy as np
import wgpu
import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop
from pygfx.renderers.wgpu import get_shared

from tinygrad import Tensor, nn, Context
import tg_wgpu_shared as S

# ----- fixed visualization tile size (keeps bytes_per_row aligned) -----
TILE_W = 64          # 64 * 4 bytes (rgba8) = 256  -> satisfies the 256 rule
TILE_H = 64

# ---------------- 1. shared device ----------------
canvas = RenderCanvas(size=(900, 360), title="tinygrad layers on pygfx device")
renderer = gfx.renderers.WgpuRenderer(canvas)
shared = get_shared()
wdev = shared.device                      # the wgpu-py GPUDevice pygfx uses
dev = S.install_shared_webgpu(wdev, name="WEBGPU")

# ---------------- 2. perceptron in tinygrad ----------------
class MLP:
    def __init__(self):
        self.l1 = nn.Linear(2, 16)
        self.l2 = nn.Linear(16, 8)
        self.l3 = nn.Linear(8, 1)
    def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        return self.l3(x).sigmoid()

net = MLP()
# nn.Linear builds weights on Device.DEFAULT (CUDA here), not WEBGPU. We must
# move them BEFORE building the optimizer (optim.py: self.device =
# params[0].device). Reassign each layer attribute to a WEBGPU copy via .to(),
# which returns a new WEBGPU-backed tensor, so net.lN.weight/bias themselves
# point at WEBGPU (not just list elements from get_parameters()).
for layer in (net.l1, net.l2, net.l3):
    layer.weight = layer.weight.to("WEBGPU")
    if getattr(layer, "bias", None) is not None:
        layer.bias = layer.bias.to("WEBGPU")
opt = nn.optim.Adam(nn.state.get_parameters(net), lr=0.02)
X = Tensor([[0,0],[0,1],[1,0],[1,1]], dtype="float32", device="WEBGPU")
Y = Tensor([[0.],[1.],[1.],[0.]], device="WEBGPU")

def train_step():
    # this tinygrad gates the optimizer on the TRAINING ContextVar
    # (tinygrad/nn/optim.py checks `if not TRAINING`). Context(TRAINING=1)
    # sets it for the block and restores on exit.
    with Context(TRAINING=1):
        opt.zero_grad()
        loss = ((net(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    return loss

# ---------------- 3. layer weights -> normalized rgba8 GPU buffer ----------------
def weights_to_rgba_tensor(w: Tensor) -> Tensor:
    """Map a weight matrix to a TILE_H x TILE_W x 4 uint8 image, all on-GPU.

    Normalizes to 0..1, nearest-pads/crops into the tile, writes grayscale RGBA.
    Returns a realized uint8 Tensor on WEBGPU whose buffer we copy to the texture.
    """
    wmin = w.min(); wmax = w.max()
    n = (w - wmin) / (wmax - wmin + 1e-8)         # 0..1, shape (out,in)
    out, inp = w.shape
    # place into a TILE grid (top-left), rest zeros
    canvas_t = Tensor.zeros(TILE_H, TILE_W, device="WEBGPU").contiguous()
    h = min(out, TILE_H); wd = min(inp, TILE_W)
    canvas_t[:h, :wd] = n[:h, :wd]
    g = (canvas_t * 255).cast("uint8")            # grayscale
    a = Tensor.full((TILE_H, TILE_W), 255, dtype="uint8", device="WEBGPU")
    rgba = Tensor.stack(g, g, g, a, dim=-1)       # (H,W,4) uint8
    return rgba.contiguous().realize()

# ---------------- 4. pygfx images, one per layer ----------------
layers = [net.l1, net.l2, net.l3]
images, textures = [], []
x_off = 30
for layer in layers:
    # data=None: no local data, so pygfx has NO pending upload that could
    # overwrite our GPU copy. COPY_DST makes it a valid copy target; our
    # copy_buffer_to_texture is then the sole writer.
    tex = gfx.Texture(size=(TILE_W, TILE_H, 1), dim=2, format=wgpu.TextureFormat.rgba8unorm,
                      usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING)
    img = gfx.Image(gfx.Geometry(grid=tex), gfx.ImageBasicMaterial(clim=(0, 255)))
    img.local.position = (x_off, 60, 0)
    img.local.scale = (3, 3, 1)
    images.append(img); textures.append(tex)
    x_off += TILE_W * 3 + 40

scene = gfx.Scene()
scene.add(gfx.Background(None, gfx.BackgroundMaterial("#1a1a1a")))
for img in images: scene.add(img)
camera = gfx.OrthographicCamera(900, 360)
camera.local.position = (450, 180, 0)

# ---------------- 5. on-GPU copy: tinygrad buffer -> pygfx texture ----------------
# pygfx's own ensure_wgpu_object() deterministically allocates a texture's
# underlying wgpu object via device.create_texture — no render frame, no upload
# hack. Because we built each gfx.Texture with data!=None, pygfx adds COPY_DST
# to its usage (see engine/update.py), so it is a valid copy target.
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object

def copy_layer_to_texture(rgba_tensor: Tensor, tex: gfx.Texture):
    src = S.tinygrad_buffer_handle(rgba_tensor)        # wgpu GPUBuffer, same device
    wgpu_tex = ensure_wgpu_object(tex)                 # creates _wgpu_object if absent
    if wgpu_tex is None:
        raise RuntimeError("pygfx did not materialize the wgpu texture object")
    bytes_per_row = TILE_W * 4                          # 256, aligned
    enc = wdev.create_command_encoder()
    enc.copy_buffer_to_texture(
        {"buffer": src, "offset": 0, "bytes_per_row": bytes_per_row, "rows_per_image": TILE_H},
        {"texture": wgpu_tex, "mip_level": 0, "origin": (0, 0, 0)},
        (TILE_W, TILE_H, 1))
    wdev.queue.submit([enc.finish()])

# ---------------- 6. animation loop ----------------
step = 0
def animate():
    global step
    loss = train_step()
    if step % 3 == 0:
        for layer, tex in zip(layers, textures):
            rgba = weights_to_rgba_tensor(layer.weight)
            copy_layer_to_texture(rgba, tex)
    if step % 100 == 0:
        print(f"step {step}  loss {loss.item():.4f}")
    step += 1
    renderer.render(scene, camera)
    canvas.request_draw()

canvas.request_draw(animate)

if __name__ == "__main__":
    print("training + visualizing; close the window to stop, then see inference.")
    loop.run()
    # inference runs outside any training context by default
    print("final predictions:", net(X).numpy().ravel().tolist())
