"""
cnn_mnist_grid.py — train a CNN on MNIST on pygfx's own wgpu device and show,
for each digit class 0-9, how its canonical (per-class average) image activates
the layers. Live, all updated from GPU buffers (no per-frame .numpy()).

Grid: 10 rows (digit 0..9). Each row:
    [ input avg ]  [ conv1 act x4 ]  [ conv2 act x4 ]
Row + column headers drawn with gfx.Text.

Representative per class = mean of all train images with that label (a smooth
"prototype" digit). Its activations show the network's response to the canonical
form of each class. Prototypes are computed once at setup.

Same shared-device machinery as before: install_shared_webgpu sets WEBGPU
default; params/optimizer/data/intermediates all on the shared device;
activations are separate realizes that don't touch the training graph.
"""
import numpy as np
import wgpu
import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop
from pygfx.renderers.wgpu import get_shared
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object

from tinygrad import Tensor, nn, Context
import tg_wgpu_shared as S

N_C1, N_C2 = 4, 4          # how many conv1 / conv2 maps to show per digit
TILE = 64                  # 64*4 = 256 bytes/row -> WebGPU aligned

# ---------------- 1. shared device ----------------
canvas = RenderCanvas(size=(1180, 760), title="MNIST per-class activations (pygfx device)")
renderer = gfx.renderers.WgpuRenderer(canvas)
wdev = get_shared().device
dev = S.install_shared_webgpu(wdev, name="WEBGPU")

# ---------------- 2. data + per-class prototypes ----------------
from tinygrad.nn.datasets import mnist
X_train, Y_train, X_test, Y_test = mnist()
X_train = X_train.float() / 255.0
X_test  = X_test.float() / 255.0

# one-time host read of labels to select per-class indices, then average on GPU
_yn = Y_train.numpy()
protos = []
for d in range(10):
    idx = np.where(_yn == d)[0]
    sub = X_train[Tensor(idx.astype(np.int32))]          # (count,1,28,28) on WEBGPU
    protos.append(sub.mean(axis=0, keepdim=True))         # (1,1,28,28)
PROTO = Tensor.cat(*protos, dim=0).realize()              # (10,1,28,28) on WEBGPU

# ---------------- 3. CNN ----------------
class CNN:
    def __init__(self):
        self.c1 = nn.Conv2d(1, 8, 5)
        self.c2 = nn.Conv2d(8, 16, 5)
        self.l1 = nn.Linear(16 * 4 * 4, 10)
    def features1(self, x): return self.c1(x).relu()             # (N,8,24,24)
    def features2(self, x): return self.c2(x).relu()             # x=pooled f1 -> (N,16,8,8)
    def __call__(self, x):
        f1 = self.features1(x).max_pool2d(2)
        f2 = self.features2(f1).max_pool2d(2)
        return self.l1(f2.flatten(1))

net = CNN()
for layer in (net.c1, net.c2, net.l1):
    layer.weight = layer.weight.to("WEBGPU")
    if getattr(layer, "bias", None) is not None:
        layer.bias = layer.bias.to("WEBGPU")
opt = nn.optim.Adam(nn.state.get_parameters(net), lr=1e-3)

BS = 128
def train_step():
    with Context(TRAINING=1):
        opt.zero_grad()
        idx = Tensor.randint(BS, high=X_train.shape[0])
        loss = net(X_train[idx]).sparse_categorical_crossentropy(Y_train[idx]).backward()
        opt.step()
    return loss

# ---------------- 4. tensor -> rgba8 tile (on-GPU) ----------------
def map2d_to_rgba(m: Tensor) -> Tensor:
    h0, w0 = m.shape[-2], m.shape[-1]
    m = m.reshape(h0, w0)
    mn = m.min(); mx = m.max()
    n = (m - mn) / (mx - mn + 1e-8)
    rh, rw = max(1, TILE // h0), max(1, TILE // w0)
    up = n.reshape(h0, 1, w0, 1).expand(h0, rh, w0, rw).reshape(h0 * rh, w0 * rw)
    canvas_t = Tensor.zeros(TILE, TILE, device="WEBGPU").contiguous()
    hh, ww = min(up.shape[0], TILE), min(up.shape[1], TILE)
    canvas_t[:hh, :ww] = up[:hh, :ww]
    g = (canvas_t * 255).cast("uint8")
    a = Tensor.full((TILE, TILE), 255, dtype="uint8", device="WEBGPU")
    return Tensor.stack(g, g, g, a, dim=-1).contiguous().realize()

# ---------------- 5. text label helper (defensive across pygfx versions) -------
def make_label(text, pos, size=14, color="#dddddd", anchor="middle-center"):
    try:
        t = gfx.Text(text=str(text), font_size=size, screen_space=False, anchor=anchor,
                     material=gfx.TextMaterial(color=color))
        t.local.position = pos
        scene.add(t)
        return t
    except Exception as e:
        print(f"(label '{text}' skipped: {type(e).__name__})")
        return None

# ---------------- 6. build labeled grid ----------------
scene = gfx.Scene()
scene.add(gfx.Background(None, gfx.BackgroundMaterial("#141414")))

SCALE = 0.82
TW = TILE * SCALE
GAP = 8
COL_GAP = 22                        # gap between the three column groups
LEFT = 70                           # room for row labels
TOP = 630                           # top row tile-center y (lowered so headers clear the tiles)
ROW_PITCH = TW + 10                 # vertical distance between row centers

# column x positions: input | conv1 x N_C1 | conv2 x N_C2 | output
col_x = []
x = LEFT
col_x.append(x); x += TW + COL_GAP          # input
c1_start = x
for _ in range(N_C1): col_x.append(x); x += TW + GAP
x += COL_GAP - GAP
c2_start = x
for _ in range(N_C2): col_x.append(x); x += TW + GAP
x += COL_GAP - GAP
out_x = x                                    # output strip column
col_x.append(out_x)

# column headers — positioned at each column GROUP's center-x (col_x[] are tile
# centers, and gfx.Image/Text both anchor at center), sitting above the top row.
header_y = TOP + TW / 2 + 24
input_cx = col_x[0]
c1_cx = (col_x[1] + col_x[N_C1]) / 2                 # midpoint of conv1 tiles
c2_cx = (col_x[1 + N_C1] + col_x[N_C1 + N_C2]) / 2   # midpoint of conv2 tiles
out_cx = out_x
make_label("input", (input_cx, header_y, 1), size=15, color="#88ccff", anchor="bottom-center")
make_label("conv1 activations", (c1_cx, header_y, 1), size=15, color="#88ffaa", anchor="bottom-center")
make_label("conv2 activations", (c2_cx, header_y, 1), size=15, color="#ffaa88", anchor="bottom-center")
make_label("output", (out_cx, header_y, 1), size=15, color="#ffdd66", anchor="bottom-center")

# rows: one per digit class
row_y = []
y = TOP
tiles = {"input": [], "c1": [], "c2": [], "out": []}
for d in range(10):
    row_y.append(y)
    make_label(str(d), (LEFT - TW/2 - 16, y, 1), size=22, color="#ffffff", anchor="middle-right")
    # input tile
    def add_tile(cx, cy):
        tex = gfx.Texture(size=(TILE, TILE, 1), dim=2, format=wgpu.TextureFormat.rgba8unorm,
                          usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING)
        img = gfx.Image(gfx.Geometry(grid=tex), gfx.ImageBasicMaterial(clim=(0, 255)))
        # negative y-scale flips the image vertically. gfx.Image's origin is a
        # corner and the quad spans +y from position, so the flip mirrors about
        # position.y; add TW to position.y to keep the tile in the same spot.
        img.local.position = (cx, cy + TW, 0)
        img.local.scale = (SCALE, -SCALE, 1)
        scene.add(img)
        return tex
    tiles["input"].append(add_tile(col_x[0], y))
    tiles["c1"].append([add_tile(col_x[1 + i], y) for i in range(N_C1)])
    tiles["c2"].append([add_tile(col_x[1 + N_C1 + i], y) for i in range(N_C2)])
    tiles["out"].append(add_tile(out_x, y))
    y -= ROW_PITCH

camera = gfx.OrthographicCamera(1180, 760)
camera.local.position = (590, 380, 0)

# ---------------- 7. on-GPU copy ----------------
def copy_to_texture(rgba_tensor, tex):
    src = S.tinygrad_buffer_handle(rgba_tensor)
    wgpu_tex = ensure_wgpu_object(tex)
    if wgpu_tex is None:
        raise RuntimeError("pygfx did not materialize the wgpu texture object")
    enc = wdev.create_command_encoder()
    enc.copy_buffer_to_texture(
        {"buffer": src, "offset": 0, "bytes_per_row": TILE * 4, "rows_per_image": TILE},
        {"texture": wgpu_tex, "mip_level": 0, "origin": (0, 0, 0)},
        (TILE, TILE, 1))
    wdev.queue.submit([enc.finish()])

# input prototypes never change -> draw once
for d in range(10):
    copy_to_texture(map2d_to_rgba(PROTO[d, 0]), tiles["input"][d])

# ---------------- 8. animation loop ----------------
# Speed comes from running many optimizer steps per RENDERED frame: the render,
# the 90 texture copies, and the loss readback are all per-frame overhead, so
# amortizing them over STEPS_PER_FRAME steps lets training run far faster while
# the visuals still refresh every frame. We avoid loss.item() (a GPU sync) on
# the hot path and only read it occasionally for the printout.
STEPS_PER_FRAME = 1
step = 0
last_loss = None
def animate():
    global step, last_loss
    for _ in range(STEPS_PER_FRAME):
        last_loss = train_step()        # tensor; not synced here
        step += 1
    # visualize current state once per frame (after the burst of steps)
    a1 = net.features1(PROTO).realize()              # (10,8,24,24)
    a2 = net.features2(a1.max_pool2d(2)).realize()   # (10,16,8,8)
    logits = net(PROTO).realize()                    # (10,10)
    probs = logits.softmax(axis=1).realize()         # (10,10), per-digit class probs
    for d in range(10):
        for i in range(N_C1):
            copy_to_texture(map2d_to_rgba(a1[d, i]), tiles["c1"][d][i])
        for i in range(N_C2):
            copy_to_texture(map2d_to_rgba(a2[d, i]), tiles["c2"][d][i])
        # output: 10 class probabilities as a horizontal strip (bright = predicted)
        copy_to_texture(map2d_to_rgba(probs[d].reshape(1, 10)), tiles["out"][d])
    if step % 500 < STEPS_PER_FRAME:    # print roughly every ~500 steps
        print(f"step {step}  loss {last_loss.item():.4f}")
    renderer.render(scene, camera)
    canvas.request_draw()

canvas.request_draw(animate)

if __name__ == "__main__":
    print("training CNN; grid shows per-class avg digit + conv1/conv2 activations")
    loop.run()
    acc = (net(X_test).argmax(axis=1) == Y_test).mean().item()
    print(f"test accuracy: {acc*100:.2f}%")
