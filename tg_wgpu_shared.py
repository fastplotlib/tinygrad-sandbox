"""
a tinygrad backend that runs on a wgpu-py GPUDevice supplied from
outside (the same device pygfx/wgpu created). No second native library, no Dawn,
no host round-trip for visualization — tinygrad buffers and pygfx textures live
on ONE wgpu-native device, so we can copy_buffer_to_texture on-GPU.

Design:
  * We do NOT use tinygrad's ops_webgpu (that one links Dawn via ctypes).
  * We reuse tinygrad's WGSLRenderer unchanged — tinygrad already emits WGSL.
  * Allocator + Program drive the wgpu-py GPUDevice through its PUBLIC Python API
    (create_buffer / create_shader_module / create_compute_pipeline /
     create_bind_group / queue.write_buffer / command_encoder ...), which is far
    more version-robust than mirroring cffi calls.
  * We construct the Compiled subclass directly and inject it into tinygrad's
    device cache, so Tensor(device="WEBGPU") resolves to OUR shared instance.

This file is import-time safe with no GPU: pass any object exposing the wgpu-py
GPUDevice method surface. test_shared.py exercises it with a faithful mock.
"""
from __future__ import annotations
import functools, struct
from tinygrad.device import Compiled, Allocator, BufferSpec
from tinygrad.renderer.wgsl import WGSLRenderer

# wgpu enum ints we need. Import real ones if wgpu is present; else fall back to
# the spec constants (stable values) so the module imports on a GPU-less box.
try:
    from wgpu import BufferUsage as _BU, MapMode as _MM, ShaderStage as _SS, BufferBindingType as _BBT
    BUF_STORAGE  = int(_BU.STORAGE);  BUF_UNIFORM = int(_BU.UNIFORM)
    BUF_COPY_DST = int(_BU.COPY_DST); BUF_COPY_SRC = int(_BU.COPY_SRC)
    BUF_MAP_READ = int(_BU.MAP_READ)
    MAP_READ     = int(_MM.READ)
    STAGE_COMPUTE = int(_SS.COMPUTE)
    BIND_STORAGE = "storage"; BIND_UNIFORM = "uniform"
except Exception:
    BUF_STORAGE, BUF_UNIFORM = 0x080, 0x040
    BUF_COPY_DST, BUF_COPY_SRC = 0x008, 0x004
    BUF_MAP_READ = 0x001
    MAP_READ = 0x0001
    STAGE_COMPUTE = 0x4
    BIND_STORAGE = "storage"; BIND_UNIFORM = "uniform"

def round_up(x:int, a:int) -> int: return (x + a - 1) // a * a


class SharedWebGPUProgram:
    """Compiles one WGSL kernel against the shared device and dispatches it.

    Binding layout mirrors tinygrad's own ops_webgpu convention:
      binding 0            -> uniform 'inf' guard (f32) tinygrad's WGSL expects
      bindings 1..nbufs    -> storage buffers (kernel args)
      bindings nbufs+1..   -> uniform i32/f32 scalars ('vals')
    """
    def __init__(self, dev:"SharedWebGpuDevice", name:str, lib:bytes, *args, **kwargs):
        # newer tinygrad passes extra metadata (runtimevars=, prg=, aux ...) to
        # the program constructor; the WGSL dispatch path doesn't need them, so
        # accept and ignore anything beyond dev/name/lib.
        self.dev, self.name, self.src = dev, name, lib.decode()
        d = dev.wdev
        self.module = d.create_shader_module(code=self.src)
        self._pipeline_cache: dict = {}

    def _uniform(self, val):
        d = self.dev.wdev
        b = d.create_buffer(size=4, usage=BUF_UNIFORM | BUF_COPY_DST)
        data = val.to_bytes(4, "little") if isinstance(val, int) else struct.pack("<f", val)
        d.queue.write_buffer(b, 0, data)
        return b

    def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
        d = self.dev.wdev
        nb = len(bufs)
        # bind group layout
        entries_l = [{"binding":0, "visibility":STAGE_COMPUTE,
                      "buffer":{"type":BIND_UNIFORM}}]
        for i in range(nb + len(vals)):
            entries_l.append({"binding":i+1, "visibility":STAGE_COMPUTE,
                              "buffer":{"type":(BIND_STORAGE if i < nb else BIND_UNIFORM)}})
        bgl = d.create_bind_group_layout(entries=entries_l)
        pl  = d.create_pipeline_layout(bind_group_layouts=[bgl])

        pipe = d.create_compute_pipeline(
            layout=pl, compute={"module":self.module, "entry_point":self.name})

        # bind group
        entries_b = [{"binding":0, "resource":{"buffer":self._uniform(float("inf")),
                                                "offset":0, "size":4}}]
        for i, b in enumerate(bufs):
            entries_b.append({"binding":i+1,
                              "resource":{"buffer":b, "offset":0, "size":b.size}})
        for j, v in enumerate(vals):
            ub = self._uniform(v)
            entries_b.append({"binding":nb+1+j,
                              "resource":{"buffer":ub, "offset":0, "size":4}})
        bg = d.create_bind_group(layout=bgl, entries=entries_b)

        enc = d.create_command_encoder()
        cpass = enc.begin_compute_pass()
        cpass.set_pipeline(pipe)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups(*global_size)
        cpass.end()
        d.queue.submit([enc.finish()])
        return None


class SharedWebGpuAllocator(Allocator):
    def _alloc(self, size:int, options:BufferSpec):
        return self.dev.wdev.create_buffer(
            size=round_up(size, 4),
            usage=BUF_STORAGE | BUF_COPY_DST | BUF_COPY_SRC)
    def _copyin(self, dest, src:memoryview):
        mv = src
        if src.nbytes % 4:
            pad = bytearray(round_up(src.nbytes, 4)); pad[:src.nbytes] = src; mv = memoryview(pad)
        self.dev.wdev.queue.write_buffer(dest, 0, mv)
    def _copyout(self, dest:memoryview, src):
        data = self.dev.read_buffer(src)            # bytes-like
        n = dest.nbytes
        dest[:] = data[:n]
    def _free(self, opaque, options:BufferSpec):
        try: opaque.destroy()
        except Exception: pass


class SharedWebGpuDevice(Compiled):
    """A tinygrad Compiled device bound to an externally-owned wgpu-py GPUDevice."""
    def __init__(self, wgpu_device, name:str="WEBGPU"):
        self.wdev = wgpu_device            # the SAME device pygfx uses
        super().__init__(name, SharedWebGpuAllocator(self), [WGSLRenderer],
                         functools.partial(SharedWebGPUProgram, self))

    def read_buffer(self, buf) -> bytes:
        """GPU->host readback via a temporary MAP_READ staging buffer."""
        d = self.wdev
        size = buf.size
        staging = d.create_buffer(size=size, usage=BUF_COPY_DST | BUF_MAP_READ)
        enc = d.create_command_encoder()
        enc.copy_buffer_to_buffer(buf, 0, staging, 0, size)
        d.queue.submit([enc.finish()])
        staging.map_sync(MAP_READ)
        data = bytes(staging.read_mapped())
        staging.unmap()
        try: staging.destroy()
        except Exception: pass
        return data

    def synchronize(self):
        # wgpu-native: device.poll / queue work-done. Best-effort across versions.
        for fn in ("_poll", "poll"):
            f = getattr(self.wdev, fn, None)
            if callable(f):
                try: f(); return
                except Exception: pass


def tinygrad_buffer_handle(t):
    """Return the underlying wgpu-py GPUBuffer object backing a realized Tensor.

    t may be a Tensor or a tinygrad Buffer. The opaque stored by our allocator
    IS the wgpu-py GPUBuffer (same device as pygfx), so pygfx can consume it
    directly in a command encoder.
    """
    from tinygrad import Tensor
    if isinstance(t, Tensor):
        t = t.realize()
        buf = t.uop.buffer
    else:
        buf = t
    buf.ensure_allocated()
    return buf._buf


def copy_tensor_to_texture(dev:"SharedWebGpuDevice", t, texture, width:int, height:int,
                           bytes_per_row:int|None=None):
    """On-GPU copy of a realized Tensor's buffer into a wgpu-py GPUTexture.

    No host round-trip: both buffer and texture are on dev.wdev. `texture` is a
    wgpu.GPUTexture pygfx owns (e.g. via gfx.Texture). bytes_per_row must be a
    multiple of 256 per WebGPU; pad your texture width or row accordingly.
    """
    t.realize()
    dev.synchronize()
    src = tinygrad_buffer_handle(t)
    bpr = bytes_per_row if bytes_per_row is not None else width * 4
    enc = dev.wdev.create_command_encoder()
    enc.copy_buffer_to_texture(
        {"buffer":src, "offset":0, "bytes_per_row":bpr, "rows_per_image":height},
        {"texture":texture, "mip_level":0, "origin":(0,0,0)},
        (width, height, 1))
    dev.wdev.queue.submit([enc.finish()])


def install_shared_webgpu(wgpu_device, name:str="WEBGPU", set_default:bool=True):
    """Make tinygrad's Device[name] resolve to a SharedWebGpuDevice on wgpu_device.

    Returns the SharedWebGpuDevice. After this, Tensor(..., device=name) and
    .to(name) run on the same wgpu-py device pygfx renders from.

    set_default=True also makes `name` the process default device (DEV.value),
    so scheduler-allocated INTERMEDIATE buffers — which don't inherit a device
    from their inputs — land on the shared device instead of falling back to
    CUDA. Without this, the optimizer's realize creates a scratch buffer on the
    default (CUDA) device and crashes trying to make a second GPU context.
    """
    from tinygrad.device import Device
    dev = SharedWebGpuDevice(wgpu_device, name)
    # Inject into the singleton's cache so the importlib/name lookup is bypassed.
    Device._Device__get_canonicalized_item.cache_clear()  # type: ignore[attr-defined]
    # Prime the functools.cache by calling, then overwrite via a wrapper.
    orig = Device._Device__get_canonicalized_item.__wrapped__  # type: ignore[attr-defined]
    def patched(ix, _orig=orig, _dev=dev, _name=name):
        return _dev if ix.split(":")[0] == _name else _orig(Device, ix)
    Device._Device__get_canonicalized_item = patched  # type: ignore[attr-defined]
    Device._opened_devices.add(name)
    if set_default:
        # DEV is a _DEV(ContextVar) holding a list[Target]; DEV.device (used by
        # Device.DEFAULT) reads _value[0].device. Setting DEV.value="WEBGPU"
        # parses it into a Target so DEFAULT becomes the shared device. The
        # injection above already ran, so any Device["WEBGPU"] resolves to `dev`.
        from tinygrad.helpers import DEV
        DEV.value = name
    return dev