#!/usr/bin/env python3
"""nv-: a local web dashboard for NVIDIA GPUs (MIT License).

Serves one page (index.html) and one JSON endpoint (/api/stats).
Needs: psutil, nvidia-ml-py (imported as pynvml). Optional: lspci for device names.
"""
import argparse
import json
import math
import os
import random
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3
# Throttle (clock event) reason bits from NVML
REASONS = [(0x4 | 0x80, "power limit"), (0x20 | 0x40, "temperature"), (0x8, "hardware slowdown"),
           (0x2, "clock setting"), (0x1, "idle")]
NVLINK_ERRORS = [("replay", 0), ("recovery", 1), ("crc_flit", 2), ("crc_data", 3)]

try:
    import pynvml
    pynvml.nvmlInit()
    NVML = True
except Exception:
    NVML = False


def _str(v):
    return v.decode() if isinstance(v, bytes) else v


def norm_bdf(bus_id):
    dom, bus, devfn = _str(bus_id).split(":")
    return f"{int(dom, 16):04x}:{bus.lower()}:{devfn.lower()}"


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                m = line.split(":", 1)[1]
                for junk in ("(R)", "(TM)", " CPU", "  "):
                    m = m.replace(junk, " " if junk == "  " else "")
                return m.split("@")[0].strip()
    except OSError:
        pass
    return None


def system_stats():
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    temp = None
    try:
        for chip, entries in psutil.sensors_temperatures().items():
            if chip in ("coretemp", "k10temp"):
                temp = max((e.current for e in entries if e.label.startswith(("Package", "Tctl", "Tdie"))), default=temp)
    except Exception:
        pass
    freq = _try(lambda: psutil.cpu_freq().current)
    cache = (getattr(vm, "cached", 0) + getattr(vm, "buffers", 0)) / GIB
    return {
        "cpu": {"threads": psutil.cpu_percent(percpu=True), "total": psutil.cpu_percent(),
                "temp": temp, "load": os.getloadavg(), "freq": freq,
                "cores": psutil.cpu_count(logical=False), "model": cpu_model()},
        "ram": {"total": vm.total / GIB, "used": (vm.total - vm.available) / GIB, "cache": cache,
                "free": vm.free / GIB, "swap_used": sw.used / GIB, "swap_total": sw.total / GIB},
    }


# ---------- DCGM (core activity and NVLink traffic) ----------

DCGM_FIELDS = [(1002, "sm"), (1004, "tensor"), (1005, "dram"), (1006, "fp64"), (1007, "fp32"), (1008, "fp16"),
               (1011, "nvl_tx"), (1012, "nvl_rx")]


class DcgmReader:
    """Runs `dcgmi dmon` in the background and keeps the last 3 readings of each field per GPU.

    DCGM takes turns between fields it cannot read at the same time and reports the skipped ones as 0,
    so the page shows the highest of the last 3 readings.
    """
    def __init__(self, interval):
        self.interval = interval
        self.hist = {}            # (gpu, name) -> list of last 3 values
        self.lock = threading.Lock()
        self.ok = shutil.which("dcgmi") is not None
        if self.ok:
            threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        cmd = ["dcgmi", "dmon", "-e", ",".join(str(f) for f, _ in DCGM_FIELDS), "-d", str(int(self.interval * 1000))]
        while True:
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                for line in p.stdout:
                    parts = line.split()
                    if len(parts) < 2 + len(DCGM_FIELDS) or parts[0] != "GPU":
                        continue
                    gpu = int(parts[1])
                    with self.lock:
                        for (_, name), raw in zip(DCGM_FIELDS, parts[2:]):
                            try:
                                v = float(raw)
                            except ValueError:
                                continue
                            h = self.hist.setdefault((gpu, name), [])
                            h.append(v)
                            del h[:-3]
            except Exception:
                pass
            time.sleep(5)          # dcgmi stopped: try again

    def get(self, gpu):
        with self.lock:
            out = {name: max(self.hist[(gpu, name)]) for _, name in DCGM_FIELDS if self.hist.get((gpu, name))}
        if not out:
            return None
        for k in ("sm", "tensor", "dram", "fp64", "fp32", "fp16"):
            if k in out:
                out[k] *= 100                          # fraction -> percent
        for k in ("nvl_tx", "nvl_rx"):
            if k in out:
                out[k] /= 1e9                          # bytes per second -> GB/s
        return out


# ---------- GPUs ----------

def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


class Sampler:
    def __init__(self, args):
        self.args = args
        self.prev = {}          # (gpu, link, dir) -> (counter KiB, time)
        self.data = {}
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.dcgm = None if args.demo else DcgmReader(args.interval)
        self.sims = parse_simulations(args.simulate)
        # --slots: the PCI bus number of each expected GPU, in screen order; a missing GPU shows as a grayed card.
        self.slots = [x.strip().lower().split(":")[-2] if ":" in x else x.strip().lower().zfill(2)
                      for x in args.slots.split(",") if x.strip()] if args.slots else []

    def nvlinks(self, h, gi, bus_to_idx, now):
        tx_id = getattr(pynvml, "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX", None)
        rx_id = getattr(pynvml, "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX", None)
        links = []
        for l in range(getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 18)):
            try:
                state = pynvml.nvmlDeviceGetNvLinkState(h, l)
            except Exception:
                continue
            lk = {"link": l, "up": bool(state), "peer": None, "tx": None, "rx": None}
            try:
                lk["peer"] = bus_to_idx.get(norm_bdf(pynvml.nvmlDeviceGetNvLinkRemotePciInfo(h, l).busId))
            except Exception:
                pass
            lk["errors"] = {name: _try(lambda c=cid: pynvml.nvmlDeviceGetNvLinkErrorCounter(h, l, c), 0)
                            for name, cid in NVLINK_ERRORS}
            if tx_id is not None and lk["up"]:
                try:
                    fv = pynvml.nvmlDeviceGetFieldValues(h, [(tx_id, l), (rx_id, l)])
                    for key, v in (("tx", fv[0]), ("rx", fv[1])):
                        if v.nvmlReturn == 0:
                            kib = v.value.ullVal
                            old = self.prev.get((gi, l, key))
                            self.prev[(gi, l, key)] = (kib, now)
                            if old and now > old[1]:
                                lk[key] = (kib - old[0]) * 1024 / (now - old[1]) / 1e9   # GB/s
                except Exception:
                    pass
            links.append(lk)
        return links

    def gpus(self):
        now = time.time()
        n = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        bdfs = [_try(lambda h=h: norm_bdf(pynvml.nvmlDeviceGetPciInfo(h).busId)) for h in handles]
        bus_to_idx = {b: i for i, b in enumerate(bdfs) if b}
        reason_fn = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        out = []
        for i, h in enumerate(handles):
            mem = _try(lambda: pynvml.nvmlDeviceGetMemoryInfo(h))
            bits = _try(lambda: reason_fn(h), 0) if reason_fn else 0
            g = {
                "index": i, "bdf": bdfs[i],
                "name": _try(lambda: _str(pynvml.nvmlDeviceGetName(h))),
                "util": _try(lambda: pynvml.nvmlDeviceGetUtilizationRates(h).gpu),
                "mem_used": mem.used / GIB if mem else None,
                "mem_total": mem.total / GIB if mem else None,
                "temp": _try(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
                "power": _try(lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000),
                "power_limit": _try(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000),
                "sm_clock": _try(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)),
                "sm_clock_max": _try(lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)),
                "held_back": [label for mask, label in REASONS if bits & mask],
                "arch": _try(lambda: pynvml.nvmlDeviceGetArchitecture(h)),
                "cores": _try(lambda: pynvml.nvmlDeviceGetNumGpuCores(h)),
                "pcie_gen": _try(lambda: pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h)),
                "pcie_width": _try(lambda: pynvml.nvmlDeviceGetCurrPcieLinkWidth(h)),
                "pcie_max_gen": _try(lambda: pynvml.nvmlDeviceGetMaxPcieLinkGeneration(h)),
                "pcie_tx": _try(lambda: pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1e6),
                "pcie_rx": _try(lambda: pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1e6),
                "nvlinks": self.nvlinks(h, i, bus_to_idx, now),
                "dcgm": self.dcgm.get(i) if self.dcgm else None,
            }
            out.append(g)
        return out

    def sample(self):
        a = self.args
        if a.demo:
            d = demo(time.time() - self.t0, a.demo_layout)
        else:
            d = {"gpus": self.gpus() if NVML else [],
                 "uptime": time.time() - psutil.boot_time()}
            d.update(system_stats())
        apply_simulations(d, self.sims)
        d.update(host="demo" if a.demo else socket.gethostname(), time=time.time(), interval=a.interval, layout=a.layout,
                 slots=self.slots)
        with self.lock:
            self.data = d

    def loop(self):
        psutil.cpu_percent(percpu=True)
        while True:
            try:
                self.sample()
            except Exception as e:
                with self.lock:
                    self.data = {"error": str(e), "time": time.time()}
            time.sleep(self.args.interval)

    def json(self):
        with self.lock:
            return json.dumps(self.data).encode()


# ---------- demo data ----------

DEMO_MODELS = {
    # name, memory GiB, power limit W, top SM clock MHz, DCGM detail available, NVML architecture, CUDA cores
    "v100": ("Tesla V100-SXM2-16GB", 15.77, 300, 1530, True, 5, 5120),
    "a100": ("NVIDIA A100-PCIE-40GB", 39.5, 250, 1410, True, 7, 6912),
    "3090": ("NVIDIA GeForce RTX 3090", 24.0, 350, 2100, False, 7, 10496),
    "a5000": ("NVIDIA RTX A5000", 24.0, 230, 2100, False, 7, 8192),
}
# layout name -> (GPU models in order, NVLink pairs as (gpu, gpu, links), whether links report traffic)
DEMO_LAYOUTS = {
    "4nvlink": (["v100"] * 4, [(a, b, 2) for a in range(4) for b in range(a + 1, 4)], False),
    "2bridge": (["3090"] * 2, [(0, 1, 4)], True),
    "2pairs": (["a5000"] * 4, [(0, 1, 4), (2, 3, 4)], True),
    "1gpu": (["a100"], [], False),
    "nolink": (["a100"] * 4, [], False),
    "6gpu": (["v100"] * 4 + ["a100"] * 2, [(a, b, 2) for a in range(4) for b in range(a + 1, 4)], False),
}


def demo(t, layout="4nvlink"):
    models, pairs, rated = DEMO_LAYOUTS.get(layout, DEMO_LAYOUTS["4nvlink"])
    gpus = []
    for i, m in enumerate(models):
        name, mem, plimit, clk, has_dcgm, arch, cores = DEMO_MODELS[m]
        u = max(0, min(100, 72 + 30 * math.sin(t / 9 + i) + random.uniform(-5, 5)))
        temp = 42 + u / 4 + (14 if i == 2 else 0)
        power = plimit * (0.25 + u / 100 * 0.7)
        links = []
        for a, b, n in pairs:
            if i not in (a, b):
                continue
            peer = b if i == a else a
            for k in range(n):
                rate = max(0.0, 6 * math.sin(t / 7 + 2 * i + peer + k) + 5) if rated else None
                links.append({"link": len(links), "up": True, "peer": peer, "tx": rate,
                              "rx": None if rate is None else rate * 0.8,
                              "errors": {"replay": 0, "recovery": 0, "crc_flit": 0, "crc_data": 0}})
        g = {
            "index": i, "bdf": f"0000:{0x1a + i:02x}:00.0", "name": name, "util": round(u),
            "mem_used": mem * (0.55 + 0.3 * math.sin(t / 30 + i)), "mem_total": mem,
            "temp": round(temp), "power": power, "power_limit": plimit,
            "sm_clock": round(clk * (0.7 + u / 400)), "sm_clock_max": clk,
            "held_back": [], "arch": arch, "cores": cores,
            "pcie_gen": 3 if m == "v100" else 4, "pcie_width": 8 if m == "v100" else 16, "pcie_max_gen": 3 if m == "v100" else 4,
            "pcie_tx": max(0, 2.5 + 2.5 * math.sin(t / 5 + i)), "pcie_rx": max(0, 3 + 3 * math.sin(t / 4 + i)),
            "nvlinks": links,
            "dcgm": None,
        }
        if has_dcgm:
            nvl = max(0, 40 + 30 * math.sin(t / 6 + i)) if links else 0.0
            g["dcgm"] = {"sm": min(100, u + 5), "tensor": 48.0 if i == 2 else 0.0, "dram": 30 + u / 3, "fp64": 0.0,
                         "fp32": 12.0 if i == 2 else u * 0.3, "fp16": 35.0 if i == 2 else 0.0,
                         "nvl_tx": nvl, "nvl_rx": nvl * 0.85}
        gpus.append(g)
    threads = [max(0, min(100, 30 + 25 * math.sin(t / 5 + c / 4) + random.uniform(-12, 12) + (40 if c % 9 == 0 else 0)))
               for c in range(72)]
    return {"gpus": gpus,
            "cpu": {"threads": threads, "total": sum(threads) / 72, "temp": 54, "load": (14.2, 13.8, 12.9),
                    "freq": 2400 + 300 * math.sin(t / 8), "cores": 36, "model": "Intel Xeon Platinum 8351N"},
            "ram": {"total": 125.5, "used": 23.8 + 2 * math.sin(t / 20), "cache": 83.2, "free": 18.5,
                    "swap_used": 0.0, "swap_total": 8.0},
            "uptime": 93784 + t}


# ---------- warning tests ----------

SIM_KEYS = ("link-down", "temp", "mem", "power", "held", "nvlink-errors", "drop")


def parse_simulations(items):
    """--simulate KEY=GPU:VALUE, e.g. temp=0:92, link-down=0:3, held=1:power, drop=2."""
    out = []
    for item in items or []:
        key, _, rest = item.partition("=")
        if key not in SIM_KEYS:
            raise SystemExit(f"--simulate: unknown test '{key}'; choose from {', '.join(SIM_KEYS)}")
        gpu, _, value = rest.partition(":")
        out.append((key, int(gpu), value))
    return out


def apply_simulations(d, sims):
    """Change the readings sent to the page. Nothing on the hardware is touched."""
    gpus = d.get("gpus", [])
    for key, gi, value in sims:
        g = next((x for x in gpus if x["index"] == gi), None)
        if g is None:
            continue
        if key == "temp":
            g["temp"] = float(value)
        elif key == "mem":
            g["mem_used"] = (g.get("mem_total") or 0) * float(value) / 100
        elif key == "power":
            g["power"] = float(value)
        elif key == "held":
            g["held_back"] = [{"power": "power limit", "heat": "temperature", "temperature": "temperature"}.get(value, value)]
        elif key == "link-down":
            for l in g.get("nvlinks", []):
                if l["link"] == int(value):
                    l["up"] = False
                    for other in gpus:                       # the same link, seen from the other GPU
                        if other["index"] == l.get("peer"):
                            for ol in other.get("nvlinks", []):
                                if ol.get("peer") == gi and ol["up"]:
                                    ol["up"] = False
                                    break
        elif key == "nvlink-errors":
            if g.get("nvlinks"):
                g["nvlinks"][0]["errors"]["replay"] += int(value or 1)
        elif key == "drop":
            gpus.remove(g)
    return d


# ---------- web ----------

def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/stats":
                self._send(200, state.json(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):
            pass

    return Handler


def main():
    p = argparse.ArgumentParser(description="nv- GPU dashboard")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--interval", type=float, default=2.0, help="seconds between readings")
    p.add_argument("--layout", default="", help="GPU numbers in screen order, left to right, top to bottom, e.g. 0,1,2,3")
    p.add_argument("--slots", default="", help="PCI bus number of each expected GPU in screen order, e.g. 1a,1b,1c,1d; "
                                                "a GPU that is missing shows as a grayed card")
    p.add_argument("--simulate", action="append", metavar="TEST=GPU:VALUE",
                   help="show a warning without touching the hardware: link-down=0:3, temp=0:92, mem=1:97, "
                        "power=2:320, held=3:power (or heat), nvlink-errors=1:5, drop=2 (GPU not detected); repeatable")
    p.add_argument("--demo", action="store_true", help="serve made-up numbers (no GPUs needed)")
    p.add_argument("--demo-layout", default="4nvlink", choices=sorted(DEMO_LAYOUTS),
                   help="demo machine: 4nvlink (default), 2bridge, 2pairs, 1gpu, nolink, 6gpu")
    args = p.parse_args()

    state = Sampler(args)
    state.sample()
    threading.Thread(target=state.loop, daemon=True).start()
    ThreadingHTTPServer((args.host, args.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()
