"""
scripts/check_install.py
=========================
Vérifie que toutes les dépendances critiques de GARF sont installées et fonctionnelles.

Usage :
    python scripts/check_install.py

Chaque test affiche OK ou FAIL avec un message explicatif.
À la fin : résumé du nombre de tests passés.
"""

import sys

PASS = "  [OK]"
FAIL = "  [FAIL]"
results = []


def check(name, fn):
    try:
        msg = fn()
        print(f"{PASS} {name}" + (f" — {msg}" if msg else ""))
        results.append((name, True))
    except Exception as e:
        print(f"{FAIL} {name} — {e}")
        results.append((name, False))


print("\n========== GARF — Vérification de l'installation ==========\n")

# ── 1. Python ────────────────────────────────────────────────────────────────
print("[ Python & système ]")

check("Python version", lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def check_platform():
    import platform
    p = platform.system()
    if p != "Linux":
        raise RuntimeError(f"Linux requis, détecté : {p}. spconv/flash-attn ne fonctionnent que sur Linux.")
    return p
check("Système Linux", check_platform)

# ── 2. PyTorch + CUDA ────────────────────────────────────────────────────────
print("\n[ PyTorch & CUDA ]")

def check_torch():
    import torch
    return f"torch {torch.__version__}"
check("torch importable", check_torch)

def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() = False. GPU non détecté ou drivers manquants.")
    return f"CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}"
check("CUDA disponible", check_cuda)

def check_cuda_version():
    import torch
    ver = torch.version.cuda
    major, minor = int(ver.split(".")[0]), int(ver.split(".")[1])
    if major < 12 or (major == 12 and minor < 6):
        raise RuntimeError(f"CUDA {ver} détecté. spconv-cu126 requiert CUDA >= 12.6")
    return f"CUDA {ver} >= 12.6 ✓"
check("Version CUDA >= 12.6", check_cuda_version)

def check_gpu_memory():
    import torch
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if mem < 8:
        raise RuntimeError(f"Seulement {mem:.1f} GB VRAM. GARF recommande >= 16 GB pour l'entraînement.")
    return f"{mem:.1f} GB VRAM"
check("Mémoire GPU", check_gpu_memory)

# ── 3. Dépendances principales ────────────────────────────────────────────────
print("\n[ Dépendances principales ]")

check("lightning",   lambda: __import__("lightning").__version__)
check("hydra-core",  lambda: __import__("hydra").__version__)
check("diffusers",   lambda: __import__("diffusers").__version__)
check("peft",        lambda: __import__("peft").__version__)
check("trimesh",     lambda: __import__("trimesh").__version__)
check("h5py",        lambda: __import__("h5py").__version__)
check("scipy",       lambda: __import__("scipy").__version__)
check("numpy",       lambda: __import__("numpy").__version__)
check("sklearn",     lambda: __import__("sklearn").__version__)

# ── 4. Dépendances "post" (les plus sensibles) ───────────────────────────────
print("\n[ Dépendances post (critiques) ]")

def check_pytorch3d():
    import pytorch3d
    # Test rapide d'une opération SO(3)
    import torch
    import pytorch3d.transforms as t
    q = torch.tensor([[1., 0., 0., 0.]])
    m = t.quaternion_to_matrix(q)
    assert m.shape == (1, 3, 3)
    return pytorch3d.__version__
check("pytorch3d (+ quaternion_to_matrix)", check_pytorch3d)

def check_spconv():
    import spconv.pytorch as spconv
    return f"spconv OK"
check("spconv", check_spconv)

def check_torch_scatter():
    import torch_scatter
    import torch
    src = torch.tensor([1., 2., 3., 4.])
    idx = torch.tensor([0, 0, 1, 1])
    out = torch_scatter.scatter(src, idx, reduce="sum")
    assert out.tolist() == [3., 7.], f"résultat inattendu : {out}"
    return "scatter OK"
check("torch_scatter (+ test scatter)", check_torch_scatter)

def check_flash_attn():
    import flash_attn
    return flash_attn.__version__
check("flash-attn", check_flash_attn)

# ── 5. Modules GARF ───────────────────────────────────────────────────────────
print("\n[ Modules GARF ]")

def check_ptv3():
    from assembly.backbones.pointtransformerv3 import PointTransformerV3
    return "PointTransformerV3 importable"
check("PointTransformerV3", check_ptv3)

def check_frac_seg():
    from assembly.models.pretraining.frac_seg import FracSeg
    return "FracSeg importable"
check("FracSeg", check_frac_seg)

def check_scheduler():
    from assembly.models.denoiser.modules.scheduler import SE3FlowMatchEulerDiscreteScheduler
    s = SE3FlowMatchEulerDiscreteScheduler()
    assert len(s.sigmas) == 1000
    return f"scheduler OK (1000 timesteps, σ_max={s.sigma_max:.3f})"
check("SE3FlowMatchEulerDiscreteScheduler", check_scheduler)

def check_denoiser():
    from assembly.models.denoiser.denoiser_flow_matching import DenoiserFlowMatching
    return "DenoiserFlowMatching importable"
check("DenoiserFlowMatching", check_denoiser)

def check_dataset():
    from assembly.data.breaking_bad.uniform import BreakingBadUniform
    return "BreakingBadUniform importable"
check("BreakingBadUniform", check_dataset)

def check_transforms():
    from assembly.data.transform import recenter_pc, rotate_pc, shuffle_pc
    import numpy as np
    pc = np.random.randn(100, 3)
    pc_c, centroid = recenter_pc(pc)
    assert abs(pc_c.mean(0)).max() < 1e-6
    pc_r, _, q = rotate_pc(pc_c)
    assert q.shape == (4,)
    return "recenter/rotate/shuffle OK"
check("transforms (recenter/rotate/shuffle)", check_transforms)

# ── 6. Résumé ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 57)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"  Résultat : {passed}/{total} tests passés")

failed = [name for name, ok in results if not ok]
if failed:
    print(f"\n  Tests échoués :")
    for name in failed:
        print(f"    - {name}")
    print()
    sys.exit(1)
else:
    print("\n  Tout est OK — environnement prêt pour GARF.\n")
    sys.exit(0)
