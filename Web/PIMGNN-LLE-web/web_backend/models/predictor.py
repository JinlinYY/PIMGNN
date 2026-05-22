# web_backend/models/predictor.py
import os
import json
import sys
import importlib.util
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch

import config as C


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"Failed to load module: {file_path}")
    loader.exec_module(module)
    return module


def _load_project_modules():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    config_path = os.path.join(src_dir, "config.py")
    utils_path = os.path.join(src_dir, "utils.py")
    data_path = os.path.join(src_dir, "data.py")
    model_path = os.path.join(src_dir, "model.py")

    project_config = _load_module("project_config", config_path)
    project_utils = _load_module("project_utils", utils_path)

    prev_config = sys.modules.get("config")
    prev_utils = sys.modules.get("utils")
    prev_data = sys.modules.get("data")

    try:
        sys.modules["config"] = project_config
        sys.modules["utils"] = project_utils
        project_data = _load_module("project_data", data_path)
        sys.modules["data"] = project_data
        project_model = _load_module("project_model", model_path)
    finally:
        if prev_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = prev_config
        if prev_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = prev_utils
        if prev_data is None:
            sys.modules.pop("data", None)
        else:
            sys.modules["data"] = prev_data

    return project_config, project_utils, project_data, project_model

class ModelPredictor:
    def __init__(self):
        self.device = torch.device(getattr(C, "DEVICE", "cpu"))
        self.model = None
        self.scaler = None
        self.project_config, self.project_utils, self.project_data, self.project_model = _load_project_modules()
        self.use_graph = bool(getattr(self.project_config, "USE_GRAPH", False))
        self.use_mix_graph = bool(getattr(self.project_config, "USE_MIX_GRAPH", False))
        self.use_fg = bool(getattr(self.project_config, "USE_FG", False))
        self.fg_token_mode = bool(getattr(self.project_config, "FG_TOKEN_MODE", False))
        self.fg_max_tokens = int(getattr(self.project_config, "FG_MAX_TOKENS", 32))
        self.fg_topk = int(getattr(self.project_config, "FG_TOPK", 0))
        self.fg_min_freq = int(getattr(self.project_config, "FG_MIN_FREQ", 1))
        self.g_cache = None
        self.mix_cache = None
        self.fg_cache = None
        self._init_caches()
        self.load_model()

    def _init_caches(self) -> None:
        if self.use_graph:
            self.g_cache = self.project_data.GraphCache(
                add_hs=getattr(self.project_config, "GRAPH_ADD_HS", False),
                add_3d=getattr(self.project_config, "GRAPH_ADD_3D", False),
                use_gasteiger=getattr(self.project_config, "GRAPH_USE_GASTEIGER", True),
                max_atoms=getattr(self.project_config, "GRAPH_MAX_ATOMS", 256),
            )
        if self.use_mix_graph:
            self.mix_cache = self.project_data.MixGraphCache(self.project_config)
        if self.use_fg:
            fg_path = os.path.join(getattr(C, "MODEL_DIR", os.path.dirname(getattr(C, "MODEL_PATH"))), "fg_corpus.json")
            if not os.path.isfile(fg_path):
                raise FileNotFoundError(f"FG corpus not found: {fg_path}")
            with open(fg_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            self.fg_cache = self.project_data.FunctionalGroupCache(
                corpus=corpus,
                vocab_size=self.fg_topk,
                min_freq=self.fg_min_freq,
            )
            self.fg_cache.set_corpus(corpus)
    
    def load_model(self):
        """加载预训练模型"""
        model_path = getattr(C, "MODEL_PATH", None)
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        use_graph = self.use_graph
        if use_graph:
            self.model = self.project_model.LLEGraphNet(
                gnn_hidden=getattr(self.project_config, "GNN_HIDDEN", 256),
                gnn_layers=getattr(self.project_config, "GNN_LAYERS", 4),
                mlp_hidden=getattr(self.project_config, "GNN_HEAD_HIDDEN", 512),
                dropout=getattr(self.project_config, "DROPOUT", 0.15),
                pool=getattr(self.project_config, "GNN_POOL", "mean"),
                use_interaction=getattr(self.project_config, "GNN_INTERACTION", True),
                use_mix_graph=getattr(self.project_config, "USE_MIX_GRAPH", False),
                mix_layers=getattr(self.project_config, "MIX_LAYERS", 2),
                mix_hidden=getattr(self.project_config, "MIX_HIDDEN", getattr(self.project_config, "GNN_HIDDEN", 256)),
                mix_dropout=getattr(self.project_config, "MIX_DROPOUT", 0.10),
                use_fg=getattr(self.project_config, "USE_FG", False),
                fg_vocab_size=int(getattr(self.project_config, "FG_TOPK", 0)),
                fg_hidden=int(getattr(self.project_config, "FG_MLP_HIDDEN", 256)),
                fg_dropout=float(getattr(self.project_config, "FG_DROPOUT", 0.10)),
                fg_token_mode=bool(getattr(self.project_config, "FG_TOKEN_MODE", False)),
                fg_max_tokens=int(getattr(self.project_config, "FG_MAX_TOKENS", 32)),
                fg_cross_attn=bool(getattr(self.project_config, "FG_CROSS_ATTN", False)),
                fg_attn_heads=int(getattr(self.project_config, "FG_ATTN_HEADS", 8)),
                s3_equivariant=bool(getattr(self.project_config, "S3_EQUIVARIANT", False)),
                fusion_mode=getattr(self.project_config, "FUSION_MODE", "concat"),
                tf_dim=int(getattr(self.project_config, "TF_DIM", getattr(self.project_config, "GNN_HIDDEN", 256))),
                tf_layers=int(getattr(self.project_config, "TF_LAYERS", 2)),
                tf_heads=int(getattr(self.project_config, "TF_HEADS", 8)),
                tf_ff=int(getattr(self.project_config, "TF_FF", 1024)),
                tf_dropout=float(getattr(self.project_config, "TF_DROPOUT", 0.10)),
                tf_pool=str(getattr(self.project_config, "TF_POOL", "cls")),
                tf_max_len=int(getattr(self.project_config, "TF_MAX_LEN", 32)),
                tf_type_vocab=int(getattr(self.project_config, "TF_TYPE_VOCAB", 16)),
            )
        else:
            in_dim = 3 * int(getattr(self.project_config, "FP_BITS", 2048)) + 2
            if getattr(self.project_config, "USE_FG", False):
                in_dim += 3 * int(getattr(self.project_config, "FG_TOPK", 0))
            self.model = self.project_model.LLECurveNet(
                in_dim=in_dim,
                hidden=getattr(self.project_config, "HIDDEN", 1024),
                dropout=getattr(self.project_config, "DROPOUT", 0.15),
            )

        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict, strict=True)

        if isinstance(ckpt, dict) and ("T_mean" in ckpt and "T_std" in ckpt):
            self.scaler = self.project_utils.Scaler(mean=float(ckpt["T_mean"]), std=float(ckpt["T_std"]))
        else:
            fallback_path = os.path.join(os.path.dirname(model_path), "last_model.pt")
            if os.path.isfile(fallback_path):
                fallback = torch.load(fallback_path, map_location="cpu")
                if isinstance(fallback, dict) and ("T_mean" in fallback and "T_std" in fallback):
                    self.scaler = self.project_utils.Scaler(
                        mean=float(fallback["T_mean"]),
                        std=float(fallback["T_std"]),
                    )
                else:
                    raise KeyError("T_mean/T_std not found in last_model.pt")
            else:
                raise FileNotFoundError("T_mean/T_std missing and last_model.pt not found for scaler")

        self.model.to(self.device)
        self.model.eval()
    
    def _build_graph_input(self, smiles_list: List[str], temperature: float, t: float) -> Dict[str, Any]:
        s1 = self.project_utils.canonicalize_smiles(smiles_list[0])
        s2 = self.project_utils.canonicalize_smiles(smiles_list[1])
        s3 = self.project_utils.canonicalize_smiles(smiles_list[2])
        if not (s1 and s2 and s3):
            raise ValueError("Invalid SMILES.")

        if self.g_cache is None:
            raise RuntimeError("Graph cache not initialized.")
        self.g_cache.build_from_smiles([s1, s2, s3])
        g1 = self.g_cache.get(s1)
        g2 = self.g_cache.get(s2)
        g3 = self.g_cache.get(s3)

        bg1 = self.project_utils.batch_graphs([g1])
        bg2 = self.project_utils.batch_graphs([g2])
        bg3 = self.project_utils.batch_graphs([g3])

        Tn = self.scaler.transform(np.array([temperature], dtype=np.float32))[0].astype(np.float32)
        scalars = torch.tensor([[Tn, float(t)]], dtype=torch.float32)

        x: Dict[str, Any] = {"g1": bg1, "g2": bg2, "g3": bg3, "scalars": scalars}

        if self.use_fg and self.fg_cache is not None:
            if self.fg_token_mode:
                ids1, m1 = self.fg_cache.get_token_ids(s1, self.fg_max_tokens)
                ids2, m2 = self.fg_cache.get_token_ids(s2, self.fg_max_tokens)
                ids3, m3 = self.fg_cache.get_token_ids(s3, self.fg_max_tokens)
                x["fg1_ids"] = torch.tensor([ids1], dtype=torch.long)
                x["fg2_ids"] = torch.tensor([ids2], dtype=torch.long)
                x["fg3_ids"] = torch.tensor([ids3], dtype=torch.long)
                x["fg1_mask"] = torch.tensor([m1], dtype=torch.float32)
                x["fg2_mask"] = torch.tensor([m2], dtype=torch.float32)
                x["fg3_mask"] = torch.tensor([m3], dtype=torch.float32)
            else:
                x["fg1"] = torch.tensor([self.fg_cache.get(s1)], dtype=torch.float32)
                x["fg2"] = torch.tensor([self.fg_cache.get(s2)], dtype=torch.float32)
                x["fg3"] = torch.tensor([self.fg_cache.get(s3)], dtype=torch.float32)

        if self.use_mix_graph and (self.mix_cache is not None):
            mix = self.mix_cache.build(s1, s2, s3, float(Tn), float(temperature))
            x["mix"] = self.project_utils.batch_mixture_graphs([mix])

        return x

    def _build_fp_input(self, smiles_list: List[str], temperature: float, t: float) -> torch.Tensor:
        s1 = self.project_utils.canonicalize_smiles(smiles_list[0])
        s2 = self.project_utils.canonicalize_smiles(smiles_list[1])
        s3 = self.project_utils.canonicalize_smiles(smiles_list[2])
        if not (s1 and s2 and s3):
            raise ValueError("Invalid SMILES.")

        fp_bits = int(getattr(self.project_config, "FP_BITS", 2048))
        fp_radius = int(getattr(self.project_config, "FP_RADIUS", 2))
        fp1 = self.project_utils.morgan_fp(s1, radius=fp_radius, n_bits=fp_bits)
        fp2 = self.project_utils.morgan_fp(s2, radius=fp_radius, n_bits=fp_bits)
        fp3 = self.project_utils.morgan_fp(s3, radius=fp_radius, n_bits=fp_bits)

        parts = [fp1, fp2, fp3]
        if self.use_fg and self.fg_cache is not None:
            parts.extend([
                self.fg_cache.get(s1),
                self.fg_cache.get(s2),
                self.fg_cache.get(s3),
            ])

        Tn = self.scaler.transform(np.array([temperature], dtype=np.float32))[0].astype(np.float32)
        parts.append(np.array([Tn, float(t)], dtype=np.float32))
        x = np.concatenate(parts, axis=0).astype(np.float32)
        return torch.from_numpy(x)

    def predict(self, input_data: Dict[str, Any]) -> Tuple[List[float], List[float]]:
        """执行预测（输入已预处理）"""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not initialized.")
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = self.device
        with torch.no_grad():
            if self.use_graph:
                x = self.project_utils.batch_to_device(input_data, model_device)
                pred = self.model(x).detach().cpu().numpy()[0]
            else:
                x = input_data.to(model_device).unsqueeze(0)
                pred = self.model(x).detach().cpu().numpy()[0]
        e_pred = pred[:3].tolist()
        r_pred = pred[3:].tolist()
        return e_pred, r_pred

    def predict_from_smiles(self, smiles_list: List[str], temperature: float, t: float = 0.5) -> Tuple[List[float], List[float]]:
        """从SMILES和温度直接预测"""
        if self.use_graph:
            x = self._build_graph_input(smiles_list, temperature, t)
        else:
            x = self._build_fp_input(smiles_list, temperature, t)
        return self.predict(x)

# 全局模型实例
predictor = ModelPredictor()