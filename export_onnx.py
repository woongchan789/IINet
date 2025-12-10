"""Export IINet disparity model to ONNX."""

import argparse
import os
from typing import Optional, Tuple

import torch

import options
from modules.disp_model import DispModel


class OnnxExportWrapper(torch.nn.Module):
    """Wrap DispModel to expose ONNX-friendly outputs."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model({"left": left, "right": right})
        disp = outputs["disp_pred_s0"]
        coarse_disp = outputs.get("coarse_disp", disp)
        return disp, coarse_disp


def _load_options(config_file: Optional[str]) -> options.Options:
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options(config_filepaths=config_file, ignore_cl_args=True)
    return option_handler.options


def _load_model(opts: options.Options, checkpoint: str, device: torch.device) -> DispModel:
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = DispModel(opts).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def export_onnx(
    config_file: Optional[str],
    checkpoint: str,
    output_path: str,
    height: Optional[int],
    width: Optional[int],
    opset: int,
    use_cpu: bool,
) -> None:
    opts = _load_options(config_file)

    if opts.feature_volume_type != "ms_cost_volume":
        raise ValueError(
            "ONNX export expects 'ms_cost_volume' feature volume. "
            "Provide the training config via --config_file to ensure settings match."
        )

    if height is None:
        height = opts.val_height
    if width is None:
        width = opts.val_width

    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    model = _load_model(opts, checkpoint, device)
    wrapped = OnnxExportWrapper(model)

    dummy_left = torch.randn(1, 3, height, width, device=device)
    dummy_right = torch.randn(1, 3, height, width, device=device)

    dynamic_axes = {
        "left": {0: "batch", 2: "height", 3: "width"},
        "right": {0: "batch", 2: "height", 3: "width"},
        "disp_pred": {0: "batch", 2: "height", 3: "width"},
        "coarse_disp": {0: "batch", 2: "height", 3: "width"},
    }

    torch.onnx.export(
        wrapped,
        (dummy_left, dummy_right),
        output_path,
        input_names=["left", "right"],
        output_names=["disp_pred", "coarse_disp"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export IINet model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint")
    parser.add_argument("--output", default="iinet.onnx", help="Output ONNX filename")
    parser.add_argument("--config_file", default=None, help="Optional YAML config used for training")
    parser.add_argument("--height", type=int, default=None, help="Export height (defaults to validation height)")
    parser.add_argument("--width", type=int, default=None, help="Export width (defaults to validation width)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--cpu", action="store_true", help="Force export on CPU even if CUDA is available")

    args = parser.parse_args()

    export_onnx(
        config_file=args.config_file,
        checkpoint=args.checkpoint,
        output_path=args.output,
        height=args.height,
        width=args.width,
        opset=args.opset,
        use_cpu=args.cpu,
    )
