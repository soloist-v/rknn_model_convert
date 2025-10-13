import argparse
import os
import yaml
from pathlib import Path
from loguru import logger
from rknn.api import RKNN

# Default values
DEFAULT_DATASET_PATH = 'datasets/COCO/coco_subset_20.txt'
DEFAULT_RKNN_PATH = 'yolo11.rknn'
DEFAULT_MEAN_VALUES = [0, 0, 0]
DEFAULT_STD_VALUES = [255, 255, 255]

# Platform choices
PLATFORMS = ['rk3562', 'rk3566', 'rk3568', 'rk3576', 'rk3588', 'rv1126b', 'rv1109', 'rv1126', 'rk1808']

# Data type choices
DTYPE_CHOICES = ['i8', 'u8', 'fp']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert ONNX model to RKNN format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic conversion
  python convert.py model.onnx rk3588
  
  # Specify output and data type
  python convert.py model.onnx rk3588 --dtype i8 --output model.rknn
  
  # Floating point model (no quantization)
  python convert.py model.onnx rk3588 --dtype fp --no-quant
  
  # Custom dataset
  python convert.py model.onnx rk3588 --dataset datasets/imagenet/ILSVRC2012_img_val_samples/dataset_20.txt
  
  # Custom hybrid quantization (for precision-sensitive models)
  python convert.py yolov8_pose.onnx rk3588 --custom-hybrid custom_hybrid.yaml
  
  # Auto hybrid quantization (for older platforms)
  python convert.py model.onnx rv1109 --auto-hybrid-quant

Notes:
  - For rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b: use 'i8' or 'fp'
  - For rv1109, rv1126, rk1808: use 'u8' or 'fp'
  - Quantization is enabled by default when dtype is 'i8' or 'u8'
  - Hybrid quantization keeps critical layers in float precision for better accuracy
        '''
    )
    
    # Required arguments
    parser.add_argument(
        'model',
        type=str,
        help='Path to the input ONNX model file'
    )
    
    parser.add_argument(
        'platform',
        type=str,
        choices=PLATFORMS,
        help='Target RKNN platform'
    )
    
    # Optional arguments
    parser.add_argument(
        '--dtype',
        type=str,
        choices=DTYPE_CHOICES,
        default='i8',
        help='Model data type (default: i8). Use i8/u8 for quantized models, fp for floating point.'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=DEFAULT_RKNN_PATH,
        help=f'Output RKNN model path (default: {DEFAULT_RKNN_PATH})'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f'Dataset path for quantization (default: {DEFAULT_DATASET_PATH})'
    )
    
    parser.add_argument(
        '--no-quant',
        action='store_true',
        help='Disable quantization (overrides dtype setting)'
    )
    
    parser.add_argument(
        '--mean-values',
        type=float,
        nargs=3,
        default=DEFAULT_MEAN_VALUES,
        metavar=('R', 'G', 'B'),
        help=f'Mean values for normalization (default: {DEFAULT_MEAN_VALUES})'
    )
    
    parser.add_argument(
        '--std-values',
        type=float,
        nargs=3,
        default=DEFAULT_STD_VALUES,
        metavar=('R', 'G', 'B'),
        help=f'Standard deviation values for normalization (default: {DEFAULT_STD_VALUES})'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output from RKNN'
    )
    
    parser.add_argument(
        '--custom-hybrid',
        type=str,
        default=None,
        metavar='YAML_FILE',
        help='Path to YAML file containing custom hybrid quantization configuration'
    )
    
    parser.add_argument(
        '--auto-hybrid-quant',
        action='store_true',
        help='Enable automatic hybrid quantization (for older platforms: rv1109, rv1126, rk1808)'
    )
    
    args = parser.parse_args()
    
    # Validate dtype based on platform
    if args.dtype == 'u8' and args.platform not in ['rv1109', 'rv1126', 'rk1808']:
        parser.error(f"dtype 'u8' is only valid for platforms: rv1109, rv1126, rk1808")
    
    if args.dtype == 'i8' and args.platform in ['rv1109', 'rv1126', 'rk1808']:
        parser.error(f"dtype 'i8' is not valid for platform '{args.platform}'. Use 'u8' or 'fp' instead.")
    
    # Determine quantization
    if args.no_quant:
        args.do_quant = False
    else:
        args.do_quant = args.dtype in ['i8', 'u8']
    
    # Load custom hybrid quantization config if provided
    args.custom_hybrid_config = None
    if args.custom_hybrid:
        custom_hybrid_path = Path(args.custom_hybrid)
        if not custom_hybrid_path.exists():
            parser.error(f"Custom hybrid config file not found: {args.custom_hybrid}")
        
        try:
            with open(custom_hybrid_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if 'custom_hybrid' not in config:
                    parser.error(f"'custom_hybrid' key not found in {args.custom_hybrid}")
                args.custom_hybrid_config = config['custom_hybrid']
                
                # Validate custom_hybrid format
                if not isinstance(args.custom_hybrid_config, list):
                    parser.error(f"'custom_hybrid' must be a list in {args.custom_hybrid}")
        except yaml.YAMLError as e:
            parser.error(f"Failed to parse YAML file {args.custom_hybrid}: {e}")
        except Exception as e:
            parser.error(f"Failed to read custom hybrid config: {e}")
    
    # Validate hybrid quantization options
    if args.custom_hybrid and not args.do_quant:
        parser.error("Custom hybrid quantization requires quantization to be enabled (remove --no-quant)")
    
    if args.auto_hybrid_quant and not args.do_quant:
        parser.error("Auto hybrid quantization requires quantization to be enabled (remove --no-quant)")
    
    return args


def convert_onnx_to_rknn(args):
    """Convert ONNX model to RKNN format with given arguments."""
    # Process output path - if no suffix, treat as directory
    output_path = Path(args.output)
    
    if output_path.suffix == '' or output_path.is_dir():
        # No extension or is existing directory - treat as directory
        # Create directory if it doesn't exist
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f'Created output directory: {output_path}')
        
        # Extract model name from input model path and append to directory
        model_name = Path(args.model).stem  # Get filename without extension
        output_filename = f"{model_name}.rknn"
        output_path = output_path / output_filename
        logger.info(f'Output file: {output_path}')
    else:
        # Has extension - treat as file path
        # Create parent directory if it doesn't exist
        output_dir = output_path.parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f'Created output directory: {output_dir}')
    
    # Convert to string for RKNN API
    args.output = str(output_path)
    
    # Create RKNN object
    rknn = RKNN(verbose=args.verbose)

    # Pre-process config
    logger.info('Configuring model')
    logger.info(f'Platform: {args.platform}')
    logger.info(f'Data type: {args.dtype}')
    logger.info(f'Quantization: {"enabled" if args.do_quant else "disabled"}')
    logger.info(f'Mean values: {args.mean_values}')
    logger.info(f'Std values: {args.std_values}')
    
    rknn.config(
        mean_values=[args.mean_values],
        std_values=[args.std_values],
        target_platform=args.platform
    )
    logger.success('Model configuration completed')

    # Load model
    logger.info(f'Loading ONNX model: {args.model}')
    ret = rknn.load_onnx(model=args.model)
    if ret != 0:
        logger.error('Failed to load model!')
        exit(ret)
    logger.success('Model loaded successfully')

    # Build model
    logger.info('Building RKNN model')
    if args.do_quant:
        logger.info(f'Using dataset: {args.dataset}')
    
    # Choose build method based on configuration
    if args.custom_hybrid_config:
        # Use custom hybrid quantization (two-step process)
        logger.info('Using custom hybrid quantization')
        logger.info(f'Custom hybrid config: {args.custom_hybrid_config}')
        
        # Step 1: Generate intermediate files
        logger.info('Hybrid quantization step 1: Analyzing model...')
        ret = rknn.hybrid_quantization_step1(
            dataset=args.dataset,
            proposal=False,
            custom_hybrid=args.custom_hybrid_config
        )
        if ret != 0:
            logger.error('Hybrid quantization step 1 failed!')
            exit(ret)
        logger.success('Hybrid quantization step 1 completed')
        
        # Step 2: Build with generated config
        model_name = Path(args.model).stem
        model_input = f"{model_name}.model"
        data_input = f"{model_name}.data"
        quantization_cfg = f"{model_name}.quantization.cfg"
        
        logger.info('Hybrid quantization step 2: Building model...')
        logger.info(f'Using generated files: {model_input}, {data_input}, {quantization_cfg}')
        ret = rknn.hybrid_quantization_step2(
            model_input=model_input,
            data_input=data_input,
            model_quantization_cfg=quantization_cfg
        )
        if ret != 0:
            logger.error('Hybrid quantization step 2 failed!')
            exit(ret)
        logger.success('Hybrid quantization step 2 completed')
        
    elif args.auto_hybrid_quant:
        # Use automatic hybrid quantization
        logger.info('Using automatic hybrid quantization')
        ret = rknn.build(
            do_quantization=args.do_quant,
            dataset=args.dataset if args.do_quant else None,
            auto_hybrid_quant=True
        )
        if ret != 0:
            logger.error('Failed to build model!')
            exit(ret)
        logger.success('Model built successfully with auto hybrid quantization')
        
    else:
        # Use standard build
        ret = rknn.build(
            do_quantization=args.do_quant,
            dataset=args.dataset if args.do_quant else None
        )
        if ret != 0:
            logger.error('Failed to build model!')
            exit(ret)
        logger.success('Model built successfully')

    # Export rknn model
    logger.info(f'Exporting RKNN model to: {args.output}')
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        logger.error('Failed to export RKNN model!')
        exit(ret)
    logger.success('Model exported successfully')

    # Release
    rknn.release()
    logger.success(f'Successfully converted {args.model} to {args.output}')


if __name__ == '__main__':
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=''),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    args = parse_args()
    convert_onnx_to_rknn(args)